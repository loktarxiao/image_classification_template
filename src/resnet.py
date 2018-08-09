#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 02:29:50 2018

@author: loktar
"""

import tensorflow as tf


class ResNet_v2(object):
    """ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """

    def __init__(self, input_tensor, block, layers_lst, channels_lst, classes=1000, thumbnail=False, training=False):
        assert len(layers_lst) == len(channels_lst) - 1
        self.input = input_tensor
        self.block = block
        self.layers_lst = layers_lst
        self.channels_lst = channels_lst
        self.classes = classes
        self.thumbnail = thumbnail
        self.training = training

    def build_model(self):
        self.features = self.build_feature()
        self.outputs = self.build_classifier(self.features)


    def build_feature(self):
        with tf.variable_scope("Features"):
            outputs = self._start_block()
            in_channels = self.channels_lst[0]
            for i, num_layer in enumerate(self.layers_lst):
                strides = (1, 1) if i == 0 else (2, 2)
                outputs = self._make_residual_block(outputs, self.block, num_layer,
                                                    self.channels_lst[i+1], strides, i+1, in_channels=in_channels)
                in_channels = self.channels_lst[i+1]
            outputs = tf.layers.batch_normalization(outputs, training=self.training)
            outputs = tf.nn.relu(outputs)

        return outputs
    
    def build_classifier(self, features):
        with tf.variable_scope("Classifier"):
            outputs = tf.reduce_mean(features, reduction_indices=[1, 2], name="avg_pool")
            outputs = tf.layers.flatten(outputs, name='flatten')
            outputs = tf.layers.dense(outputs, self.classes, name="outputs")
            self.probability = tf.nn.softmax(outputs)

        return outputs



    def _make_residual_block(self, input_tensor, block, layers, channels, strides, stage_index, in_channels=0):
        if block == "basic_block":
            block = self._basic_block
        elif block == "bottle_neck":
            block = self._bottle_neck

        with tf.variable_scope('stage%d'%stage_index):
            output = block(input_tensor, channels, strides, channels != in_channels, name="stage_%d_0"%stage_index)
            for i in range(layers - 1):
                output = block(output, channels, 1, False, name="stage_{}_{}".format(stage_index, str(i+1)))

        return output 

    # block setting
    def _start_block(self):
        with tf.variable_scope("start_block"):
            output = tf.layers.batch_normalization(
                self.input, training=self.training, scale=False, center=False)
            if self.thumbnail:
                output = tf.layers.conv2d(output, self.channels_lst[0], 3, padding='SAME', use_bias=False)
            else:
                output = tf.layers.conv2d(output, self.channels_lst[0], 7, strides=(2, 2), use_bias=False)
                output = tf.layers.batch_normalization(output, training=self.training)
                output = tf.nn.relu(output)
                output = tf.layers.max_pooling2d(output, 3, 2, padding='SAME')
        
        return output

    def _basic_block(self, input_tensor, channels, strides, downsample=False, name='Untitled'):
        """BasicBlock V2 from
        `"Identity Mappings in Deep Residual Networks"
        <https://arxiv.org/abs/1603.05027>`_ paper.
        This is used for ResNet V2 for 18, 34 layers.

        Parameters
        ----------
        channels : int
            Number of output channels.
        stride : int
            Stride size.
        downsample : bool, default False
            Whether to downsample the input.
        in_channels : int, default 0
            Number of input channels. Default is 0, to infer from the graph.
        """
        with tf.variable_scope("BasicBLock_{}".format(name)):
            residual = input_tensor
            out = tf.layers.batch_normalization(input_tensor, training=self.training)
            out = tf.nn.relu(out)
            if downsample:
                residual = tf.layers.conv2d(out, channels, 1, strides=strides ,use_bias=False)
            out = tf.layers.conv2d(out, channels, 1, strides=strides, use_bias=False)

            out = tf.layers.batch_normalization(out, training=self.training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, channels, 3, padding='SAME', use_bias=False)

        return out + residual

    def _bottle_neck(self, input_tensor, channels, strides, downsample=False, name='Untitled'):
        """Bottleneck V2 from
        `"Identity Mappings in Deep Residual Networks"
        <https://arxiv.org/abs/1603.05027>`_ paper.
        This is used for ResNet V2 for 50, 101, 152 layers.

        Parameters
        ----------
        channels : int
            Number of output channels.
        stride : int
            Stride size.
        downsample : bool, default False
            Whether to downsample the input.
        in_channels : int, default 0
            Number of input channels. Default is 0, to infer from the graph.
        """
        with tf.variable_scope("BottleNeck_{}".format(name)):
            residual = input_tensor
            out = tf.layers.batch_normalization(input_tensor, training=self.training)
            out = tf.nn.relu(out)
            if downsample:
                residual = tf.layers.conv2d(out, channels, 1, strides=strides ,use_bias=False)
            out = tf.layers.conv2d(out, channels//4, 1, padding='SAME', use_bias=False)

            out = tf.layers.batch_normalization(out, training=self.training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, channels//4, 3, strides=strides, padding='SAME', use_bias=False)

            out = tf.layers.batch_normalization(out, training=self.training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, channels, 1, use_bias=False)

            return out + residual


def get_resnet(model_name, device="cpu:0", num_classes=1000, training=False, thumbnail=False):

    resnet_spec = {'resnet-18': ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
                   'resnet-34': ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
                   'resnet-50': ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
                   'resnet-101': ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
                   'resnet-152': ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}
    
    def resnet(input_tensor, config=resnet_spec[model_name]):
        with tf.device('/{}'.format(device)):
            net = ResNet_v2(input_tensor, config[0], config[1], config[2], num_classes, training=training, thumbnail=thumbnail)
        return net

    return resnet
    

if __name__ == '__main__':
    input_tensor = tf.random_normal((12, 224, 224, 3), dtype=tf.float32)
    resnet = get_resnet('resnet-152', num_classes=2, training=True, thumbnail=False)(input_tensor)

    resnet.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    a, b = sess.run([resnet.features, resnet.probability])
    print(a.shape, b.shape)
    print(b)
    writer = tf.summary.FileWriter('test', sess.graph)
    sess.close()
    writer.close()
