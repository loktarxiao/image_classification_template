#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

"""
import yaml
with open("config.yml") as fp:
    tmp_confg = yaml.load(fp)
    config_preprocessing = tmp_confg['preprocessing']
    config_augmentaion = tmp_confg['augmentation']
"""

def preprocessing_func(image, label):


    image = tf.divide((tf.cast(image, tf.float32) - 0), tf.constant(255.))
    if True:
        label = tf.one_hot(tf.cast(label, tf.int64), 2)
    return image, label


def augmentation_func(image, label):
    with tf.variable_scope("DataAugementation"):
        # random shift
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        # random color and brightness
        """
        image = tf.image.random_hue(image, 0.5)
        image = tf.image.random_saturation(image, 0.3, 0.7)
        image = tf.image.random_contrast(image, 0.4, 0.7)
        image = tf.image.random_brightness(image, 0.4)
        """

        # random crop
        image = _resize_image_shorter_edge(image, 299)
        image = tf.random_crop(image, (224, 224, 3))

        # random noise
        """
        image = tf.layers.dropout(image, rate=0.1, training=True)
        image = _gaussian_noise_layer(image, 0.3)
        image = _salt_and_pepper_noise(image, 0.01, 0.1)
        """
    return image, label


def resize_output_image(image, label):
    with tf.variable_scope("ResizeOutputImage"):
        image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

    return image, label


def _resize_image_shorter_edge(image, new_shorter_edge=299):
    with tf.variable_scope("resize_image_shorter_edge"):
        shape = tf.shape(image)
        height = shape[0]
        width = shape[1]
        height_smaller_than_width = tf.less_equal(height, width)

        new_height, new_width = control_flow_ops.cond(
            height_smaller_than_width,
            lambda: (new_shorter_edge, (width / height) * new_shorter_edge),
            lambda: (new_shorter_edge, (height / width) * new_shorter_edge))

        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, (new_height, new_width, ))
        image = tf.squeeze(image, [0])
    return image


def _gaussian_noise_layer(input_tensor, scale):
    with tf.variable_scope("gaussian_noise"):
        noise = tf.random_normal(shape=tf.shape(input_tensor),
                                 mean=0.0, stddev=0.2, dtype=tf.float32)
    return input_tensor + scale*noise


def _salt_and_pepper_noise(input_tensor, salt_ratio, pepper_ratio):
    with tf.variable_scope("SaltAndPepperNoise"):
        random_image = tf.random_uniform(shape=tf.shape(input_tensor),
                                         minval=0.0, maxval=1.0, dtype=tf.float32)
        with tf.variable_scope("salt"):
            salt_image = tf.to_float(tf.greater_equal(
                random_image, 1.0 - salt_ratio))
        with tf.variable_scope("pepper"):
            pepper_image = tf.to_float(
                tf.greater_equal(random_image, pepper_ratio))

        noised_image = tf.minimum(tf.maximum(
            input_tensor, salt_image), pepper_image)

    return noised_image
