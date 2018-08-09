#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 02:29:50 2018

@author: loktar
"""

from __future__ import print_function
import sys
import os
sys.dont_write_bytecode = True

import tensorflow as tf
from data_io import data_loader
from resnet import get_resnet
from losses import get_loss
from metrics import categorical_accuracy
from visualization import TensorBoard

class base_trainer(object):
    """Trainer module for most projects
    - data loader
    - model loader
    - restore model or initialization
    - setting
        - loss
        - summary
        - metrics
        - tensorboard
    - train_build
        - train_loop
        - train_step
    - run
    """

    def __init__(self):
        self.label = None
        self.pred = None
        self.loss = None
        self.saver = None
        self.save_path = './'
        self.sess = None
        self._set_placeholder()

    def _load_data(self):
        """loading data
        """
        config_data_loader = {'input_handle': '/data/dogs_vs_cats/train',
                              'mode': 'folder',
                              'num_repeat': 1,
                              'shuffle': True,
                              'batch_size': 32,
                              'num_processors': 4,
                              'augmentation': True,
                              'name': 'train_dataloader'}
        with tf.variable_scope('DataLoader'):
            self.train_iterator = data_loader(**config_data_loader)
            self.train_loader = self.train_iterator.get_next()

    
    def _set_placeholder(self):
        """seting placeholder
        """
        self.input = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='input')
        self.label = tf.placeholder(dtype=tf.float32, name='label')
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, name='lr_placeholder')

    def _load_model(self, network=None):
        """loading model
        """
        self.net = get_resnet('resnet-18', num_classes=2, training=True)(self.input)
        self.net.build_model()
        self.pred = self.net.outputs
        self.tvars = tf.trainable_variables()

    def _set_loss(self):
        self.loss = get_loss('softmax_cross_entropy')(self.label, self.pred)
        self.grads = tf.gradients(self.loss, self.tvars)
    def _set_metrics(self):
        self.accuracy = categorical_accuracy(self.label, self.pred)
        self.metrics_lst = [self.accuracy]
    
    def _set_optimizer(self):
        self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.lr_placeholder, momentum=0.0)
        
        self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.tvars))

    def _set_tensorboard(self):
        summary_tool = TensorBoard()
        summary_tool.scalar_summary("all_loss", self.loss)
        summary_tool.scalar_summary("accuracy", self.accuracy)
        self.summary_ops = summary_tool.merge_all_summary()

    def _set_session(self):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=sess_config)

    def _init_or_restore(self):
        model_file = tf.train.latest_checkpoint(self.save_path)
        try:
            self.saver.restore(self.sess, model_file)
            print('Restore Sucessful!')
        except:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            print('Restore Failed!')

    
    def build(self):
        with tf.device('/cpu:0'):
            self._load_data()
        with tf.device('/cpu:0'):
            self._set_placeholder()
            self._load_model()
            self._set_loss()
            self._set_metrics()
            self._set_optimizer()
        self._set_session()
        self._init_or_restore()


    def train_loop(self):
        steps_per_epoch = 300
        self.sess.run(self.train_iterator.initializer)
        for self.step in range(steps_per_epoch):
            self.train_step()
        


    def train_step(self):
        try:
            image_batch, label_batch = self.sess.run(self.train_loader)
        except tf.errors.OutOfRangeError:
            self.sess.run(self.train_iterator.initializer)

        train_feed = {self.input: image_batch,
                      self.label: label_batch,
                      self.lr_placeholder: 0.001}
        
        run_lst = [self.train_op, self.loss, self.accuracy]
        _, loss, accuracy = self.sess.run(run_lst, feed_dict=train_feed)
        print("[epoch {}] loss={:.4f} accuracy={:.4f}".format(self.epoch, loss, float(accuracy)))

        # train_feed = {self.model.}

    def run(self):
        for self.epoch in range(2):
            self.train_loop()
        

if __name__ == '__main__':
    trainer = base_trainer()
    trainer.build()
    trainer.run()
