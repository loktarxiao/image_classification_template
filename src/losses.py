#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 02:29:50 2018

@author: loktar
"""
from __future__ import print_function
import tensorflow as tf

def softmax_cross_entropy(y_true, y_pred, label_smoothing=0):
    with tf.variable_scope("softmax_cross_entropy"):
        loss = tf.losses.softmax_cross_entropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss

def get_loss(loss_name):
    loss_lst = {
                'softmax_cross_entropy': softmax_cross_entropy
                }
    return loss_lst[loss_name]

if __name__ == '__main__':
    y = tf.constant([1., 1., 1., 1., 0., 0.])
    y_ = tf.constant([0.3, 0.0, 0.7, 1, 0.1, 0.9])

    loss = get_loss('softmax_cross_entropy')(y, y_)
    sess = tf.Session()
    print("loss is", sess.run(loss))
    sess.close()