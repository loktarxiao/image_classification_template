#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 02:29:50 2018

@author: loktar
"""

from __future__ import print_function
import sys
sys.dont_write_bytecode = True

import tensorflow as tf


def _RMSProp(lr, momentum=0, decay=0.9, epsilon=1e-7):
    return tf.train.RMSPropOptimizer(lr, decay=decay,
                                     momentum=momentum,
                                     epsilon=epsilon)
def _Momentum(lr, momentum=0):
    return tf.train.MomentumOptimizer(lr, momentum=momentum)

def get_optimizer(optimizer_name):
    optimizer_lst = {
                        "rmsprop":   _RMSProp,
                        "momentum":  _Momentum
                    }
        return optimizer_lst[optimizer_name]

def _exponential_decay(start_lr, lr, global_step, decay_steps, decay_rate, staircase=False):
    return tf.train.exponential_decay(start_lr, learning_rate, global_step, )
