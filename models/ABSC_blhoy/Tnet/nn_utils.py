#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-31 20:52:32
# @Author  : blhoy
# @email   : hjcai@njust.edu.cn

import tensorflow as tf
import numpy as np

def softmax(inputs, length, max_length):
    inputs = tf.cast(inputs, tf.float32)
    max_axis = tf.reduce_max(inputs, 2, keepdims=True)
    inputs = tf.exp(inputs)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
    inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=2, keepdims=True) + 1e-9
    return inputs / _sum

def hard_sigmoid(x):
    """
    An approximation of sigmoid.
    Approx in 3 parts: 0, scaled linear, 1.
    """
    slope = tf.constant(0.2)
    shift = tf.constant(0.5)
    x = (x * slope) + shift
    x = tf.clip_by_value(x, 0, 1)
    return x