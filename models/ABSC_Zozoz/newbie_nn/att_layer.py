#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import numpy as np
import tensorflow as tf


def softmax_with_len(inputs, length, max_len):
    inputs = tf.cast(inputs, tf.float32)
    # max_axis = tf.reduce_max(inputs, -1, keep_dims=True)
    # inputs = tf.exp(inputs - max_axis)
    inputs = tf.exp(inputs)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=-1, keep_dims=True) + 1e-9
    return inputs / _sum


def bilinear_attention_layer(inputs, attend, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param attend: batch * n_hidden
    :param length:
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id:
    :return:
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, w), [-1, max_len, n_hidden])
    attend = tf.expand_dims(attend, 2)
    tmp = tf.reshape(tf.matmul(tmp, attend), [batch_size, 1, max_len])
    # M = tf.expand_dims(tf.matmul(attend, w), 2)
    # tmp = tf.reshape(tf.batch_matmul(inputs, M), [batch_size, 1, max_len])
    return softmax_with_len(tmp, length, max_len)


def dot_produce_attention_layer(inputs, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    u = tf.get_variable(
        name='att_u_' + str(layer_id),
        shape=[n_hidden, 1],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + 1))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + 1)), np.sqrt(6.0 / (n_hidden + 1))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha


def mlp_attention_layer(inputs, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[n_hidden],
        # initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-0., 0.),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    u = tf.get_variable(
        name='att_u_' + str(layer_id),
        shape=[n_hidden, 1],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + 1))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + 1)), np.sqrt(6.0 / (n_hidden + 1))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.tanh(tf.matmul(inputs, w) + b)
    tmp = tf.reshape(tf.matmul(tmp, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha


def Mlp_attention_layer(inputs, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    u = tf.get_variable(
        name='att_u_' + str(layer_id),
        shape=[n_hidden, 1],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + 1))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + 1)), np.sqrt(6.0 / (n_hidden + 1))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.transpose(tf.reshape(inputs, [-1, n_hidden]), [1, 0])
    tmp = tf.transpose(tf.tanh(tf.matmul(w, inputs)), [1, 0])
    tmp = tf.reshape(tf.matmul(tmp, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha
