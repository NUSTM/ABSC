#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-30 14:22:00
# @Author  : blhoy
# @email   : hjcai@njust.edu.cn

import os, sys
sys.path.append(os.getcwd())

import numpy as np
import codecs as cs
import tensorflow as tf
from layer import TNet

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 64, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_filter', 100, 'number of convolutional filters')
tf.app.flags.DEFINE_integer('kernels', 3, 'number of convolutional kernels')
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.002, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 84, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_target_len', 9, 'max number of tokens per target')
tf.app.flags.DEFINE_float('l2_reg', 0.0001, 'l2 regularization')
tf.app.flags.DEFINE_integer('n_iter', 40, 'number of train iter')

tf.app.flags.DEFINE_string('train_file_path', 'data/absa/laptop/laptop_2014_train.txt', 'training file')
# tf.app.flags.DEFINE_string('train_file_path', 'data/absa/restaurant/rest_2014_train.txt', 'training file')
# tf.app.flags.DEFINE_string('train_file_path', 'data/absa/twitter/train_new.txt', 'training file')

tf.app.flags.DEFINE_string('test_file_path', 'data/absa/laptop/laptop_2014_test.txt', 'testing file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/absa/restaurant/rest_2014_test.txt', 'testing file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/absa/twitter/test.txt', 'testing file')

# tf.app.flags.DEFINE_string('embedding_file_path', 'data/absa/laptop/laptop_word_embedding_42b.txt', 'embedding file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/absa/restaurant/rest_2014_lstm_word_embedding_42b_300.txt', 'embedding file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/absa/twitter/twitter_word_embedding_partial_100.txt', 'embedding file')
# tf.app.flags.DEFINE_string('embedding_file_path', '/home/hjcai/1080/hjcai/glove/glove.840B.300d.txt', 'embedding file')
tf.app.flags.DEFINE_string('embedding_file_path', '/home/hjcai/1080/hjcai/glove/laptop_2014_embedding_840B.300d.txt', 'embedding file')

tf.app.flags.DEFINE_string('aspect_id_file_path', 'data/absa/laptop/laptop_train.txt', 'word-id mapping file')
# tf.app.flags.DEFINE_string('aspect_id_file_path', 'data/absa/restaurant/rest_train.txt', 'word-id mapping file')
# tf.app.flags.DEFINE_string('aspect_id_file_path', 'data/absa/twitter/twitter_train.txt', 'word-id mapping file')

tf.app.flags.DEFINE_string('type', 'AS', 'model type: ''(default), LF or AS')

def main(_):
    tnet = TNet()
    tnet.run()


if __name__ == '__main__':
    tf.app.run()