#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 25, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 10, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob1', 1.0, 'dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'dropout keep prob')
tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
# tf.app.flags.DEFINE_string('method', 'bilstm_bilstm', 'type of hidden output')
tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')


tf.app.flags.DEFINE_string('train_file_path', 'data/restaurant/rest_2014_lstm_train_new.txt', 'training file')
tf.app.flags.DEFINE_string('validate_file_path', 'data/restaurant/rest_2014_lstm_test_new.txt', 'validating file')
tf.app.flags.DEFINE_string('test_file_path', 'data/restaurant/rest_2014_lstm_test_new.txt', 'testing file')
tf.app.flags.DEFINE_string('train_file_path_r', 'data/restaurant/rest_2014_lstm_train_new.txt', 'training file')
tf.app.flags.DEFINE_string('validate_file_path_r', 'data/restaurant/rest_2014_lstm_test_new.txt', 'validating file')
tf.app.flags.DEFINE_string('test_file_path_r', 'data/restaurant/rest_2014_lstm_test_new.txt', 'testing file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/restaurant/rest_2014_word_embedding_300_new.txt', 'embedding file')
tf.app.flags.DEFINE_string('word_id_file_path', 'data/restaurant/word_id_new.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('aspect_id_file_path', 'data/restaurant/aspect_id_new.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('prob_file', 'prob1.txt', 'prob')
tf.app.flags.DEFINE_string('saver_file', 'prob1.txt', 'prob')
tf.app.flags.DEFINE_string('model_num', '100', 'prob')


def print_config():
    FLAGS._parse_flags()
    print '\nParameters:'
    for k, v in sorted(FLAGS.__flags.items()):
        print '{}={}'.format(k, v)
    print


def loss_func(y, prob):
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prob, y))
    loss = - tf.reduce_mean(y * tf.log(prob)) + sum(reg_loss)
    return loss


def acc_func(y, prob):
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
    acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc_num, acc_prob


def train_func(loss, r, global_step, optimizer=None):
    # global_step = tf.Variable(0, name="tr_global_step", trainable=False)
    if optimizer:
        return optimizer(learning_rate=r).minimize(loss, global_step=global_step)
    else:
        return tf.train.AdamOptimizer(learning_rate=r).minimize(loss, global_step=global_step)
        # optimizer = tf.train.AdagradOptimizer(learning_rate=r).minimize(loss, global_step=global_step)


def summary_func(loss, acc, test_loss, test_acc, _dir, title, sess):
    summary_loss = tf.summary.scalar('loss' + title, loss)
    summary_acc = tf.summary.scalar('acc' + title, acc)
    test_summary_loss = tf.summary.scalar('loss' + title, test_loss)
    test_summary_acc = tf.summary.scalar('acc' + title, test_acc)
    train_summary_op = tf.summary.merge([summary_loss, summary_acc])
    validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
    test_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc])
    train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(_dir + '/test')
    validate_summary_writer = tf.summary.FileWriter(_dir + '/validate')
    return train_summary_op, test_summary_op, validate_summary_op, \
        train_summary_writer, test_summary_writer, validate_summary_writer


def saver_func(_dir):
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000)
    import os
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return saver






