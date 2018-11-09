#!/usr/bin/env python
# encoding: utf-8
# @Date    : 2018-10-30 14:21:14
# @Author  : blhoy
# @email   : hjcai@njust.edu.cn

import tensorflow as tf
import numpy as np
from utils import *
from layer import *
from nn_utils import *

FLAGS = tf.app.flags.FLAGS

class Bi_LSTM:
    def __init__(self, n_out, name):
        """

        :param n_out: hidden size
        :param name: alias of layer
        """

        self.n_out = n_out
        self.name = name

        self.cell_fw=tf.contrib.rnn.LSTMCell(self.n_out)
        self.cell_bw=tf.contrib.rnn.LSTMCell(self.n_out)
        self.params = []

    def __str__(self):
        return "%s: LSTM(%s, %s)" % (self.name, self.n_out)

    __repr__ = __str__

    def __call__(self, x, sen_len, out_type=''):
        """

        :param x: input tensor, shape: (bs, max_len, n_in)
        :return: generated hidden states
        """
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell_fw,
            cell_bw=self.cell_bw,
            inputs=x,
            sequence_length=sen_len,
            dtype=tf.float32,
            scope=self.name
        )

        batch_size = tf.shape(x)[0]
        max_len = tf.shape(x)[1]
        if out_type == 'last':
            index = tf.range(0, batch_size) * max_len + (sen_len - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, 2 * self.n_out]), index)
        else:
            outputs = tf.concat(outputs, 2)
        return outputs

class CNN:
    def __init__(self, n_in, sen_len, kernel_size, n_filters, strides, padding, l2_reg, name, active_func=None):
        """

        :param n_in: input size
        :param sen_len: sentence length
        :param kernel_size: size of convolutional kernel
        :param n_filters: number of filters
        :param strides: size of strides
        :param padding: padding methods
        :param l2_reg: L2 coefficient
        :param name: layer alias
        """
        self.n_in = n_in
        self.sen_len = sen_len
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.strides = strides
        self.padding = padding
        self.filter_shape = [self.kernel_size, self.n_in, 1, self.n_filters]
        self.l2_reg = l2_reg
        self.name = name
        self.active_func = active_func
        self.W = tf.get_variable(
            name='%s_W' % self.name,
            shape=self.filter_shape,
            initializer=tf.random_uniform_initializer(-INIT_RANGE, INIT_RANGE),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.b = tf.get_variable(
            name='%s_b' % self.name,
            shape=[self.filter_shape[-1]],
            initializer=tf.constant_initializer(0.0),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.params = [self.W, self.b]

    def __str__(self):
        return "%s: CNN(%s, %s, kernel_size=%s)" % (self.name, self.n_in, self.n_filters, self.kernel_size)

    __repr__ = __str__

    def __call__(self, x):
        """

        :param x: input tensor, shape: (bs, sen_len, n_in)
        :return: features after pooling
        """
        _x = tf.reshape(x, [-1, self.sen_len, self.n_in, 1])
        _x = tf.nn.conv2d(_x, self.W, self.strides, self.padding) + self.b
        if self.active_func is None:
            self.active_func = tf.nn.relu
        c = self.active_func(_x)
        conv_out_pool = tf.nn.max_pool(c, ksize=[1, self.sen_len-self.kernel_size+1, 1, 1], strides=self.strides, padding=self.padding)
        return conv_out_pool

class Linear:
    """
    fully connected layer
    """
    def __init__(self, n_in, n_out, l2_reg, name, use_bias=True):
        """

        :param n_in: input size
        :param n_out: output size
        :param name: layer name
        :param l2_reg: L2 coefficient
        :param use_bias: use bias or not
        """
        self.n_in = n_in
        self.n_out = n_out
        self.l2_reg = l2_reg
        self.name = name
        self.use_bias = use_bias
        # sample weight from uniform distribution [-INIT_RANGE, INIT_RANGE]
        # initialize bias as zero vector
        self.W = tf.get_variable(
                    name="%s_W" % name,
                    shape=[n_in, n_out],
                    initializer=tf.random_uniform_initializer(-INIT_RANGE, INIT_RANGE),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
        self.b = tf.get_variable(
                    name="%s_b" % name,
                    shape=[n_out],
                    initializer=tf.constant_initializer(0.0),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
        self.params = [self.W]
        if self.use_bias:
            self.params.append(self.b)

    def __str__(self):
        return "%s: Linear(%s, %s)" % (self.name, self.n_in, self.n_out)

    __repr__ = __str__

    def __call__(self, x, bs=None):
        """

        :param x: input tensor, shape: (bs, *, n_in)
        :return: y: output tensor, shape: (bs, *, n_out)
        """
        bs = tf.shape(x)[0]
        output = tf.matmul(x, tf.tile(tf.expand_dims(self.W, 0),[bs, 1, 1]))

        if self.use_bias:
            output = output + self.b
        return output

class CPT_AS:
    # Context-Preserving Transformation with Adaptive-Scaling
    def __init__(self, sen_len, n_in, n_out, l2_reg, name):
        self.sen_len = sen_len
        self.n_in = n_in
        self.n_out = n_out
        self.name = name
        self.l2_reg = l2_reg
        self.fc_gate = Linear(n_in=self.n_in, n_out=self.n_out, l2_reg=self.l2_reg, name="Gate")
        self.fc_trans = Linear(n_in=2*self.n_in, n_out=self.n_out, l2_reg=self.l2_reg, name="Trans")
        self.layers = [self.fc_gate, self.fc_trans]
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def __str__(self):
        des_str = 'CPT(%s, %s)' % (self.n_in, self.n_out)
        for layer in self.layers:
            des_str += ', %s' % layer
        return des_str

    __repr__ = __str__

    def __call__(self, x, xt):
        """

        :param x: input sentence, shape: (bs, max_sentence_len, n_in)
        :param xt: input target, shape: (bs, max_target_len, n_in)
        :return:
        """
        batch_size = tf.shape(x)[0]
        trans_gate = tf.sigmoid(self.fc_gate(x))
        for i in xrange(self.sen_len):
            index = tf.range(0, batch_size) * self.sen_len + i
            x_i = tf.gather(tf.reshape(x, [-1, 1, self.n_in]), index)
            alpha = tf.matmul(x_i, tf.transpose(xt, perm=[0,2,1]))
            alpha = tf.nn.softmax(alpha)
            xi_new = tf.tanh(self.fc_trans(x=tf.concat([x_i, tf.matmul(alpha, xt)], 2) ) )
            if i == 0:
                x_new = xi_new
            else:
                x_new = tf.concat([x_new, xi_new], 1)
        return trans_gate * x_new + (tf.constant(1.0) - trans_gate) * x

class CPT_LF:
    # Context-Preserving Transformation with Lossless-Forwarding
    def __init__(self, sen_len, n_in, n_out, l2_reg, name):
        self.sen_len = sen_len
        self.n_in = n_in
        self.n_out = n_out
        self.name = name
        self.l2_reg = l2_reg
        self.fc_trans = Linear(n_in=2*self.n_in, n_out=self.n_out, l2_reg=self.l2_reg, name="Trans")
        self.layers = [self.fc_trans]
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def __str__(self):
        des_str = 'CPT(%s, %s)' % (self.n_in, self.n_out)
        for layer in self.layers:
            des_str += ', %s' % layer
        return des_str

    __repr__ = __str__

    def __call__(self, x, xt):
        """

        :param x: input sentence, shape: (bs, max_sentence_len, 2*n_hidden)
        :param xt: input target, shape: (bs, max_target_len, 2*n_hidden)
        :return:
        """
        batch_size = tf.shape(x)[0]
        for i in xrange(self.sen_len):
            index = tf.range(0, batch_size) * self.sen_len + i
            x_i = tf.gather(tf.reshape(x, [-1, 1, self.n_in]), index)
            alpha = tf.matmul(x_i, tf.transpose(xt, perm=[0,2,1]))
            alpha = tf.nn.softmax(alpha)
            xi_new = tf.tanh(self.fc_trans(x=tf.concat([x_i, tf.matmul(alpha, xt)], 2) ) )
            if i == 0:
                x_new = xi_new
            else:
                x_new = tf.concat([x_new, xi_new], 1)
        return x + x_new

class TNet:
    """
    Transformation Networks for Target-Oriented Sentiment Analysis
    """
    def __init__(self):
        self.embedding_dim = FLAGS.embedding_dim
        self.batch_size = FLAGS.batch_size
        self.n_filter = FLAGS.n_filter
        self.kernels = FLAGS.kernels
        self.n_hidden = FLAGS.n_hidden
        self.learning_rate = FLAGS.learning_rate
        self.n_class = FLAGS.n_class
        self.max_sentence_len = FLAGS.max_sentence_len
        self.max_target_len = FLAGS.max_target_len
        self.l2_reg = FLAGS.l2_reg
        self.n_iter = FLAGS.n_iter
        self.type = FLAGS.type

        self.word_id_mapping, self.w2v = load_w2v(FLAGS.embedding_file_path, self.embedding_dim)
        self.word_id_mapping, self.w2v = fine_tune_vocab(FLAGS.train_file_path, self.word_id_mapping, self.w2v, self.embedding_dim)
        self.word_embedding = tf.constant(self.w2v, name='word_embedding')
        # self.word_embedding = tf.Variable(self.w2v, dtype=tf.float32, name='word_embedding')

        self.ctx_LSTM = Bi_LSTM(self.n_hidden, "CTX_LSTM")
        self.tgt_LSTM = Bi_LSTM(self.n_hidden, "TGT_LSTM")
        if self.type == 'AS':
            self.CPT = CPT_AS(sen_len=self.max_sentence_len, n_in=2*self.n_hidden, n_out=2*self.n_hidden, l2_reg=self.l2_reg, name="CPT")
        else:
            self.CPT = CPT_LF(sen_len=self.max_sentence_len, n_in=2*self.n_hidden, n_out=2*self.n_hidden, l2_reg=self.l2_reg, name="CPT")

        self.Conv_layers = []
        self.Conv_layers.append(CNN(n_in=2*self.n_hidden, sen_len=self.max_sentence_len, kernel_size=self.kernels,
            n_filters=self.n_filter, strides=[1, 1, 1, 1], padding='VALID', l2_reg=self.l2_reg, name='Conv2D_'))
        self.FC = Linear(n_in=self.n_filter, n_out=self.n_class, l2_reg=self.l2_reg, name="LAST_FC")
        self.layers = [self.CPT, self.FC]
        self.layers.extend(self.Conv_layers)

        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)
        print(self.params)

        self.build_model()

    def __str__(self):
        strs = []
        for layer in self.layers:
            strs.append(str(layer))
        return ', '.join(strs)

    __repr__ = __str__

    def build_model(self):
        """
        build the computational graph of TNet
        :return:
        """

        self.dropout_keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.xt = tf.placeholder(tf.int32, [None, self.max_target_len])
            self.sen_pos = tf.placeholder(tf.float32, [None, self.max_sentence_len])
            # self.sen_pos = None
            self.y = tf.placeholder(tf.float32, [None, self.n_class])
            self.aspect_id = tf.placeholder(tf.string, None, name='aspect_id')
            self.sen_len = tf.placeholder(tf.int32, None)
            self.tar_len = tf.placeholder(tf.int32, None)

        self.input_x = tf.nn.embedding_lookup(self.word_embedding, self.x)
        self.input_x = tf.nn.dropout(self.input_x, keep_prob=self.dropout_keep_prob)
        self.input_xt = tf.nn.embedding_lookup(self.word_embedding, self.xt)
        self.input_xt = tf.nn.dropout(self.input_xt, keep_prob=self.dropout_keep_prob)

        H0 = self.ctx_LSTM(self.input_x, self.sen_len)
        Ht = self.tgt_LSTM(self.input_xt, self.tar_len)

        if self.sen_pos is not None:
            pw = tf.reshape(self.sen_pos, [-1, self.max_sentence_len, 1])
        H1 = self.CPT(H0, Ht)
        if self.sen_pos is not None:
            H1 = pw * H1
        H2 = self.CPT(H1, Ht)
        if self.sen_pos is not None:
            H2 = pw * H2
        # H3 = self.CPT(H2, Ht)
        # if self.sen_pos is not None:
        #     H3 = pw * H3

        conv_res = tf.reshape(self.Conv_layers[0](H2), [-1, 1, self.n_filter])
        conv_res = tf.nn.dropout(conv_res, keep_prob=self.dropout_keep_prob)
        self.pred = tf.reshape(self.FC(conv_res), [-1, self.n_class])

        with tf.name_scope('loss'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
            # _y = tf.split(tf.nn.softmax(self.pred), self.n_class, axis=1)
            # t1 = (_y[1] - _y[0])*(_y[1]-_y[2])
            # t2 = _y[0]+_y[2]-tf.constant(2.0)*_y[1]
            # self.cost += tf.constant(0.001) * tf.reduce_mean(t1+t2)

        with tf.name_scope('train'):
            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.cost, global_step=self.global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))

    def get_batch_data(self, x, sen_len, y, p, tgt_wd, tgt, tar_len, batch_size, keep_prob):
        for index in batch_index(len(y), batch_size, 1):
            feed_dict = {
                self.x: x[index],
                self.sen_len: sen_len[index],
                self.y: y[index],
                self.sen_pos: p[index],
                self.aspect_id: tgt_wd[index],
                self.xt: tgt[index],
                self.tar_len: tar_len[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)

    def run(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            print "# loading train and test data ..."
            tr_x, tr_sen_len, tr_y, tr_p, tr_tgt_wd, tr_tar_label, tr_tgt, tr_tar_len = load_inputs_twitter(
                FLAGS.train_file_path,
                self.word_id_mapping,
                self.max_sentence_len
            )
            te_x, te_sen_len, te_y, te_p, te_tgt_wd, te_tar_label, te_tgt, te_tar_len = load_inputs_twitter(
                FLAGS.test_file_path,
                self.word_id_mapping,
                self.max_sentence_len
            )

            init = tf.global_variables_initializer()
            sess.run(init)

            max_acc = 0.
            for i in xrange(self.n_iter):
                for train, _ in self.get_batch_data(tr_x, tr_sen_len, tr_y, tr_p, tr_tgt_wd, tr_tgt, tr_tar_len, self.batch_size, 0.7):
                    # print "######## Batch size is : ", _, ' ',self.xt
                    # print "###################", len(train[self.x]), ' ', len(train[self.xt])
                    # print "########### ",train[self.xt].shape,' ',train[self.x].shape
                    _, step = sess.run([self.optimizer, self.global_step], feed_dict=train)
                    # ggi, ggx, ggtx = sess.run([self.CPT.x_i, self.H0, self.Ht], feed_dict=train)
                    # print(ggi.shape, ggx.shape, ggtx.shape)
                    # print "#######", self.Ht.shape,' ',self.Ht[0]
                    acc, loss, cnt = 0., 0., 0
                    for test, num in self.get_batch_data(te_x, te_sen_len, te_y, te_p, te_tgt_wd, te_tgt, te_tar_len, 2000, 1.0):
                        _loss, _acc = sess.run([self.cost, self.accuracy], feed_dict=test)
                        acc += _acc
                        loss += _loss * num
                        cnt += num
                    print cnt
                    print acc
                    print 'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(step, loss / cnt, acc / cnt)
                    if acc / cnt > max_acc:
                        max_acc = acc / cnt
                    print 'iter {}: Max acc={}'.format(i, max_acc)