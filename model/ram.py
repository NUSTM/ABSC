#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import tensorflow as tf

from nn_layer import bi_dynamic_rnn
from att_layer import dot_produce_attention_layer
from utils import load_w2v, load_aspect2id, batch_index, load_inputs_rdmn


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 25, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 20, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob1', 1.0, 'dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'dropout keep prob')


tf.app.flags.DEFINE_string('train_file_path', 'data/restaurant/rest_2014_rdmn_train.txt', 'training file')
tf.app.flags.DEFINE_string('validate_file_path', 'data/restaurant/rest_2014_rdmn_test.txt', 'validating file')
tf.app.flags.DEFINE_string('test_file_path', 'data/restaurant/rest_2014_rdmn_test.txt', 'testing file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/restaurant/rest_2014_word_embedding_300_new.txt', 'embedding file')
tf.app.flags.DEFINE_string('word_id_file_path', 'data/restaurant/word_id_new.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('aspect_id_file_path', 'data/restaurant/aspect_id_new.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('t', 'last', 'model type: ')


class RAM(object):

    def __init__(self, config):
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_hidden = config.n_hidden
        self.learning_rate = config.learning_rate
        self.n_class = config.n_class
        self.max_len = config.max_len
        self.l2_reg = config.l2_reg
        self.display_step = config.display_step
        self.n_iter = config.n_iter
        self.embedding_file = config.embedding_file_path
        self.word2id_file = config.word_id_file_path
        self.aspect_id_file = config.aspect_id_file_path
        self.train_file = config.train_file_path
        self.test_file = config.test_file_path
        self.val_file = config.validate_file_path

        self.word2id, self.w2v = load_w2v(self.embedding_file, self.embedding_dim)
        self.word_embedding = tf.constant(self.w2v, name='word_embedding')
        self.aspect2id, self.a2v = load_aspect2id(self.aspect_id_file, self.word2id, self.w2v, self.embedding_dim)
        self.aspect_embedding = tf.constant(self.a2v, name='aspect_embedding')

        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_len], name='x')
            self.y = tf.placeholder(tf.float32, [None, self.n_class], name='y')
            self.sen_len = tf.placeholder(tf.int32, None, name='sen_len')
            self.aspect_id = tf.placeholder(tf.int32, None, name='aspect_id')
            self.position = tf.placeholder(tf.int32, [None, self.max_len], name='position')

        with tf.name_scope('GRU'):
            self.w_r = tf.get_variable(
                name='W_r',
                shape=[2 * self.n_hidden + 1, self.n_hidden],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            self.u_r = tf.get_variable(
                name='U_r',
                shape=[self.n_hidden, self.n_hidden],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            self.w_z = tf.get_variable(
                name='W_z',
                shape=[2 * self.n_hidden + 1, self.n_hidden],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            self.u_z = tf.get_variable(
                name='U_z',
                shape=[self.n_hidden, self.n_hidden],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            self.w_x = tf.get_variable(
                name='W_x',
                shape=[self.n_hidden, self.n_hidden],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            self.w_g = tf.get_variable(
                name='W_g',
                shape=[2 * self.n_hidden + 1, self.n_hidden],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )

    def ram(self, inputs, aspect):
        # bi-lstm
        cell = tf.contrib.rnn.LSTMCell
        M = bi_dynamic_rnn(cell, inputs, self.n_hidden, self.sen_len, self.max_len, 'memory', 'all')
        u_t = tf.expand_dims(tf.cast(self.position, tf.float32) / tf.expand_dims(tf.cast(self.sen_len, tf.float32), -1), -1)
        w_t = 1.0 - tf.abs(u_t)
        M = tf.concat([w_t * M, u_t], 2)  # batch_size * max_len * (2 * n_hidden + 1)

        batch_size = tf.shape(M)[0]

        # Attention Layer-1
        e_0 = tf.zeros([batch_size, self.n_hidden])
        aspect = tf.reshape(aspect, [-1, 1, self.embedding_dim])
        aspect = tf.ones([batch_size, self.max_len, self.embedding_dim], dtype=tf.float32) * aspect
        e = tf.zeros([batch_size, self.max_len, self.n_hidden])
        t_M = tf.concat([M, e, aspect], 2)
        t_M_dim = 2 * self.n_hidden + 1 + self.n_hidden + self.embedding_dim
        alpha = dot_produce_attention_layer(t_M, self.sen_len, t_M_dim, self.l2_reg, scope_name='att_1')
        i_al = tf.matmul(alpha, M)  # batch_size * 1 * (2n_hidden + 1)
        i_al = tf.reshape(i_al, [batch_size, 2 * self.n_hidden + 1])

        r = tf.sigmoid(tf.matmul(i_al, self.w_r) + tf.matmul(e_0, self.u_r))  # batch_size * n_hidden
        z = tf.sigmoid(tf.matmul(i_al, self.w_z) + tf.matmul(e_0, self.u_z))  # batch_size * n_hidden
        e_t1 = tf.tanh(tf.matmul(i_al, self.w_g) + tf.matmul(r * e_0, self.w_x))  # batch_size * n_hidden
        e_1 = (1.0 - z) * e_0 + z * e_t1

        # Attention Layer-2
        e = tf.ones([batch_size, self.max_len, self.n_hidden]) * tf.reshape(e_1, [batch_size, 1, self.n_hidden])
        t_M = tf.concat([M, e, aspect], 2)
        alpha = dot_produce_attention_layer(t_M, self.sen_len, t_M_dim, self.l2_reg, scope_name='att_2')
        i_al = tf.matmul(alpha, M)
        i_al = tf.reshape(i_al, [batch_size, 2 * self.n_hidden + 1])

        r = tf.sigmoid(tf.matmul(i_al, self.w_r) + tf.matmul(e_1, self.u_r))  # batch_size * n_hidden
        z = tf.sigmoid(tf.matmul(i_al, self.w_z) + tf.matmul(e_1, self.u_z))  # batch_size * n_hidden
        e_t2 = tf.tanh(tf.matmul(i_al, self.w_g) + tf.matmul(r * e_1, self.w_x))  # batch_size * n_hidden
        e_2 = (1.0 - z) * e_1 + z * e_t2

        scores = tf.contrib.layers.fully_connected(
            inputs=e_2,
            num_outputs=self.n_class,
            # activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
            scope='softmax'
        )
        return scores

    def run(self):
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        aspect = tf.nn.embedding_lookup(self.aspect_embedding, self.aspect_id)

        prob = self.ram(inputs, aspect)

        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='softmax')
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prob, labels=self.y)) + tf.add_n(reg_loss)
            self.vars = [var for var in tf.global_variables()]

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name='tr_global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, self.vars), 5.0)
            train_op = optimizer.apply_gradients(zip(grads, self.vars), name='train_op', global_step=global_step)

        with tf.name_scope('predict'):
            cor_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(cor_pred, tf.float32))
            accuracy_num = tf.reduce_sum(tf.cast(cor_pred, tf.int32))

        tr_x, tr_y, tr_sen_len, tr_aspect, tr_position = load_inputs_rdmn(
            self.train_file,
            self.word2id,
            self.aspect2id,
            self.max_len
        )
        te_x, te_y, te_sen_len, te_aspect, te_position = load_inputs_rdmn(
            self.test_file,
            self.word2id,
            self.aspect2id,
            self.max_len
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            summary_loss = tf.summary.scalar('loss', loss)
            summary_acc = tf.summary.scalar('acc', accuracy)
            train_summary_op = tf.summary.merge([summary_loss, summary_acc])
            validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
            test_summary_op = tf.summary.merge([summary_loss, summary_acc])
            import time
            timestamp = str(int(time.time()))
            _dir = 'logs/' + str(timestamp) + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(self.l2_reg)
            train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(_dir + '/test', sess.graph)
            validate_summary_writer = tf.summary.FileWriter(_dir + '/validate', sess.graph)

            init = tf.global_variables_initializer()
            sess.run(init)

            best_accuracy = 0.
            best_iter = 0.
            for i in range(self.n_iter):
                total_loss, total_num, total_acc_num = [], [], []
                for train, l in self.get_batch_data(tr_x, tr_y, tr_sen_len, tr_aspect, tr_position, self.batch_size):
                    _, step, summary, _loss, _acc_num = sess.run([train_op, global_step, train_summary_op, loss, accuracy_num], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                    total_loss.append(_loss)
                    total_acc_num.append(_acc_num)
                    total_num.append(l)
                print '[INFO-Train] iter %s, loss = %s, acc = %s' % \
                      (i, sum(total_loss) * 1. / len(total_num), sum(total_acc_num) * 1. / sum(total_num))

                total_loss, total_acc_num, total_num = [], [], []
                for test, l in self.get_batch_data(te_x, te_y, te_sen_len, te_aspect, te_position, self.batch_size):
                    summary, step, _loss, _acc_num = sess.run([test_summary_op, global_step, loss, accuracy_num], feed_dict=test)
                    test_summary_writer.add_summary(summary, step)
                    total_loss.append(_loss)
                    total_acc_num.append(_acc_num)
                    total_num.append(l)
                test_acc = sum(total_acc_num) * 1. / sum(total_num)
                test_loss = sum(total_loss) * 1. / len(total_num)
                print '[INFO-Test] iter %s, loss = %s, acc = %s' % (i, test_loss, test_acc)
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_iter = i
                if i - best_iter > 10:
                    print 'Normal early stop!'
                    break
            print 'Best test acc = {}'.format(best_accuracy)

    def get_batch_data(self, x, y, sen_len, aspect, position, batch_size, is_shuffle=True):
        for index in batch_index(len(y), batch_size, 1, is_shuffle):
            feed_dict = {
                self.x: x[index],
                self.y: y[index],
                self.sen_len: sen_len[index],
                self.aspect_id: aspect[index],
                self.position: position[index]
            }
            yield feed_dict, len(index)


def main(_):
    ram = RAM(config=FLAGS)
    ram.run()


if __name__ == '__main__':
    tf.app.run()


