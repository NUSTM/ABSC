#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from newbie_nn.nn_layer import dynamic_rnn, softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from newbie_nn.att_layer import dot_produce_attention_layer, bilinear_attention_layer, mlp_attention_layer, Mlp_attention_layer
from newbie_nn.config import *
from data_prepare.utils import load_w2v, batch_index, load_inputs_twitter
tf.app.flags.DEFINE_string('is_m', '1', 'prob')
tf.app.flags.DEFINE_string('is_r', '1', 'prob')
tf.app.flags.DEFINE_string('is_bi', '1', 'prob')
tf.app.flags.DEFINE_integer('max_target_len', 10, 'max target length')


def ian(inputs, sen_len, target, sen_len_tr, keep_prob1, keep_prob2, _id='all'):
    cell = tf.contrib.rnn.LSTMCell
    # sentence hidden
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    hiddens_s = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, FLAGS.max_sentence_len, 'sen' + _id, 'all')
    pool_s = reduce_mean_with_len(hiddens_s, sen_len)
    # target hidden
    target = tf.nn.dropout(target, keep_prob=keep_prob1)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # attention sentence
    att_s = bilinear_attention_layer(hiddens_s, pool_t, sen_len, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'sen')
    outputs_s = tf.squeeze(tf.matmul(att_s, hiddens_s))
    # attention target
    att_t = bilinear_attention_layer(hiddens_t, pool_s, sen_len_tr, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 't')
    outputs_t = tf.squeeze(tf.matmul(att_t, hiddens_t))

    outputs = tf.concat([outputs_s, outputs_t], 1)
    prob = softmax_layer(outputs, 4 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)
    return prob, att_s, att_t


def main(_):
    print_config()
    with tf.device('/gpu:1'):
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim)
        word_embedding = tf.constant(w2v, name='word_embedding')
        # word_embedding = tf.Variable(w2v, name='word_embedding')

        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
            sen_len = tf.placeholder(tf.int32, None)

            target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len = tf.placeholder(tf.int32, [None])

        inputs_s = tf.nn.embedding_lookup(word_embedding, x)
        target = tf.nn.embedding_lookup(word_embedding, target_words)
        # target = reduce_mean_with_len(target, tar_len)
        # for MLP & DOT
        # target = tf.expand_dims(target, 1)
        # batch_size = tf.shape(inputs_bw)[0]
        # target = tf.zeros([batch_size, FLAGS.max_sentence_len, FLAGS.embedding_dim]) + target
        # for BL
        # target = tf.squeeze(target)
        alpha_fw, alpha_bw = None, None
        prob, att_s, att_t = ian(inputs_s, sen_len, target, tar_len, keep_prob1, keep_prob2, FLAGS.t1)

        loss = loss_func(y, prob)
        acc_num, acc_prob = acc_func(y, prob)
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9).minimize(loss, global_step=global_step)
        # optimizer = train_func(loss, FLAGS.learning_rate, global_step)
        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prob, 1)

        title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
            FLAGS.keep_prob1,
            FLAGS.keep_prob2,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.l2_reg,
            FLAGS.max_sentence_len,
            FLAGS.embedding_dim,
            FLAGS.n_hidden,
            FLAGS.n_class
        )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        # saver = saver_func(save_dir)

        init = tf.initialize_all_variables()
        sess.run(init)

        # saver.restore(sess, '/-')

        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        tr_x, tr_sen_len, tr_target_word, tr_tar_len, tr_y = load_inputs_twitter(
            FLAGS.train_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'IAN',
            is_r,
            FLAGS.max_target_len
        )
        te_x, te_sen_len, te_target_word, te_tar_len, te_y = load_inputs_twitter(
            FLAGS.test_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'IAN',
            is_r,
            FLAGS.max_target_len
        )

        def get_batch_data(x_f, sen_len_f, yi, target, tl, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc = 0.
        max_s, max_t = None, None
        max_ty, max_py = None, None
        max_prob = None
        step = None
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_y, tr_target_word, tr_tar_len,
                                           FLAGS.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                # _, step = sess.run([optimizer, global_step], feed_dict=train)
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                # sess.run(embed_update)
            # saver.save(sess, save_dir, global_step=step)

            acc, cost, cnt = 0., 0., 0
            s, t, ty, py = [], [], [], []
            p = []
            for test, num in get_batch_data(te_x, te_sen_len, te_y,
                                            te_target_word, te_tar_len, 2000, 1.0, 1.0, False):
                _loss, _acc, _s, _t, _ty, _py, _p = sess.run(
                    [loss, acc_num, att_s, att_t, true_y, pred_y, prob], feed_dict=test)
                s += list(_s)
                t += list(_t)
                ty += list(_ty)
                py += list(_py)
                p += list(_p)
                acc += _acc
                cost += _loss * num
                cnt += num
            print 'all samples={}, correct prediction={}'.format(cnt, acc)
            acc = acc / cnt
            cost = cost / cnt
            print 'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, cost, acc)
            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
            test_summary_writer.add_summary(summary, step)
            if acc > max_acc:
                max_acc = acc
                max_s = s
                max_t = t
                max_ty = ty
                max_py = py
                max_prob = p
        P = precision_score(max_ty, max_py, average=None)
        R = recall_score(max_ty, max_py, average=None)
        F1 = f1_score(max_ty, max_py, average=None)
        print 'P:', P, 'avg=', sum(P) / FLAGS.n_class
        print 'R:', R, 'avg=', sum(R) / FLAGS.n_class
        print 'F1:', F1, 'avg=', sum(F1) / FLAGS.n_class

        fp = open(FLAGS.prob_file, 'w')
        for item in max_prob:
            fp.write(' '.join([str(it) for it in item]) + '\n')
        fp = open(FLAGS.prob_file + '_s', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_s):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        fp = open(FLAGS.prob_file + '_t', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_t):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')

        print 'Optimization Finished! Max acc={}'.format(max_acc)

        print 'Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            FLAGS.learning_rate,
            FLAGS.n_iter,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            FLAGS.l2_reg
        )


if __name__ == '__main__':
    tf.app.run()
