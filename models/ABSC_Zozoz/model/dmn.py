#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import sys
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf


class DeepMemoryNetwork(object):
    """
    A Deep Memory Network classification.
    Uses a linear layer, a attention layer as a hop, multiple hops are stacked.
    """
    def __init__(self, batch_size, feats_length, sentence_length, class_num,
                 vocab_size, l2_reg_lambda=0.0, depth=1, random_base=0.003):
        self.batch_size = batch_size
        self.feats_length = feats_length
        self.sentence_length = sentence_length
        self.class_num = class_num
        self.vocab_size = vocab_size
        self.l2_reg = l2_reg_lambda
        self.depth = depth
        self.content_input_x = tf.placeholder(tf.float32, [None, feats_length, sentence_length], name='content_x')
        self.content_length = tf.placeholder(tf.int32, [None], name='content_length')
        self.aspect_input_x = tf.placeholder(tf.float32, [None, feats_length, 1], name='aspect_x')
        self.input_y = tf.placeholder(tf.float32, [None, class_num], name='input_y')
        self.W0 = tf.Variable(tf.random_uniform([feats_length, feats_length], -random_base, random_base), name='W0')
        self.b0 = tf.Variable(tf.random_uniform([feats_length, 1], -random_base, random_base), name="b0")
        self.W1 = tf.Variable(tf.random_uniform([1, 2 * feats_length], -random_base, random_base), name='W1')
        self.b1 = tf.Variable(tf.constant(0.01), name='b1')
        self.W2 = tf.Variable(tf.random_uniform([feats_length, class_num], -random_base, random_base), name='softmax_w')
        self.b2 = tf.Variable(tf.random_uniform([class_num], -random_base, random_base), name="softmax_b")

    def softmax_with_len(self, inputs, length, max_len):
        inputs = tf.cast(inputs, tf.float32)
        max_axis = tf.reduce_max(inputs, -1, keep_dims=True)
        inputs = tf.exp(inputs - max_axis)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=-1, keep_dims=True) + 1e-9
        return inputs / _sum

    def run(self):
        aspect_x = self.aspect_input_x
        with tf.name_scope('ff'):
            batch_size = tf.shape(self.input_y)[0]
            # W0 = tf.zeros([batch_size, self.feats_length, self.feats_length]) + self.W0
            for i in xrange(self.depth):
                # self.linear_aspect_x = tf.batch_matmul(W0, aspect_x) + self.b0
                linear_aspect_x = aspect_x
                content_aspect_x = tf.concat([self.content_input_x, tf.zeros((self.feats_length, self.sentence_length)) + linear_aspect_x], 1)

                # New
                content_aspect_x = tf.reshape(tf.transpose(content_aspect_x, [1, 2, 0]), [2 * self.feats_length, -1])
                g = tf.tanh(tf.matmul(self.W1, content_aspect_x) + self.b1)
                g = tf.transpose(tf.reshape(g, [1, self.sentence_length, -1]), [2, 0, 1])

                # Old
                # g = tf.tanh(tf.batch_matmul(W1, content_aspect_x) + self.b1)
                tg = self.softmax_with_len(tf.reshape(g, [-1, self.sentence_length]), self.content_length, self.sentence_length)
                # tg = tf.nn.softmax(tf.reshape(g, [-1, self.sentence_length]))
                alpha = tf.reshape(tg, [-1, self.sentence_length, 1])
                self.alpha = tf.reshape(alpha, [-1, self.sentence_length])
                aspect_x = tf.matmul(self.content_input_x, alpha) + linear_aspect_x

        with tf.name_scope('softmax'):
            x = tf.reshape(aspect_x, [-1, self.feats_length])
            self.scores = tf.matmul(x, self.W2) + self.b2

            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            loss = tf.reduce_mean(loss)

            correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            self.ty = tf.argmax(self.input_y, 1)
            self.py = tf.argmax(self.scores, 1)

            return loss, acc


def fetch_corpus_2014(src_file, dest_file):
    dest_fp = open(dest_file, 'w')
    sentences = ET.parse(src_file).getroot().findall('sentence')
    for sentence in sentences:
        text = sentence.find('text').text
        for opinion in sentence.iter('aspectTerm'):
            term = opinion.attrib['term']
            polarity = opinion.attrib['polarity']
            _from = int(opinion.attrib['from'])
            _to = int(opinion.attrib['to'])
            pre = nltk.word_tokenize(text[:_from])
            for i in xrange(len(pre)):
                pre[i] = pre[i] + '/' + str(len(pre) - i)
            last = nltk.word_tokenize(text[_to:])
            for i in xrange(len(last)):
                last[i] = last[i] + '/' + str(i + 1)
            sentence = ' '.join(pre) + ' ' + ' '.join(last)
            # sentence = text[:_from] + text[_to:]
            dest_fp.write((term + '||' + polarity + '||' + sentence + '\n').encode('utf8'))


def load_embedding(embed_file):
    fp = open(embed_file)
    embed_dict = dict()
    for line in fp:
        line = line.strip().split()
        vec = []
        for v in line[1:]:
            vec.append(float(v))
        embed_dict[line[0]] = np.asarray(vec, dtype=np.float32)
    print 'load word_embedding done!'
    return embed_dict


def load_x_y(input_file, embed_dict):
    dim = len(embed_dict.values()[0])
    sentence_length = 80
    aspects, contents, content_aspects, y = [], [], [], []
    content_length = []
    fp = open(input_file)
    pos, neg, neu = 0., 0., 0.
    for line in fp:
        aspect, polarity, sentence = line.lower().strip().split('||')

        if polarity == 'positive':
            pos += 1
            y.append([1, 0, 0])
        elif polarity == 'negative':
            neg += 1
            y.append([0, 0, 1])
        elif polarity == 'neutral':
            neu += 1
            y.append([0, 1, 0])
        else:
            continue
        # aspect_x
        # aspect = nltk.word_tokenize(aspect.lower())
        aspect = aspect.lower().split()
        # print aspect
        tmp = []
        for word in aspect:
            if word in embed_dict:
                tmp.append(embed_dict[word])
        if tmp:
            aspect = np.sum(tmp, axis=0) / len(tmp)
        else:
            aspect = np.random.uniform(-0.1, 0.1, (dim,))
        aspect = np.reshape(aspect, (dim, 1))
        # print aspect
        # raw_input()
        aspects.append(aspect)
        # content_x
        words = sentence.split()
        content = np.zeros((dim, sentence_length))
        cnt = 0
        for word in words:
            word, i = word.split('/')[0], word.split('/')[-1]
            if word in embed_dict:
                content[:,cnt] = embed_dict[word] * (1.0 - float(i) / len(words))
                cnt += 1
        contents.append(content)
        content_length.append(cnt)
    print 'load x,y done!'
    print 'pos: {}, neg: {}, neu: {}, sum: {}'.format(pos/len(y), neg/len(y), neu/len(y), len(y))
    pos, neg, neu = 0., 0., 0.
    for f in y:
        if f == [1, 0, 0]:
            pos += 1
        elif f == [0, 1, 0]:
            neu += 1
        else:
            neg += 1
    print 'pos: {}, neg: {}, neu: {}, sum: {}'.format(pos/len(y), neg/len(y), neu/len(y), len(y))
    return np.asarray(aspects), np.asarray(contents), np.asarray(y), np.asarray(content_length)


if __name__ == '__main__':
    with tf.Graph().as_default():
        sess = tf.Session()
        # sess = tf.InteractiveSession()
        with sess.as_default():
            # batch_size=100, sentence_length=80
            dmn = DeepMemoryNetwork(
                batch_size=100,
                feats_length=300,
                sentence_length=80,
                class_num=3,
                vocab_size=20000,
                l2_reg_lambda=float(sys.argv[5]),
                depth=int(sys.argv[4])
            )
            loss, acc = dmn.run()
            reg_loss = tf.nn.l2_loss(dmn.W0) + tf.nn.l2_loss(dmn.W1) + tf.nn.l2_loss(dmn.W2) \
                       + tf.nn.l2_loss(dmn.b0) + tf.nn.l2_loss(dmn.b1) + tf.nn.l2_loss(dmn.b2)
            loss += reg_loss * dmn.l2_reg
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.01).minimize(loss, global_step=global_step)
            train_op = optimizer
            # optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(dmn.loss)
            # optimizer = tf.train.GradientDescentOptimizer(0.01)
            # grads_and_vars = optimizer.compute_gradients(loss)
            # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            summary_loss = tf.summary.scalar('loss', loss)
            summary_acc = tf.summary.scalar('acc', acc)
            train_summary_op = tf.summary.merge([summary_loss, summary_acc])
            train_summary_writer = tf.summary.FileWriter('logs/train/', sess.graph)
            test_summary_writer = tf.summary.FileWriter('logs/test/', sess.graph)

            sess.run(tf.global_variables_initializer())

            def train_step(aspect_x_batch, content_x_batch, y_batch, content_length):
                batch_size = 200
                for i in xrange(int(len(aspect_x_batch) / batch_size)):
                    feed_dict = {
                        dmn.aspect_input_x: aspect_x_batch[i*batch_size:(i+1)*batch_size],
                        dmn.content_input_x: content_x_batch[i*batch_size:(i+1)*batch_size],
                        dmn.input_y: y_batch[i*batch_size:(i+1)*batch_size],
                        dmn.content_length: content_length[i*batch_size:(i+1)*batch_size]
                    }
                    _, step, loss_, acc_, summary = sess.run([train_op, global_step, loss, acc, train_summary_op], feed_dict)
                    train_summary_writer.add_summary(summary, step)
                    print 'step {}, loss {}, acc {}'.format(step, loss_, acc_)

            def train_dev(aspect_x_batch, content_x_batch, y_batch, content_length):
                feed_dict = {
                    dmn.aspect_input_x: aspect_x_batch,
                    dmn.content_input_x: content_x_batch,
                    dmn.input_y: y_batch,
                    dmn.content_length: content_length
                }
                step, loss_, acc_, summary, alpha, ty, py = sess.run([global_step, loss, acc, train_summary_op, dmn.alpha, dmn.ty, dmn.py], feed_dict=feed_dict)
                test_summary_writer.add_summary(summary, step)
                print '\nTest: step {}, loss {:g}, acc {:g}\n'.format(step, loss_, acc_)
                return acc_, alpha, ty, py

            word_embedding_dict = load_embedding(sys.argv[3])
            aspect_x_batch, content_x_batch, y_batch, content_length = load_x_y(sys.argv[1], word_embedding_dict)
            aspect_x_batch_dev, content_x_batch_dev, y_batch_dev, content_length_dev = load_x_y(sys.argv[2], word_embedding_dict)
            max_acc = 0.
            max_alpha, ty, py = None, None, None
            for i in xrange(50):
                length = len(aspect_x_batch)
                indices = np.random.permutation(np.arange(length))
                train_step(aspect_x_batch[indices], content_x_batch[indices], y_batch[indices], content_length[indices])
                acc_, alpha, _ty, _py = train_dev(aspect_x_batch_dev, content_x_batch_dev, y_batch_dev, content_length_dev)
                if max_acc < acc_:
                    max_acc = acc_
                    max_alpha = alpha
                    ty = _ty
                    py = _py
            print '\n Max Acc:', max_acc
            fp = open('weight.txt', 'w')
            for y1, y2, ws in zip(ty, py, max_alpha):
                fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws]) + '\n')





