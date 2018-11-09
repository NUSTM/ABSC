#!/usr/bin/env python
# encoding: utf-8
# @Author  : blhoy
# @email   : hjcai@njust.edu.cn

import numpy as np
import codecs as cs
import pickle

INIT_RANGE = 0.01

def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = range(length)
    for j in xrange(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in xrange(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]


def load_word_id_mapping(word_id_file, encoding='utf8'):
    """
    :param word_id_file: word-id mapping file path
    :param encoding: file's encoding, for changing to unicode
    :return: word-id mapping, like hello=5
    """
    word_to_id = dict()
    for line in open(word_id_file):
        line = line.decode(encoding, 'ignore').lower().split()
        word_to_id[line[0]] = int(line[1])
    print '\nload word-id mapping done!\n'
    return word_to_id


def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    id_word = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    # w2v.append(np.random.uniform(-0.01, 0.01, embedding_dim))
    cnt = 0
    for line in fp:
        line = line.decode('utf-8', 'ignore').split()
        if len(line) != embedding_dim + 1:
            # print 'a bad word embedding ...'
            print 'a bad word embedding: {}'.format(line[0]),' embedding dim: ',len(line) - 1
            continue
        if line[0] not in word_dict:
            cnt += 1
            w2v.append([float(v) for v in line[1:]])
            word_dict[line[0]] = cnt
        # word_id_file.write(line[0]+' '+str(cnt)+'\n')
        id_word[cnt] = line[0]
    w2v = np.asarray(w2v, dtype=np.float32)
    # w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print np.shape(w2v)
    # word_dict['$t$'] = (cnt + 1)
    # id_word[cnt+1] = '$t$'
    # # w2v -= np.mean(w2v, axis=0)
    # # w2v /= np.std(w2v, axis=0)
    # print word_dict['$t$'], len(w2v)
    return word_dict, w2v

def fine_tune_vocab(input_file, word_id_mapping, w2v, embedding_dim, encoding='utf8'):
    cnt = len(w2v)
    lines = open(input_file).readlines()
    for i in xrange(0, len(lines), 3):
        words = lines[i].decode(encoding).lower().split()
        for word in words:
            if word not in word_id_mapping:
                w2v = np.row_stack((w2v, np.random.uniform(-0.25, 0.25, embedding_dim)))
                word_id_mapping[word] = cnt
                cnt += 1
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    word_id_mapping['$t$'] = (cnt + 1)
    # w2v -= np.mean(w2v, axis=0)
    # w2v /= np.std(w2v, axis=0)
    return word_id_mapping, np.asarray(w2v, dtype=np.float32)

def load_aspect2id(input_file, word_id_mapping, w2v, embedding_dim):
    aspect2id = dict()
    id2aspect = dict()
    a2v = list()
    a2v.append([0.] * embedding_dim)
    cnt = 0
    for line in open(input_file):
        line = line.lower().split()
        cnt += 1
        aspect2id[' '.join(line[:-1])] = cnt
        id2aspect[cnt] = ' '.join(line[:-1])
        tmp = []
        for word in line:
            if word in word_id_mapping:
                tmp.append(w2v[word_id_mapping[word]])
        if tmp:
            a2v.append(np.sum(tmp, axis=0) / len(tmp))
        else:
            a2v.append(np.random.uniform(-0.01, 0.01, (embedding_dim,)))
    print len(aspect2id), len(a2v)
    return id2aspect, aspect2id, np.asarray(a2v, dtype=np.float32)

def change_y_to_onehot(y):
    from collections import Counter
    print Counter(y)
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    print y_onehot_mapping
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)

def change_y_d_to_onehot(y_domain):
    onehot = []
    for label in y_domain:
        tmp = [0] * 2
        tmp[label] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)

def load_inputs_twitter(input_file, word_id_file, sentence_len, C = 30.0, type_='', is_r=True, target_len=9, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print 'load word-to-id done!'

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    print_target_words = []
    tar_len = []
    tar_label = []
    pos = []
    tag = [word_to_id['$t$']]
    lines = open(input_file).readlines()
    for i in xrange(0, len(lines), 3):
        target_begin = target_end = sen_len_p = 0
        tmp = lines[i + 1].decode(encoding).lower().strip()
        words = tmp.split()
        # target_word = map(lambda w: word_to_id.get(w, 0), target_word)
        # target_words.append([target_word[0]])
        print_target_words.append(tmp)

        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        y.append(lines[i + 2].strip().split()[0])

        words = lines[i].decode(encoding).lower().split()

        cnt = 0
        for word in words:
            if word == '$t$':
                target_begin = cnt
                target_end = cnt + l - 1
            if word in word_to_id:
                cnt += 1
                sen_len_p += 1
        sen_len_p += l

        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
        words = words_l + target_word + words_r
        words = words[:sentence_len]
        sen_len.append(len(words))
        x.append(words + [0] * (sentence_len - len(words)))

        pos_t = []
        for index in xrange(len(x[-1])):
            #get position weights
            weight = 0.0
            if cnt < target_end:
                weight = 1.0 - (target_end - cnt) / C
            elif target_end <= cnt and cnt <= sen_len_p:
                weight = 1.0 - (cnt - target_begin) / C
            pos_t.append(weight)
        pos_t = pos_t[:sentence_len]
        pos.append(pos_t + [0] * (sentence_len - len(pos_t)))

        tar_lab = [0] * len(words_l) + [1] * len(target_word)+ [0] * (sentence_len - len(words_l) - len(target_word))
        tar_label.append(tar_lab)
    y = change_y_to_onehot(y)
    return np.asarray(x), np.asarray(sen_len), np.asarray(y), np.asarray(pos), \
        np.asarray(print_target_words), np.asarray(tar_label), np.asarray(target_words), np.asarray(tar_len)

def data_analysis(tr_target, te_target):
    tar_len = [0] * 25
    tr_tar_len = [0] * 25
    te_tar_len = [0] * 25
    for target in tr_target:
        t_len = len(target.split())
        tar_len[t_len-1] += 1
        tr_tar_len[t_len-1] += 1

    for target in te_target:
        t_len = len(target.split())
        tar_len[t_len-1] += 1
        te_tar_len[t_len-1] += 1
    return tar_len, tr_tar_len, te_tar_len

def data_statistic(tar_len, tr_tar_len, te_tar_len):
    print "Statistic Information : "
    print "{:^14}".format("target len"),
    for i in xrange(25):
        print "{:^5}".format(i+1),
    print "\n{:^14}".format("count"),
    for ele in tar_len:
        print "{:^5}".format(ele),
    print "\n{:^14}".format("percent"),
    for ele in tar_len:
        print "{:^.2%}".format((ele+.0)/sum(tar_len)),
    print "\n\n"

    print "{:^14}".format("train tar len"),
    for i in xrange(25):
        print "{:^5}".format(i+1),
    print "\n{:^14}".format("count"),
    for ele in tr_tar_len:
        print "{:^5}".format(ele),
    print "\n{:^14}".format("percent"),
    for ele in tr_tar_len:
        print "{:^.2%}".format((ele+.0)/sum(tr_tar_len)),
    print "\n\n"

    print "{:^14}".format("test tar len"),
    for i in xrange(25):
        print "{:^5}".format(i+1),
    print "\n{:^14}".format("count"),
    for ele in te_tar_len:
        print "{:^5}".format(ele),
    print "\n{:^14}".format("percent"),
    for ele in te_tar_len:
        print "{:^.2%}".format((ele+.0)/sum(te_tar_len)),
    print "\n\n"