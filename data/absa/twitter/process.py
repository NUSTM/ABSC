#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys


def count_aspect_len(sf):
    # embedding_words = set()
    # for line in open(wf):
    #     word = line.split()[0]
    #     embedding_words.add(word)
    i = 0
    aspect_len = 0.
    single_word_cnt = 0.
    mul_word_cnt = 0.
    cnt2, cnt3, cnt4 = 0., 0., 0.
    for line in open(sf):
        i += 1
        if i % 3 == 2:
            words = line.split()
            # words = []
            # for word in ws:
            #     if word in embedding_words:
            #         words.append(word)
            aspect_len += len(words)
            if len(words) == 1:
                single_word_cnt += 1
            if len(words) > 1:
                mul_word_cnt += 1
            if len(words) == 2:
                cnt2 += 1
            if len(words) == 3:
                cnt3 += 1
            if len(words) > 2:
                cnt4 += 1
    print 'average len of target: {}/{}={}'.format(aspect_len, i / 3, aspect_len / i * 3)
    print 'single word: {}/{}={}'.format(single_word_cnt, i / 3, single_word_cnt / i * 3)
    print 'multi-word(>1): {}/{}={}'.format(mul_word_cnt, i / 3, mul_word_cnt / i * 3)
    print 'multi-word(=2): {}/{}={}'.format(cnt2, i / 3, cnt2 / i * 3)
    print 'multi-word(=3): {}/{}={}'.format(cnt3, i / 3, cnt3 / i * 3)
    print 'multi-word(>2){}/{}={}'.format(cnt4, i / 3, cnt4 / i * 3)


def fetch_words(sf, df):
    words = set()
    i = 0
    for line in open(sf):
        i += 1
        if i % 3 == 1 or i % 3 == 2:
            words |= set(line.split())
    print len(words)
    df = open(df, 'w')
    i = 1
    for word in words:
        df.write(word + '\t' + str(i) + '\n')
        i += 1


if __name__ == '__main__':
    count_aspect_len(sys.argv[1])
    # fetch_words(sys.argv[1], sys.argv[2])
