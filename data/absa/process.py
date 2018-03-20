#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import sys

def change_3_to_1(f1, f2):
    f1 = open(f1).readlines()
    f2 = open(f2, 'w')
    for i in xrange(0, len(f1), 3):
        f1[i] = f1[i].replace('$T$', f1[i + 1].strip())
        f2.write(f1[i + 1].strip() + ' || ' + f1[i].strip() + ' || ' + f1[i+2].strip() + '\n')


def change_to_dmn(f1, f2):
    f1 = open(f1).readlines()
    f2 = open(f2, 'w')
    for i in xrange(0, len(f1), 3):
        tmp = ''
        words = f1[i].lower().split()
        ind = words.index('$t$')
        cnt = 0
        for word in words[:ind]:
            word = word.split('/')[0]
            tmp += (word + '/' + str(ind - cnt) + ' ')
            cnt += 1
        cnt = 1
        for word in words[ind+1:]:
            word = word.split('/')[0]
            tmp += (word + '/' + str(cnt) + ' ')
        target = f1[i+1].strip()
        y = f1[i+2].strip()
        if y == '-1':
            y = 'negative'
        elif y == '0':
            y = 'neutral'
        else:
            y = 'positive'
        f2.write(target + '||' + y + '||' + tmp + '\n')


def data_wash(sf, df):
    dp = open(df, 'w')
    i = 0
    for line in open(sf):
        i += 1
        if i % 3 == 0:
            dp.write(line)
            continue
        words = line.split()
        new_words = []
        for word in words:
            word = word.strip('-')
            if word == "n't":
                word = 'not'
            if word == "'s":
                word = 'is'
            if word == "'re":
                word = 'are'
            for num in range(0, 10):
                if str(num) in word:
                    word = 'NUMBER'
                    break
            new_words.append(word)
        dp.write(' '.join(new_words) + '\n')


if __name__ == '__main__':
    # change_3_to_1(sys.argv[1], sys.argv[2])
    change_to_dmn(sys.argv[1], sys.argv[2])
    # data_wash(sys.argv[1], sys.argv[2])
