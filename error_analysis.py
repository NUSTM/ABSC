#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys


data_dir = 'analysis_dir'
label_name = ['1', '0', '-1']
mp = {'1':0, '0': 1, '-1':2}


def split_test_to_error_and_right(test_f, prob_f, error_f=None, right_f=None):
    if error_f is None:
        error_f = 'error.txt'
    if right_f is None:
        right_f = 'right.txt'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ep = open(os.path.join(data_dir, error_f), 'w')
    rp = open(os.path.join(data_dir, right_f), 'w')
    labels = []
    i = 0
    for line in open(test_f):
        i += 1
        if i % 3 == 1:
            tp = line.strip()
        if i % 3 == 2:
            tp = line.strip() + '||' + tp
        if i % 3 == 0:
            tp = line.strip() + '||' + tp
            labels.append(tp + '\n')
    cnt = 0
    cnt_r, cnt_e = 0., 0.
    cnt_s, cnt_m = 0., 0.
    cnt_rs, cnt_rm = 0., 0.
    for prob, line in zip(open(prob_f), labels):
        label = line.split('||')[0].strip()
        target_len = len(line.split('||')[1].split())
        if target_len > 1:
            cnt_m += 1
        else:
            cnt_s += 1
        prob = [float(item) for item in prob.split()]
        max_index = prob.index(max(prob))
        if label_name.index(label) == max_index:
            rp.write(str(cnt) + '||' + line)
            cnt_r += 1
            if target_len > 1:
                cnt_rm += 1
            else:
                cnt_rs += 1
        else:
            ep.write(str(cnt) + '||' + label_name[max_index] + '||' + line)
            cnt_e += 1
        cnt += 1
    print 'cnt = {}, right cnt = {}, error cnt = {}, acc={}'.format(cnt, cnt_r, cnt_e, cnt_r / cnt)
    print 'single:{}/{}={}, multi:{}/{}={}'.format(cnt_rs, cnt_s, cnt_rs / cnt_s, cnt_rm, cnt_m, cnt_rm / cnt_m)
    ep.close()
    rp.close()


if __name__ == '__main__':
    split_test_to_error_and_right(sys.argv[1], sys.argv[2])
