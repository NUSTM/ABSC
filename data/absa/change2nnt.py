#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import sys

f1 = open(sys.argv[1])
f2 = open(sys.argv[2], 'w')

lines = f1.readlines()
for i in range(0, len(lines), 3):
    words = lines[i].lower().split()
    targets = lines[i+1].lower().split()
    p = lines[i+2].strip()
    if p == '1':
        p = 'positive'
    elif p == '0':
        p = 'neutral'
    else:
        p = 'negative'
    for word in words:
        if word != '$t$':
            f2.write(word + ' o\n')
        else:
            f2.write(targets[0] + ' b-' + p + '\n')
            for word in targets[1:]:
                f2.write(word + ' i-' + p + '\n')
    f2.write('\n')
