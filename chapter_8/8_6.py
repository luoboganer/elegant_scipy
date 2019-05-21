'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-21 09:49:26
@LastEditors: shifaqiang
@LastEditTime: 2019-05-21 10:52:39
@Software: Visual Studio Code
@Description: 果蝇全基因组数据的马尔科夫模型分析
    数据来源：http://hgdownload.cse.UCSC.edu/goldenPath/dm6/bigZips/dm6.fa.gz
    解压指令：gzip -d dm6.fa.gz
'''

# packages for Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')

import itertools as it
import toolz as tz
from toolz import curried as c
import numpy as np
from glob import glob
import gzip
import os

np.set_printoptions(precision=3, suppress=True)
# 建立字典
LDICT = dict(zip('ACGTacgt', range(8)))
PDICT = {(a, b): (LDICT[a], LDICT[b]) for a, b in it.product(LDICT, LDICT)}
print(f'The dictionary is\n{PDICT}')


def is_sequence(line: str):
    return len(line) > 0 and not line.startswith('>')


def is_nucleotide(letter):
    return letter in LDICT  # 忽略N标记的未知序列


@tz.curry
def increment_model(model, index):
    model[index] += 1


def markov(seq):
    '''
    @Description: 
        get a 1st-order Markov model from a sequence of nucleotides
    '''
    model = np.zeros((8, 8))
    tz.last(
        tz.pipe(seq, c.sliding_window(2), c.map(PDICT.__getitem__),
                c.map(increment_model(model))))
    # 将计数矩阵转为概率矩阵
    model /= np.sum(model, axis=1)[:, np.newaxis]
    return model


def genome(file_pattern):
    if os.path.basename(file_pattern).split('.')[-1] == "gz":
        gzopen = tz.curry(gzip.open)
        result = tz.pipe(file_pattern, glob, sorted, c.map(gzopen(mode='rt')),
                         tz.concat, c.filter(is_sequence), tz.concat,
                         c.filter(is_nucleotide))
    else:
        result = tz.pipe(file_pattern, glob, sorted, c.map(open), tz.concat,
                         c.filter(is_sequence), tz.concat,
                         c.filter(is_nucleotide))
    return result


def plot_model(model, labels, figure=None):
    fig = figure or plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(model, cmap='magma')
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    plt.colorbar(im, cax=axcolor)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks(range(8))
        axis.set_ticks_position('none')
        axis.set_ticklabels(labels)
    plt.savefig('./8_3_markov_model.png')


if __name__ == "__main__":
    dm = '../data/dm6.fa'
    dm_gz = '../data/dm6.fa.gz'
    demo = False
    if demo:
        model = tz.pipe(dm_gz, genome, c.take(10**7), markov)
    else:
        model = tz.pipe(dm_gz, genome, markov)
    print('The model is:\n')
    print('   ', '     '.join('ACGTacgt'), '\n')
    print(model)
    print('visualization ...')
    plot_model(model, labels='ACGTacgt')


    '''
    The dictionary is
    {('A', 'A'): (0, 0), ('A', 'C'): (0, 1), ('A', 'G'): (0, 2), ('A', 'T'): (0, 3), ('A', 'a'): (0, 4), ('A', 'c'): (0, 5), ('A', 'g'): (0, 6), ('A', 't'): (0, 7), ('C', 'A'): (1, 0), ('C', 'C'): (1, 1), ('C', 'G'): (1, 2), ('C', 'T'): (1, 3), ('C', 'a'): (1, 4), ('C', 'c'): (1, 5), ('C', 'g'): (1, 6), ('C', 't'): (1, 7), ('G', 'A'): (2, 0), ('G', 'C'): (2, 1), ('G', 'G'): (2, 2), ('G', 'T'): (2, 3), ('G', 'a'): (2, 4), ('G', 'c'): (2, 5), ('G', 'g'): (2, 6), ('G', 't'): (2, 7), ('T', 'A'): (3, 0), ('T', 'C'): (3, 1), ('T', 'G'): (3, 2), ('T', 'T'): (3, 3), ('T', 'a'): (3, 4), ('T', 'c'): (3, 5), ('T', 'g'): (3, 6), ('T', 't'): (3, 7), ('a', 'A'): (4, 0), ('a', 'C'): (4, 1), ('a', 'G'): (4, 2), ('a', 'T'): (4, 3), ('a', 'a'): (4, 4), ('a', 'c'): (4, 5), ('a', 'g'): (4, 6), ('a', 't'): (4, 7), ('c', 'A'): (5, 0), ('c', 'C'): (5, 1), ('c', 'G'): (5, 2), ('c', 'T'): (5, 3), ('c', 'a'): (5, 4), ('c', 'c'): (5, 5), ('c', 'g'): (5, 6), ('c', 't'): (5, 7), ('g', 'A'): (6, 0), ('g', 'C'): (6, 1), ('g', 'G'): (6, 2), ('g', 'T'): (6, 3), ('g', 'a'): (6, 4), ('g', 'c'): (6, 5), ('g', 'g'): (6, 6), ('g', 't'): (6, 7), ('t', 'A'): (7, 0), ('t', 'C'): (7, 1), ('t', 'G'): (7, 2), ('t', 'T'): (7, 3), ('t', 'a'): (7, 4), ('t', 'c'): (7, 5), ('t', 'g'): (7, 6), ('t', 't'): (7, 7)}
    The model is:

        A     C     G     T     a     c     g     t 

    [[0.351 0.181 0.189 0.279 0.    0.    0.    0.   ]
    [0.322 0.223 0.199 0.255 0.    0.    0.    0.   ]
    [0.261 0.272 0.223 0.243 0.    0.    0.    0.   ]
    [0.216 0.194 0.239 0.351 0.    0.    0.    0.   ]
    [0.001 0.001 0.001 0.001 0.349 0.184 0.18  0.283]
    [0.001 0.001 0.001 0.001 0.327 0.211 0.185 0.273]
    [0.001 0.001 0.001 0.001 0.28  0.229 0.209 0.277]
    [0.001 0.001 0.001 0.001 0.247 0.185 0.216 0.348]]
    visualization ...

    real    2m42.068s
    user    2m42.697s
    sys     0m1.956s
    '''