#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
 * @File    : 1_1.py.py
 * @Author  : shifaqiang(石发强)--[14061115@buaa.edu.cn]
 * @Data    : 2019/4/16
 * @Time    : 20:22
 * @Site    : 
 * @Software: PyCharm
 * @Last Modified by: 
 * @Last Modified time: 
 * @Desc    : 
'''

import os
import sys
import numpy as np


def rpkm(counts, lengths):
    N = np.sum(counts, axis=0)
    L = lengths
    C = counts

    normed = 1e9*C/(N[np.newaxis, :]*L[:, np.newaxis])

    return normed


def mian():
    '''
    main function, entry of program
    '''
    print("hello world!")


if __name__ == '__main__':
    mian()

    # some basic method
    x=np.array([1,2,3,4])
    a=x.reshape((len(x),1))
    b=x.reshape(1,len(x))
    c=a*b
    print(a)
    print(b)
    print(a.shape,b.shape)
    print(c)
    print(c.shape)