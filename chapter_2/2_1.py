#!/usr/bin/env python
# coding=UTF-8
'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@LastEditors: shifaqiang
@Software: Visual Studio Code
@Description: 
@Date: 2019-05-11 11:25:29
@LastEditTime: 2019-05-15 10:20:57
'''

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')

def quantile_norm(x):
    '''
    @description: normalize the column of x to have the same dsitribution
    @parameters: 
        x: 2D array of float, shape=(m,n), the input data, with m rows (genes/features) and n columns (samples)
    @return:
        ans: 2D array of float, shape=(m,n), the normalized x 
    '''
    # 计算分位数
    quantiles=np.mean(np.sort(x,axis=0),axis=1)
    # 计算每列的秩次
    ranks=np.apply_along_axis(stats.rankdata,axis=0,arr=x)
    # 秩次转化为下标
    rank_indices=ranks.astype(int)-1
    # 以秩次下标在分位数中索引得到分位数标准化的值
    return quantiles[rank_indices]

def quantile_norm_log(x):
    logx=np.log(x+1)
    normalizedLogx=quantile_norm(logx)
    return normalizedLogx

def load_data(filename):
    '''
    @description: read data from csv/txt file and return it as a numpy ndarray
    @parameters: 
    @return: 
    '''
    data_table=pd.read_csv(filename,index_col=0)
    print("data shape : {}".format(data_table.shape))
    print("example of data table:")
    print(data_table.iloc[:5,:5])
    return data_table.values

def plot_col_density(data):
    '''
    @description: for each column of data, produce a density plot over all rows 
    @parameters: 
    @return: 
    '''
    print("difference of genes expression among independent samples ...")
    log_data=np.log(data+1)
    quantile_norm_log_data=quantile_norm_log(data)
    density_per_col_before=[stats.gaussian_kde(col) for col in log_data.T]
    density_per_col_after=[stats.gaussian_kde(col) for col in quantile_norm_log_data.T]
    x=np.linspace(log_data.min(),log_data.max(),100)
    _,ax=plt.subplots(1,2,figsize=(4.8*2,3.6))
    for density in density_per_col_before:
        ax[0].plot(x,density(x))
    x=np.linspace(quantile_norm_log_data.min(),quantile_norm_log_data.max())
    for density in density_per_col_after:
        ax[1].plot(x,density(x))
    ax[0].set_xlabel("Data values (per column)")
    ax[1].set_xlabel("Data values (per column)")
    ax[0].set_ylabel("Density (with log scale)")
    ax[1].set_ylabel("Density (with log scale and quantile normalization)")
    plt.savefig('./2_1_quantile_normalization.png')

def main():
    # load data
    filename='../data/counts.txt'
    counts=load_data(filename)
    # 独立样本间的基因表达差异
    plot_col_density(counts)

if __name__ == "__main__":
    main()