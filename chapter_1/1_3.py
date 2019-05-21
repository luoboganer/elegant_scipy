#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
 * @File    : 1_3.py
 * @Author  : shifaqiang(石发强)--[14061115@buaa.edu.cn]
 * @Data    : 2019/5/8
 * @Time    : 20:22
 * @Site    : 
 * @Software: VS-code
 * @Last Modified by: 
 * @Last Modified time: 
 * @Desc    : 
'''

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg') # for using matplotlib without GUI
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('../style/elegant.mplstyle')
import itertools
from collections import defaultdict

def kde_of_total_counts(counts):
    print("kde plotting...")
    # 手动绘制计数值分布的概率密度分布，采用核密度估计(KDE, kernel density estimation)
    total_counts=np.sum(counts,axis=0)
    density_scipy=stats.kde.gaussian_kde(total_counts)  # 该函数拟合出一个概率密度函数
    x=np.arange(min(total_counts),max(total_counts),10000)
    plt.plot(x,density_scipy(x),color='r',label="by scipy manually")
    # 使用seaborn的内置库函数绘制KDE图
    sns.kdeplot(total_counts,label="by seaborn")
    plt.legend()
    plt.show()
    plt.savefig('./1_4_kde_total_sum.png')
    print("Count Statistics:")
    print("min {}".format(total_counts.min()))
    print("max {}".format(total_counts.max()))
    print("mean {}".format(total_counts.mean()))

def reduce_xaxis_labels(ax,factor):
    '''
    show only every ith label to prevent crowding on x-axis
    e.g. factor=2 would plot every sceond x-axis label, starting at the first
    '''
    plt.setp(ax.xaxis.get_ticklabels(),visible=False)
    for label in ax.xaxis.get_ticklabels()[factor-1::factor]:
        label.set_visible(True)
    return

def box_of_total_counts(counts:np.array):
    print("box plotting...")
    # 取一个只有70列的数据子集用于绘图，原数据集shape为(20500,375)
    np.random.seed(seed=7)
    sample_index=np.random.choice(range(counts.shape[1]),size=70,replace=False)
    counts_subset=counts[:,sample_index]
    _,ax=plt.subplots(3,1,figsize=(6.8,2.4*3))
    with plt.style.context("../style/thinner.mplstyle"):
        # 原始数据
        ax[0].boxplot(counts_subset)
        ax[0].set_xlabel("Individuals")
        ax[0].set_ylabel("Gene expression counts")
        reduce_xaxis_labels(ax[0],5)
        # 对数标度的原始数据
        ax[1].boxplot(np.log(counts_subset+1))
        ax[1].set_xlabel("Individuals")
        ax[1].set_ylabel("log Gene expression counts")
        reduce_xaxis_labels(ax[1],5)
        # 库容量标准化的对数标度数据
        counts_lib_norm=counts/counts.sum(axis=0)*1e6
        counts_subset_lib_norm=counts_lib_norm[:,sample_index]
        ax[2].boxplot(np.log(counts_subset_lib_norm+1))
        ax[2].set_xlabel("Individuals")
        ax[2].set_ylabel("log Gene expression counts \n with lib normalization")
        reduce_xaxis_labels(ax[2],5)
        plt.savefig("./1_4_box_total_sum.png")    

def class_boxplot(data,classes,colors=None,*,xlabel="sample number",ylabel="log gene expression counts",filename="./1_4_comparsion_between_count_and_normalizedCount.png",**kwargs):
    """
    make a boxplot with boxes colored according to the class they belong to.
    NOTE. 注意强制关键字参数的设计
    """
    print("comparsion between raw gene expression count and normalized by library size count ...")
    _=plt.figure()
    all_classes=sorted(set(classes))
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    class2color=dict(zip(all_classes,itertools.cycle(colors)))
    class2data=defaultdict(list)
    for distribute,cls in zip(data,classes):
        for c in all_classes:
            class2data[c].append([])
        class2data[cls][-1]=distribute
    # 用适当的颜色以依次生成箱线图
    lines=[]
    for cls in all_classes:
        # 为箱型图的所有元素设定颜色
        for key in ['boxprops','whiskerprops','flierprops']:
            kwargs.setdefault(key,{}).update(color=class2color[cls])
        # 画出箱线图
        box=plt.boxplot(class2data[cls],**kwargs)
        lines.append(box['whiskers'][0])
    plt.legend(lines,all_classes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)

def rpkm(counts,length):
    """
    calculate reads per kilobase transcript per million reads
    NOTE. rpkm=(1e9*C)/(L*N)
    """
    return 1e9*counts/(length[:,np.newaxis]*np.sum(counts,axis=0))

def binned_boxplot(x,y,ax,*,xlabel='gene length (log scale)',ylabel='average log counts'):
    '''
    plot the distribution of y dependent on x using many boxplots
    Note. x and y are expected to be log-scale
    '''
    print('relation of gene length and average gene expression ...')
    # 根据观测密度定义x的分箱
    x_hist,x_bins=np.histogram(x,bins='auto')
    x_bin_idxs=np.digitize(x,x_bins[:-1])
    binned_y=[y[x_bin_idxs==i] for i in range(np.max(x_bin_idxs))]

    # 用分箱的中心作为x轴的标签
    x_bin_centers=(x_bins[1:]+x_bins[:-1])/2
    x_ticklabels=np.round(np.exp(x_bin_centers)).astype(int)
    # 生成箱线图
    ax.boxplot(binned_y,labels=x_ticklabels)
    reduce_xaxis_labels(ax,10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def mian():
    '''
    main function, entry of program
    '''

    # 导入TGCA黑色素瘤数据
    filename_TGCA="../data/counts.txt"
    tgca=pd.read_csv(filename_TGCA,index_col=0)
    print("samples of TGCA count data:")
    print(tgca.head())
    print("information of TGCA count data:")
    print(tgca.info())

    # 导入基因长度数据
    filename_gene="../data/genes.csv"
    gene_info=pd.read_csv(filename_gene,index_col=0)
    print("samples of gene info data:")
    print(gene_info.head())
    print("information of gene info data:")
    print(gene_info.info())

    print("genes in TGCA count:{}".format(tgca.shape[0]))
    print("genes in gene_info:{}".format(gene_info.shape[0]))

    # 数据匹配
    match_index=pd.Index.intersection(tgca.index,gene_info.index)
    counts=np.array(tgca.loc[match_index],dtype=int)
    gene_names=np.array(match_index)
    print(f'{counts.shape[0]} genes measured in {counts.shape[1]} individuals.')
    gene_lengths=np.array(gene_info.loc[match_index]['GeneLength'],dtype=int)
    # 最后检查对象的维度
    print("dimension of data:")
    print(counts.shape)
    print(gene_lengths.shape)


    '''样本间的标准化'''

    # 用KED生成每个独立样本基因表达的计数密度图
    kde_of_total_counts(counts)
    # 独立样本基因表达的箱线图
    box_of_total_counts(counts)
    # 比较原始数据与库标准化后的数据
    log_count_3=list(np.log(counts.T[:3]+1))
    log_n_count_3=list(np.log((counts/counts.sum(axis=0)*1e6).T[:3]+1))
    class_boxplot(log_count_3+log_n_count_3,['raw counts']*3+['normalized by library size']*3,labels=[1,2,3,1,2,3])
    
    '''基因间的标准化，样本与基因标准化:RPKM'''
    log_counts=np.log(counts/counts.sum(axis=0)*1e6+1)
    mean_log_counts=np.mean(log_counts,axis=1)
    counts_rpkm=rpkm(counts,gene_lengths)
    log_counts_rpkm=np.log(counts_rpkm/counts_rpkm.sum(axis=0)*1e6+1)
    mean_log_counts_rpkm=np.mean(log_counts_rpkm,axis=1)
    log_gene_lengths=np.log(gene_lengths)

    with plt.style.context('../style/thinner.mplstyle'):
        _,ax=plt.subplots(2,1,figsize=(8.0,3.2*2))
        binned_boxplot(x=log_gene_lengths,y=mean_log_counts,ax=ax[0])
        binned_boxplot(x=log_gene_lengths,y=mean_log_counts_rpkm,ax=ax[1],xlabel='gene length (log scale)',ylabel='average log counts with rpkm')
    plt.savefig('./1_4_box_length_of_genes.png')

    "rpkm前后两个基因表达计数的比较"
    gene_idx=np.array([15188,18959])
    gene1,gene2=gene_names[gene_idx]
    len1,len2=gene_lengths[gene_idx]
    gene_labels=['{},{}bp'.format(gene1,len1),'{},{}bp'.format(gene2,len2)]
    log_counts_2=list(np.log(counts[gene_idx]+1))
    log_rpkm_counts_2=list(np.log(counts_rpkm[gene_idx]+1))
    class_boxplot(log_counts_2+log_rpkm_counts_2,['raw counts']*2+['rpkm normalized']*2,xlabel='Genes',ylabel='log gene expression counts over all samples',filename='./1_4_comparsion_between_count_and_RPKM_normalized.png',labels=gene_labels*2)


if __name__ == '__main__':
    mian()