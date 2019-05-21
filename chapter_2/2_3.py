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
@Date: 2019-05-11 13:55:44
@LastEditTime: 2019-05-15 10:20:56
'''

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')
from scipy.cluster import hierarchy as hc

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

def most_variable_rows(data,*,n=1500):
    '''
    @description: find subset data to the n most variable rows in input data
    @parameters: 
    @return: the n rows of data that exhabit the most variance
    '''
    # 沿着列轴计算方差
    rowvar=data.var(axis=1)
    sorted_indices=np.argsort(rowvar)[-n:]
    return data[sorted_indices]

def bicluster(data,linkage_method='average',distance_metric="correlation"):
    '''
    @description: cluster the rows and columns of a matrix 
    @parameters: 
    @return: 
        y_rows, linkage matrix, the clustering of the rows of the input data
        y_cols, linkage matrix, the clustering of the columns of the input data
    '''
    y_rows=hc.linkage(data,method=linkage_method,metric=distance_metric)
    y_cols=hc.linkage(data.T,method=linkage_method,metric=distance_metric)
    return y_rows,y_cols

def clean_spines(axes):
    '''
    @description: erase spines of a axes
    '''
    for loc in ['left','right','top','bottom']:
        axes.spines[loc].set_visible(False)
    axes.set_xticks([])
    axes.set_yticks([])

def plot_bicluster(data,row_linkage,col_linkage,row_clusters=10,col_clusters=3):
    '''
    @description: performance a biclustering, plot a heatmap with dendrograms on each axis
    @parameters: 
    @return: 
    '''
    fig=plt.figure(figsize=(4.8,4.8))
    # row cluster merge tree
    ax1=fig.add_axes([0.09,0.1,0.2,0.6])
    threshold_r=(row_linkage[-row_clusters,2]+row_linkage[-row_clusters+1,2])/2
    with plt.rc_context({'lines.linewidth':0.75}):
        hc.dendrogram(col_linkage,orientation='left',color_threshold=threshold_r,ax=ax1)
    clean_spines(ax1)
    # column cluster merge tree
    ax2=fig.add_axes([0.3,0.71,0.6,0.2])
    threshold_c=(col_linkage[-col_clusters,2]+col_linkage[-col_clusters+1,2])/2
    with plt.rc_context({"lines.linewidth":0.75}):
        hc.dendrogram(col_linkage,color_threshold=threshold_c,ax=ax2)
    clean_spines(ax2)
    
    # data heatmap
    ax3=fig.add_axes([0.3,0.1,0.6,0.6])
    # 按照树状图的叶子节点对数据排序
    idx_rows=hc.leaves_list(row_linkage)
    data=data[idx_rows,:]
    idx_cols=hc.leaves_list(col_linkage)
    data=data[:,idx_cols]
    im=ax3.imshow(data,aspect="auto",origin='lower',cmap=plt.cm.get_cmap('YlGnBu_r'))
    clean_spines(ax3)
    # cmap 色彩映射空间
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Genes',labelpad=75)

    # 图例 legend
    axcolor=fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im,cax=axcolor)

    plt.savefig('./2_3_bicluster.png')

def main():
    # load data
    filename='../data/counts.txt'
    counts=load_data(filename)
    # 计数数据的双向可视化与簇的聚类
    counts_log=np.log(counts+1)
    counts_var=most_variable_rows(counts_log,n=1500)
    yr,yc=bicluster(counts_var,linkage_method='ward',distance_metric='euclidean')
    with plt.style.context('../style/thinner.mplstyle'):
        plot_bicluster(counts_var,yr,yc)

if __name__ == "__main__":
    main()