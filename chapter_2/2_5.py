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
@Date: 2019-05-11 18:02:25
@LastEditTime: 2019-05-15 10:21:07
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
    return data_table.values,data_table.columns

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
    
def survival_distribution_function(lifetimes:np.ndarray,right_censored=None):
    '''
    @description: return the survival distribution function of a set of lifetimes
    @parameters: 
        lifetimes, the observed lifetimes of a population
        right_censored, a value of 'True' here indicates that this lifetime was not observed,
            values of np.nan in lifetimes are also considered to be right_censored
    @return: 
    '''
    n_obs=len(lifetimes)
    rc=np.isnan(lifetimes)
    if right_censored is not None:
        rc|=right_censored
    observed=lifetimes[~rc]
    xs=np.concatenate(([0],np.sort(observed)))
    ys=np.linspace(1,0,n_obs+1)
    ys=ys[:len(xs)]
    return xs,ys

def plot_cluster_survival_curves(clusters,sample_names,patients,right_censored=None):
    _,ax=plt.subplots()
    if type(clusters) == np.ndarray :
        cluster_ids=np.unique(clusters)
        cluster_name=['cluster {}'.format(i) for i in cluster_ids]
    elif type(clusters) == pd.Series:
        cluster_ids=clusters.cat.categories
        cluster_name=list(cluster_ids)
    n_clusters=len(clusters)
    for c in cluster_ids:
        clust_samples=np.flatnonzero(clusters==c)
        # 去除不存在存活数据中的患者
        clust_samples=[sample_names[i] for i in clust_samples if sample_names[i] in patients.index]
        patient_clusters=patients.loc[clust_samples]
        survival_time=patient_clusters['melanoma-survival-time'].values
        if right_censored:
            censored=~patient_clusters['melanoma-dead'].values.astype(bool)
        else:
            censored=None
        stimes,sfracs=survival_distribution_function(survival_time,censored)
        ax.plot(stimes/365,sfracs)
    ax.set_xlabel('survival time (years)')
    ax.set_ylabel('fraction alive')
    ax.legend(cluster_name)
    plt.savefig('./2_5_survival_ratio.png')

def main():
    # load data
    filename='../data/counts.txt'
    counts,cols=load_data(filename)
    patients=pd.read_csv('../data/patients.csv',index_col=0)
    print("data shape : {}".format(patients.shape))
    print("example of data table:")
    print(patients.head())
    # 预测幸存者
    n_clusters=3
    counts_log=np.log(counts+1)
    counts_var=most_variable_rows(counts_log,n=1500)
    _,yc=bicluster(counts_var,linkage_method='ward',distance_metric='euclidean')
    threshold_distance=(yc[-n_clusters,2]+yc[-n_clusters+1,2])/2
    clusters=hc.fcluster(yc,threshold_distance,'distance')
    plot_cluster_survival_curves(clusters,cols,patients)

if __name__ == "__main__":
    main()