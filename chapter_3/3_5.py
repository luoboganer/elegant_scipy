'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-13 19:17:13
@LastEditors: shifaqiang
@LastEditTime: 2019-05-15 10:20:07
@Software: Visual Studio Code
@Description: 
'''

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')
from scipy.optimize import curve_fit

def construct_graph_from_csv(filename):
    conn=pd.read_excel(filename)
    print('data shape:{}'.format(conn.shape))
    print('data examples:')
    print(conn.head())
    # preprocessing
    conn_edges=[(n1,n2,{'weight':s}) for n1,n2,_,s in conn.itertuples(index=False,name=None)]
    wormbrain=nx.DiGraph()
    wormbrain.add_edges_from(conn_edges)
    return wormbrain

def first_5_centrality(wormbrain):
    assert type(wormbrain)==nx.DiGraph
    centrality=nx.betweenness_centrality(wormbrain)
    central=sorted(centrality,key=centrality.get,reverse=True)
    print('Top-5 neuron : {}'.format(central[:5]))

def find_strongly_connected_component(wormbrain):
    assert type(wormbrain)==nx.DiGraph
    sccs=nx.strongly_connected_component_subgraphs(wormbrain)
    giantscc=max(sccs,key=len)
    print('The largest strongly connected component has {} nodes, out of {} nodes'.format(giantscc.number_of_nodes(),wormbrain.number_of_nodes()))


def plot_fraction(wormbrain):
    in_degrees=[in_degree for _,in_degree in wormbrain.in_degree()]
    in_degrees_distribution=np.bincount(in_degrees)
    avg_in_degrees=np.mean(in_degrees)
    cum_frq=np.cumsum(in_degrees_distribution)/np.sum(in_degrees_distribution)
    survival=1-cum_frq
    _=plt.figure()
    plt.loglog(np.arange(1,len(survival)+1),survival)
    plt.xlabel('in-degrees distribution')
    plt.ylabel('fraction of neurons with higher in-degree distribution')
    plt.scatter(avg_in_degrees,0.0022,marker='v')
    plt.text(avg_in_degrees-0.5,0.003,'mean={:.2f}'.format(avg_in_degrees))

    def fraction_higher(degree,alpha,gamma):
        return alpha*degree**(-gamma)

    x=1+np.arange(len(survival))
    valid=x>10
    x,y=x[valid],survival[valid]
    alpha_fit,gamma_fit=curve_fit(fraction_higher,x,y)[0]
    y_fit=fraction_higher(x,alpha_fit,gamma_fit)
    plt.loglog(x,y_fit,c='red')

    plt.savefig('./3_5_fraction.png')

if __name__ == "__main__":
    # 'http://www.wormatlas.org/images/NeuronConnect.xls'
    wormbrain=construct_graph_from_csv('../data/NeuronConnect.xls')
    first_5_centrality(wormbrain)
    find_strongly_connected_component(wormbrain)
    plot_fraction(wormbrain)