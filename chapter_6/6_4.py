'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-19 21:02:44
@LastEditors: shifaqiang
@LastEditTime: 2019-05-20 12:17:57
@Software: Visual Studio Code
@Description: 
'''
# packages for Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def pagerank_plot(pagerank, in_degrees, names, *, annotations=[], ax, c,
                  label):
    '''
    @Description: 
        plot node pagerank against in-degree, with hand-picked node names 
    '''
    ax.scatter(in_degrees, pagerank, c=c, label=label)
    for name, indeg, pr in zip(names, in_degrees, pagerank):
        if name in annotations:
            text = ax.text(indeg + 0.1, pr, name)
    ax.set_ylim(0, np.max(pagerank) * 1.1)
    ax.set_xlim(-1, np.max(in_degrees) * 1.1)
    ax.set_ylabel('PageRank')
    ax.set_xlabel('In-Degree')
    ax.legend()


def power(Trans: np.array, damping=0.85, max_iter=10**5):
    n = Trans.shape[0]
    r0 = np.full(n, 1 / n)
    r = r0
    for _iter_num in range(max_iter):
        r_next = damping * Trans @ r + (1 - damping) / n
        if np.allclose(r_next, r):
            break
        r = r_next
    return r


def power2(Trans: np.array, damping=0.85, max_iter=10**5):
    n = Trans.shape[0]
    dangling = (1 / n) * np.ravel(Trans.sum(axis=0) == 0)
    r0 = np.full(n, 1 / n)
    r = r0
    for _iter_num in range(max_iter):
        r_next = damping * (Trans @ r + dangling @ r) + (1 - damping) / n
        if np.allclose(r_next, r):
            break
        r = r_next
    return r


if __name__ == "__main__":
    # load data
    stmarks = nx.read_gml('../data/stmarks.gml')
    species = np.array(stmarks.node())
    Adj = nx.to_scipy_sparse_matrix(stmarks, dtype=np.float)
    n_total_species = len(species)
    np.seterr(divide='ignore')  # 忽略除0错误
    # 计算出度矩阵(这是一个对角矩阵，每个对角元素表示该节点的出度的倒数)
    degrees = np.ravel(Adj.sum(axis=1))
    Deginv = sparse.diags(1 / degrees).tocsr()
    Trans = (Deginv @ Adj).T
    damping = 0.85  # 阻尼系数
    beta = 1 - damping
    I = sparse.eye(n_total_species, format='csc')  # 与Trans相同的稀疏格式
    PageRank = spsolve(I - damping * Trans,
                       np.full(n_total_species, beta / n_total_species))
    PageRank_power = power(Trans)
    PageRank_power_2 = power2(Trans)
    interesting = [
        'detritus', 'phytoplankton', 'benthic algae', 'micro-epiphytes',
        'microfauna', 'zooplankton', 'predatory shrimps', 'meiofauna', 'gulls'
    ]
    in_degrees = np.ravel(Adj.sum(axis=0))
    _, axes = plt.subplots(3, 1, figsize=(4.8, 3.2 * 3))
    pagerank_plot(PageRank,
                  in_degrees,
                  species,
                  annotations=interesting,
                  ax=axes[0],
                  c='red',
                  label='Spsolove')
    pagerank_plot(PageRank_power,
                  in_degrees,
                  species,
                  annotations=interesting,
                  ax=axes[1],
                  c='green',
                  label='Power')
    pagerank_plot(PageRank_power_2,
                  in_degrees,
                  species,
                  annotations=interesting,
                  ax=axes[2],
                  c='blue',
                  label='Power_2')
    plt.tight_layout()
    plt.savefig('./6_4_PI.png')
    identity = np.corrcoef([PageRank, PageRank_power, PageRank_power_2])
    print("the results of three different methods are identity?")
    print(identity)