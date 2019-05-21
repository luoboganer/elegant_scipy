'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-19 14:37:50
@LastEditors: shifaqiang
@LastEditTime: 2019-05-19 20:48:59
@Software: Visual Studio Code
@Description: 
'''

import numpy as np
import networkx as nx
# packages for Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap


def demo_of_variable_name():
    m, n = (5, 6)
    M = np.ones((5, 6))
    v = np.random.random((n, ))
    w = M @ v
    print(f'scale m={m} and n={n}')
    print(f'matrix:\n{M}')
    print(f'vector:\nv={v}\nw={w}')


def demo_of_vector_rotation():
    def R_generation(angle):
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        return R

    # 将一个三维向量旋转45 degree
    R = R_generation(45)
    print(f'R = {R}')
    print(f'R times the x-axis [1,0,0]:{R@[1,0,0]}')
    print(f'R times the y-axis [0,1,0]:{R@[0,1,0]}')
    print(f'R times a 45 degree vector [1,1,0]:{R@[1,1,0]}')
    # 将一个三维向量旋转45 degree两次，等于旋转90度
    print(f'R times a 45 degree vector [1,1,0] (two pass):{R@R@[1,1,0]}')
    print(f'R times a 90 degree vector [1,1,0]:{R_generation(90)@[1,1,0]}')
    # 验证该R的结构不改变z值
    print(f'R times the z-axis [0,0,1]:{R@[0,0,1]}')
    # 求R的特征值
    lamda, V = np.linalg.eig(R)
    print(f'The eigenvalue of R {lamda}')
    print(f'The eigenvector of R {V}')


def demo_of_Fiedler_vector():
    '''
    @Description: Fiedler_vector是拉普拉斯矩阵的次小特征值对应的特征向量
    '''
    A = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]],
                 dtype=float)
    print(f'adjacency matrix(A):\n{A}')
    g = nx.from_numpy_matrix(A)
    _, axes = plt.subplots(2, 2, figsize=(4.8 * 2, 3.2 * 2))
    # 原始的图
    layout = nx.spring_layout(g, pos=nx.circular_layout(g))
    nx.draw(g, pos=layout, with_labels=True, node_color='white', ax=axes[0][0])
    axes[0][0].set_title('base Graph Visualization')
    # 计算度矩阵和拉普拉斯矩阵
    d = np.sum(A, axis=0)
    D = np.diag(d)
    print(f'degree matrix(D):\n{D}')
    L = D - A
    print(f'lapalace matrix(L=D-A)\n{L}')
    # 计算特征值和特征向量
    val, vec = np.linalg.eig(L)
    print(f'val={val}')
    print(f'vec=\n{vec}')
    axes[1][0].plot(np.sort(val), linestyle='-', marker='o')
    axes[1][0].set_title('sorted eigen value')
    Fiedler_vector = vec[:, np.argsort(val)[1]]
    axes[1][1].plot(Fiedler_vector, linestyle='-', marker='o')
    axes[1][1].set_title('Fiedler vector of L')
    colors = ['orange' if eigv > 0 else 'gray' for eigv in Fiedler_vector]
    nx.draw(g, pos=layout, with_labels=True, node_color=colors, ax=axes[0][1])
    axes[0][1].set_title('Graph Visualization with Fiedler vector')
    plt.tight_layout()
    # 保存绘图结果
    plt.savefig("./6_1_graph.png")


if __name__ == "__main__":
    demo_of_variable_name()
    demo_of_vector_rotation()
    demo_of_Fiedler_vector()