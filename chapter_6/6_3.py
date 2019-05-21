'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-19 14:37:50
@LastEditors: shifaqiang
@LastEditTime: 2019-05-19 20:57:15
@Software: Visual Studio Code
@Description: 
'''

import numpy as np
# packages for Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

def plot_connectome(x_coords,
                    y_coords,
                    conn_matrix,
                    *,
                    filename,
                    labels=(),
                    types=None,
                    type_names=('', ),
                    xlabel='',
                    ylabel=''):
    if types is None:
        types = np.zeros(x_coords.shape, dtype=int)
    n_types = len(np.unique(types))
    colors = plt.rcParams['axes.prop_cycle'][:n_types].by_key()['color']
    cmap = ListedColormap(colors)

    _, ax = plt.subplots()
    for neuron_type in range(n_types):
        plotting = (types == neuron_type)
        pts = ax.scatter(x_coords[plotting],
                         y_coords[plotting],
                         c=cmap(neuron_type),
                         s=4,
                         zorder=1)
        pts.set_label(type_names[neuron_type])
    # 添加文本标签
    for x, y, label in zip(x_coords, y_coords, labels):
        ax.text(x,
                y,
                '  ' + label,
                verticalalignment='center',
                size=3,
                zorder=2)
    # 绘制边
    pre, post = np.nonzero(conn_matrix)
    links = np.array([[x_coords[pre], x_coords[post]],
                      [y_coords[pre], y_coords[post]]]).T
    ax.add_collection(
        LineCollection(links, color='lightgray', lw=0.3, alpha=0.5, zorder=0))
    ax.legend(scatterpoints=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(filename)


def neuron_data():
    '''
    @Description: 拉普拉斯矩阵在大脑数据中的应用
    '''
    Chem = np.load('../data/chem-network.npy')
    Gap = np.load('../data/gap-network.npy')
    neuron_ids = np.load('../data/neurons.npy')
    neuron_types = np.load('../data/neuron-types.npy')
    A = Chem + Gap
    C = (A + A.T) / 2  # 获得链接矩阵C以去除方向性
    degrees = np.sum(C, axis=0)
    D = np.diag(degrees)
    L = D - C
    b = np.sum(C * np.sign(A - A.T), axis=1)
    z = np.linalg.pinv(L) @ b
    # 获取度标准化的拉普拉斯矩阵@
    Dinv2 = np.diag(1 / np.sqrt(degrees))
    Q = Dinv2 @ L @ Dinv2
    val, vec = np.linalg.eig(Q)
    smallest_first = np.argsort(val)
    val = val[smallest_first]
    vec = vec[:, smallest_first]
    x = Dinv2 @ vec[:, 1]
    vc2_index = np.argwhere(neuron_ids == 'VC02')
    if x[vc2_index] < 0:
        x = -x
    # 绘制神经元图片
    plot_connectome(
        x,
        z,
        C,
        filename='./6_3_neuron_links.png',
        labels=neuron_ids,
        types=neuron_types,
        type_names=['Sensor neurons', 'Inter neurons', 'Motor neurons'],
        xlabel='Affinity eigenvector 1',
        ylabel='Processing depth')
    # 绘制近邻视图
    y=Dinv2@vec[:,2]
    asjl_index=np.argwhere(neuron_ids=="ASJL")
    if y[asjl_index]<0:
        y=-y
    plot_connectome(
        x,
        y,
        C,
        filename='./6_3_neuron_adjaency.png',
        labels=neuron_ids,
        types=neuron_types,
        type_names=['Sensor neurons', 'Inter neurons', 'Motor neurons'],
        xlabel='Affinity eigenvector 1',
        ylabel='Affinity eigenvector 2')


if __name__ == "__main__":
    neuron_data()