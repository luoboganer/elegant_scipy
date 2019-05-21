'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-18 14:00:10
@LastEditors: shifaqiang
@LastEditTime: 2019-05-18 21:41:48
@Software: Visual Studio Code
@Description: 以信息变异(variation of information)指标来估计图像分割的效果
'''
# packages for Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')
# other packages
import numpy as np
from skimage import io, color, future, segmentation
from scipy import ndimage as ndi, sparse
import networkx as nx


def plot_tiger_and_label(tiger, label):
    tiger_and_label, num_classes = ndi.label(boundaries > 100)
    human_seg = color.label2rgb(tiger_and_label, tiger)
    print('classes = {}'.format(num_classes))
    _, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    axes[0][0].imshow(tiger)
    axes[0][0].set_title('Origianl image(tiger)')
    axes[1][0].imshow(label)
    axes[1][0].set_title('Label')
    axes[0][1].imshow(tiger_and_label)
    axes[0][1].set_title('Label2Class')
    axes[1][1].imshow(human_seg)
    axes[1][1].set_title('Class on Image')
    plt.tight_layout()
    plt.savefig('./5_9_tiger_and_boundaries.png')


def add_edge_filter(values, graph):
    current = values[0]
    neighbors = values[1:]
    for neighbor in neighbors:
        graph.add_edge(current, neighbor)
    return 0.0


def build_RAG(labels, image):
    '''
    @Description: build RAG (Region Adjacent Graph) for a image and its segmentation label
    @Parameters: 
    @Return: 
    '''
    g = nx.Graph()
    footprint = ndi.generate_binary_structure(labels.ndim, connectivity=1)
    for j in range(labels.ndim):
        fp = np.swapaxes(footprint, j, 0)
        fp[0, ...] = 0
    _ = ndi.generic_filter(labels,
                           add_edge_filter,
                           footprint=footprint,
                           mode='nearest',
                           extra_arguments=(g, ))
    for n in g:
        g.node[n]['total color'] = np.zeros(3, np.double)
        g.node[n]['pixel count'] = 0
    for index in np.ndindex(labels.shape):
        n = labels[index]
        g.node[n]['total color'] += image[index]
        g.node[n]['pixel count'] += 1
    return g


def threshold_graph(g, t):
    to_remove = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] > t]
    g.remove_edges_from(to_remove)


def rag_segmentation(base_seg, image, threshold=80):
    g = build_RAG(base_seg, image)
    for n in g:
        node = g.node[n]
        node['mean'] = node['total color'] / node['pixel count']
    for u, v in g.edges:
        d = g.node[u]['mean'] - g.node[v]['mean']
        g[u][v]['weight'] = np.linalg.norm(d)
    threshold_graph(g, threshold)
    map_array = np.zeros(np.max(base_seg) + 1, int)
    for i, segment in enumerate(nx.connected_components(g)):
        for initial in segment:
            map_array[int(initial)] = i
    segmented = map_array[base_seg]
    return segmented


def entropy(arr_or_mat):
    '''
    @Description: 
        out = [ -xlog2x for x in input_data ]
        compute the element-wise entropy function of an array or matrix
    @Parameters: 
        arr_or_mat: np.ndarray or np.sparse_matrix
    @Return: 
        out: return value must be same type with the input
    '''
    out = arr_or_mat.copy()
    if isinstance(out, sparse.spmatrix):
        arr = out.data
    else:
        arr = out
    nz = np.nonzero(arr)
    arr[nz] *= -np.log2(arr[nz])
    # arr=out.copy(), 这里是浅拷贝，应用传值，因此直接返回out即可
    return out


def invert_nonzero(arr):
    arr_inv = arr.copy()
    nz = np.nonzero(arr_inv)
    arr_inv[nz] = 1 / arr_inv[nz]
    return arr_inv


def variation_of_information(x, y):
    n = x.size
    # 联合概率，即列联矩阵
    Pxy = sparse.coo_matrix((np.full(n, 1 / n), (x.ravel(), y.ravel())),
                            dtype=float).tocsr()
    # 边缘概率
    Px = np.ravel(Pxy.sum(axis=1))
    Py = np.ravel(Pxy.sum(axis=0))
    Px_inv = sparse.diags(invert_nonzero(Px))
    Py_inv = sparse.diags(invert_nonzero(Py))
    hygx = Px @ entropy(Px_inv @ Pxy).sum(axis=1)
    hxgy = entropy(Pxy @ Py_inv).sum(axis=0) @ Py
    return float(hygx + hxgy)


def seg(tiger, human_seg):
    human_seg, num_classes = ndi.label(boundaries > 100)
    seg_slic = segmentation.slic(tiger,
                                 n_segments=30,
                                 compactness=40,
                                 enforce_connectivity=True,
                                 sigma=3)
    _, axes = plt.subplots(2, 3, figsize=(4.8 * 3, 3.6 * 2))
    axes[0][0].imshow(tiger)
    axes[0][0].set_title('Original Image(tiger)')
    axes[1][0].imshow(color.label2rgb(seg_slic, tiger))
    axes[1][0].set_title('Base slic segmentation')
    auto_seg_10 = rag_segmentation(seg_slic, tiger, threshold=10)
    axes[0][1].imshow(color.label2rgb(auto_seg_10, tiger))
    axes[0][1].set_title('Segmentation(threshold=10)')
    axes[0][1].set_xlabel('threshold={},VI={}'.format(
        10, variation_of_information(auto_seg_10, human_seg)))
    auto_seg_40 = rag_segmentation(seg_slic, tiger, threshold=40)
    axes[0][2].imshow(color.label2rgb(auto_seg_40, tiger))
    axes[0][2].set_title('Segmentation')
    axes[0][2].set_xlabel('threshold={},VI={}'.format(
        40, variation_of_information(auto_seg_40, human_seg)))
    auto_seg_80 = rag_segmentation(seg_slic, tiger, threshold=80)
    axes[1][1].imshow(color.label2rgb(auto_seg_80, tiger))
    axes[1][1].set_title('Segmentation')
    axes[1][1].set_xlabel('threshold={},VI={}'.format(
        80, variation_of_information(auto_seg_80, human_seg)))
    auto_seg_120 = rag_segmentation(seg_slic, tiger, threshold=120)
    axes[1][2].imshow(color.label2rgb(auto_seg_120, tiger))
    axes[1][2].set_title('Segmentation')
    axes[1][2].set_xlabel('threshold={},VI={}'.format(
        120, variation_of_information(auto_seg_120, human_seg)))
    # VI-threshold curve
    plt.tight_layout()
    plt.savefig('5_9_rag_seg.png')

    def vi_at_threshold(auto_seg, human_seg, threshold):
        auto_seg = rag_segmentation(seg_slic, tiger, threshold)
        return variation_of_information(auto_seg, human_seg)

    _ = plt.figure()
    plt.title('threshold-VI curve')
    plt.xlabel('threshold')
    plt.ylabel('VI(variation of information)')
    thresholds = np.arange(0, 110, 2)
    vi_per_threshold = [
        vi_at_threshold(seg, human_seg, threshold) for threshold in thresholds
    ]
    plt.plot(thresholds, vi_per_threshold)
    plt.savefig('./5_9_variation_of_information_curve.png')


if __name__ == "__main__":
    '''
    image_url= https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/108073.jpg
    human_seg_url= https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/human/normal/outline/color/1122/108073.jpg
    
    Result:
        (base) qianlicaody@qianlicaody:~/scipy/chapter_5$ time python 5_9.py 
            classes = 25

            real    1m24.174s
            user    1m28.244s
            sys     0m8.260s
    '''
    tiger = io.imread('../data/108073.jpg')
    boundaries = io.imread('../data/108073_seg.jpg')
    plot_tiger_and_label(tiger, boundaries)
    seg(tiger, boundaries)
