'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-16 11:15:31
@LastEditors: shifaqiang
@LastEditTime: 2019-05-18 21:07:28
@Software: Visual Studio Code
@Description: computation of confusion matrix
'''

import numpy as np
from scipy import sparse


def confusion_matrix(pred, gt):
    '''需要遍历4遍gt'''
    cont = np.zeros((2, 2))
    for i in [0, 1]:
        for j in [0, 1]:
            cont[i, j] = np.sum((pred == i) & (gt == j))
    return cont


def confusion_matrix_one_pass(pred, gt):
    '''
    @description: 一次遍历计算confusion matrix
        更加Python化的方法，符合 the Zen of Python
    '''
    cont = np.zeros((2, 2))
    for i, j in zip(pred, gt):
        cont[i, j] += 1
    return cont


def confusion_matrix_one_pass_2(pred, gt):
    '''
    @description: 一次遍历计算confusion matrix
        更加C++风格的写法，有利用使用Cython，Numbad等工具编译后提高运行速度
    '''
    cont = np.zeros((2, 2))
    count = len(pred)
    for k in range(count):
        i = pred[k]
        j = gt[k]
        cont[i, j] += 1
    return cont


def confusion_matrix_with_sparse_coo(pred, gt):
    cont = sparse.coo_matrix((np.ones_like(pred), (pred, gt)))
    return cont


def confusion_matrix_with_sparse_coo_and_reduce_memory(pred, gt):
    # 使用numpy广播机制建立虚拟数组压缩内存，np.broadcast_to(1,pred.size)
    cont = sparse.coo_matrix((np.broadcast_to(1, pred.size), (pred, gt)))
    return cont


def confusion():
    pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 1])
    gt = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    confusion_matrix_1 = confusion_matrix(pred, gt)
    print('四次遍历算法:\n{}'.format(confusion_matrix_1))
    confusion_matrix_2 = confusion_matrix_one_pass(pred, gt)
    print('一次遍历算法:\n{}'.format(confusion_matrix_2))
    confusion_matrix_2_1 = confusion_matrix_one_pass_2(pred, gt)
    print('一次遍历算法2:\n{}'.format(confusion_matrix_2_1))
    confusion_matrix_sparse_coo = confusion_matrix_with_sparse_coo(pred, gt)
    print('稀疏矩阵实现:\n{}'.format(confusion_matrix_sparse_coo.toarray()))
    confusion_matrix_sparse_coo_2 = confusion_matrix_with_sparse_coo_and_reduce_memory(
        pred, gt)
    print('稀疏矩阵并压缩内存实现:\n{}'.format(confusion_matrix_sparse_coo_2.toarray()))


def sparse_coo():
    s = np.array([[4, 0, 3], [0, 32, 0]])
    data = np.array([4, 3, 32])
    row = np.array([0, 0, 1])
    col = np.array([0, 2, 1])
    s_coo = sparse.coo_matrix((data, (row, col)))
    print('Original matrix:\n{}'.format(s))
    print('Sparse_coo matrix:\n{}'.format(s_coo.toarray()))
    print('.A attribute of sparse_coo matrix:\n{}'.format(s_coo.A))

    # practice
    data_2 = np.array([6, 1, 2, 4, 5, 1, 9, 6, 7])
    row_2 = np.array([0, 1, 1, 1, 1, 2, 3, 4, 4])
    col_2 = np.array([2, 0, 1, 3, 4, 1, 0, 3, 4])
    practice_coo = sparse.coo_matrix((data_2, (row_2, col_2)))
    print('practice:\n{}'.format(practice_coo.toarray()))


def sparse_csr():
    data = np.array([6, 1, 2, 4, 5, 1, 9, 6, 7])
    row = np.array([0, 1, 1, 1, 1, 2, 3, 4, 4])
    col = np.array([2, 0, 1, 3, 4, 1, 0, 3, 4])
    indptr = np.array([0, 1, 5, 6, 7, 9])

    coo = sparse.coo_matrix((data, (row, col)))
    csr = sparse.csr_matrix((data, col, indptr))
    print('The CSR to Numpy array:\n{}'.format(csr.toarray()))
    print('The CSR and COO arrays are equal : {}'.format(
        np.all(coo.toarray() == csr.toarray())))


def demo_accumulated_sum_of_sparse_coo():
    row = [0, 0, 2]
    col = [1, 1, 2]
    data = [5, 7, 1]
    s = sparse.coo_matrix((data, (row, col)))
    print('demo of accumulated sum of sparse coo:')
    print(s.toarray())


def seg_of_image():
    seg = np.array([[1, 1, 1], [1, 2, 2], [3, 3, 3]], dtype=int)
    gt = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=int)
    # flat()返回深拷贝， ravel()返回浅拷贝
    print("flat of seg:{}".format(seg.ravel()))
    print("flat of gt:{}".format(gt.ravel()))
    cont = sparse.coo_matrix(
        (np.broadcast_to(1, seg.size), (seg.ravel(), gt.ravel())))
    print('sparse matrix of cont:\n{}'.format(cont))
    print('cont:\n{}'.format(cont.toarray()))


def demo_of_entropy():
    prains = np.array([25, 27, 24, 18, 14, 11, 7, 8, 10, 15, 18, 23]) / 100
    pshine = 1 - prains
    p_rain_g_month = np.column_stack([prains, pshine])
    print("1-12月下雨与晴天的概率：\n{}".format(p_rain_g_month))
    print('table total"{}'.format(np.sum(p_rain_g_month)))
    # p_rain_month 联合概率
    # p_rain_g_month 边缘概率
    # p_month_g_rain 边缘概率
    # p_rain_g_month 条件概率
    # p_month_g_rain 条件概率
    p_rain_month = p_rain_g_month / np.sum(p_rain_g_month)
    p_rain = np.sum(p_rain_month, axis=0)
    p_month_g_rain = p_rain_month / p_rain
    p_month = np.sum(p_rain_month, axis=1)
    print("P_rain:{}".format(p_rain))
    print("P_month:{}".format(p_month))
    Hmr = np.sum(p_rain * p_month_g_rain * np.log2(1 / p_month_g_rain))
    Hm = np.sum(p_month * np.log2(1 / p_month))
    print('H(M|R):{}'.format(Hmr))
    print('H(M):{}'.format(Hm))


def demo_of_log_zero():
    print('The log of 0 is:{}'.format(np.log2(0)))  # -np.inf
    print('0 times the log of 0 is:{}'.format(0 * np.log2(0)))  # np.nan


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


def demo_of_entropy():
    arr = np.array([0.25, 0.25, 0, 0.25, 0.25])
    mat = sparse.coo_matrix([[0.125, 0.125, 0.125, 0],
                             [0.125, 0.125, 0, 0.125]])
    print("Demo of entropy:")
    print("\tOriginal probability (np.ndarray):{}".format(arr))
    print("\tThe entropy of the probability (np.ndarray):{}".format(
        entropy(arr)))
    print("\tOriginal probability (np.sparse_matrix):\n\t\t{}".format(mat.A))
    print(
        "\tThe entropy of the probability (np.sparse_matrix):\n\t\t{}".format(
            entropy(mat).A))


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


def demo_of_VI():
    S = np.array([[0, 1], [2, 3]], dtype=int)
    T = np.array([[0, 1], [0, 1]], dtype=int)
    cont = sparse.coo_matrix(
        (np.broadcast_to(1, S.size), (S.ravel(), T.ravel()))).toarray()
    print("S:\n{}".format(S))
    print("T:\n{}".format(T))
    print("confusion matrix (count):\n{}".format(cont))
    cont = cont / np.sum(cont)
    print("confusion matrix (probability):\n{}".format(cont))
    p_S = np.sum(cont, axis=1)
    p_T = np.sum(cont, axis=0)
    print("p_S:{}".format(p_S))
    print("p_T:{}".format(p_T))
    H_ST = np.sum(np.sum(entropy(cont / p_T), axis=0) * p_T)
    H_TS = np.sum(np.sum(entropy(cont / p_S[:, np.newaxis]), axis=1) * p_S)
    print('VI(variation of information) = H_ST({}) + H_TS({}) = {}'.format(
        H_ST, H_TS, H_ST + H_TS))
    print('VI(from sparse matrix) = {}'.format(variation_of_information(S, T)))


if __name__ == "__main__":
    confusion()
    sparse_coo()
    sparse_csr()
    demo_accumulated_sum_of_sparse_coo()
    seg_of_image()
    demo_of_entropy()
    demo_of_log_zero()
    demo_of_entropy()
    demo_of_VI()