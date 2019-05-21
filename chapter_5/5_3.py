'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-16 16:11:15
@LastEditors: shifaqiang
@LastEditTime: 2019-05-17 12:53:58
@Software: Visual Studio Code
@Description: 
'''

# packages for Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')

from skimage import data
from scipy import sparse, ndimage as ndi
import numpy as np
from itertools import product
import timeit


def homography(tf, image_shape):
    '''
    @Description:
        represent homographic transformation and interpolation as linear operator 
    @Parameters: 
        tf: (3,3) ndarray, transformation matrix
        image_shape: (M,N), shape of the input gray image
    @Return: 
        A: (M*N,M*N) sparse matrix
            linear-operator representing transformation + bilinear interpolation
    '''
    H = np.linalg.inv(tf)
    m, n = image_shape
    row, col, value = [], [], []
    for sparse_op_row, (out_row,
                        out_col) in enumerate(product(range(m), range(n))):
        # 计算输出矩阵中的每个位置(i,j)在输入矩阵中的位置
        in_row, in_col, in_abs = H @ [out_row, out_col, 1]
        in_row /= in_abs
        in_col /= in_abs
        # 如果坐标点跑出了图像外，在忽略这个点
        if not (0 <= in_row < m - 1 and 0 <= in_col < n - 1):
            continue
        else:
            # 双线性插值
            top = int(np.float(in_row))
            left = int(np.float(in_col))
            t = in_row - top
            u = in_col - left
            row.extend([sparse_op_row] * 4)
            sparse_op_col = np.ravel_multi_index(
                ([top, top, top + 1, top + 1
                  ], [left, left + 1, left, left + 1]),
                dims=(m, n))
            col.extend(sparse_op_col)
            value.extend([(1 - t) * (1 - u), (1 - t) * u, t * (1 - u), t * u])
    operator = sparse.coo_matrix((value, (row, col)),
                                 shape=(m * n, m * n)).tocsr()
    return operator


def rotate(angle):
    c = np.cos(np.deg2rad(angle))
    s = np.sin(np.deg2rad(angle))
    H = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    point = np.array([1, 0, 1])
    print("Demo of point {}".format(point))
    print('after one rotate by {} degree, {}'.format(angle, H @ point))
    print('after three rotate by {} degree, {}'.format(angle,
                                                       H @ H @ H @ point))


def apply_transfrom(image, tf):
    return (tf @ image.flat).reshape(image.shape)


def convert_image(angle):
    c = np.cos(np.deg2rad(angle))
    s = np.sin(np.deg2rad(angle))
    image = data.camera()
    H = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def transform_rotate_about_center(shape, degree):
        center = np.array(shape) / 2
        # 图像从中心移动到原点
        H_tr0 = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
        # 图像从原点移动回中心
        H_tr1 = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
        H_rot_tr0_tr1 = H_tr1 @ H @ H_tr0
        sparse_op = homography(H_rot_tr0_tr1, image.shape)
        return sparse_op

    tf = homography(H, image.shape)
    tf_center = transform_rotate_about_center(image.shape, 30)
    out = apply_transfrom(image, tf)
    out_center = apply_transfrom(image, tf_center)
    _, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    axes[0].imshow(image)
    axes[1].imshow(out)
    axes[2].imshow(out_center)
    plt.suptitle('result of image rotation', size='x-large')
    plt.tight_layout()
    plt.savefig('./5_3_camera_rotate.png')


if __name__ == "__main__":
    rotate(angle=30)
    convert_image(angle=30)

# #%%
# c = np.cos(np.deg2rad(30))
# s = np.sin(np.deg2rad(30))
# image = data.camera()
# H = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
# tf = homography(H, image.shape)

# %timeit apply_transfrom(image,tf)
# %timeit ndi.rotate(image,30,reshape=False,order=1)

# 1.35 ms ± 5.52 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 13.7 ms ± 35.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)