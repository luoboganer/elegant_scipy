'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-20 12:19:43
@LastEditors: shifaqiang
@LastEditTime: 2019-05-20 20:20:20
@Software: Visual Studio Code
@Description: 
'''

# packages for Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')

import numpy as np
from skimage import io, data, color, transform, filters, util
from scipy import ndimage as ndi, optimize
from scipy.stats import entropy


def mse(arr1, arr2):
    return np.mean((arr1 - arr2)**2)


def normalized_mutual_information(A, B):
    '''
    @Description: compute normalized mutual information between A and B
        definition: NMI(A,B)=(H(A)+H(B))/(H(A,B)
        where, H(X)=entropy(X)=-(Xlog(X))
    '''
    hist, bin_edges = np.histogramdd([np.ravel(A), np.ravel(B)], bins=100)
    hist /= np.sum(hist)
    H_A = entropy(np.sum(hist, axis=0))
    H_B = entropy(np.sum(hist, axis=1))
    H_AB = entropy(np.ravel(hist))
    return (H_A + H_B) / H_AB


def downsample2x(image):
    offsets = [((s + 1) % 2) / 2 for s in image.shape]
    slices = [
        slice(offset, end, 2) for offset, end in zip(offsets, image.shape)
    ]
    coords = np.mgrid[slices]
    return ndi.map_coordinates(image, coords, order=1)


def gaussian_pyramid(image, levels=6):
    pyramid = [image]
    for level in range(levels - 1):
        blurred = ndi.gaussian_filter(image, sigma=2 / 3)
        image = downsample2x(image)
        pyramid.append(image)
    return list(reversed(pyramid))


def make_rigid_transform(param):
    r, tc, tr = param
    return transform.SimilarityTransform(rotation=r, translation=(tc, tr))


def cost_mse(param, reference_image, target_image):
    transformation = make_rigid_transform(param)
    transformed = transform.warp(target_image, transformation, order=3)
    return mse(reference_image, transformed)


def align(reference, target, cost=cost_mse, BasinHopping=False):
    # 高斯金字塔优化的使用前一级的重点作为下一级的起点
    nlevel = 7
    pyramid_ref = gaussian_pyramid(reference, levels=nlevel)
    pyramid_tgt = gaussian_pyramid(target, levels=nlevel)
    levels = range(nlevel, 0, -1)
    image_pairs = zip(pyramid_ref, pyramid_tgt)
    p = np.zeros(3)
    for n, (ref, tgt) in zip(levels, image_pairs):
        p[1:] *= 2
        if n > 4 and BasinHopping:
            res = optimize.basinhopping(cost,
                                        p,
                                        minimizer_kwargs={'args': (ref, tgt)})
        else:
            res = optimize.minimize(cost, p, args=(ref, tgt), method='Powell')
        p = res.x
        # 输出当前前级别的优化结果
        print(f'Level:{n}, Angle:{np.rad2deg(res.x[0]):.3}, ',
              f'Offset ({res.x[1]*2**n:.3}, {res.x[2]*2**n:.3}), ',
              f'Cost: {res.fun:.3}',
              end='\r')
    print('')
    return make_rigid_transform(p)


def plot_shift():
    astronaut = color.rgb2gray(data.astronaut())
    shifted = ndi.shift(astronaut, (0, 50))
    _, axes = plt.subplots(3, 2, figsize=(4.8 * 2, 3.2 * 3))
    axes[0][0].set_title('Original')
    axes[0][0].imshow(astronaut)
    axes[0][1].set_title('Shifted')
    axes[0][1].imshow(shifted)
    # 通均方误差(MSE)相似度差异进行平移优化
    ncol = astronaut.shape[1]
    shifts = np.linspace(-0.9 * ncol, 0.9 * ncol, 181)
    mse_cost = []
    for shift in shifts:
        shifted_back = ndi.shift(shifted, (0, shift))
        mse_cost.append(mse(astronaut, shifted_back))
    axes[1][0].set_xlabel('Shift')
    axes[1][0].set_ylabel('MSE')
    axes[1][0].set_title('Correct example of MSE')
    axes[1][0].plot(shifts, mse_cost)

    # 定义了损失函数之后使用scipy.optimize.minimize()函数来搜索最小误差
    def astronaut_shift_error(shift, image):
        corrected = ndi.shift(image, (0, shift))
        return mse(astronaut, corrected)

    res = optimize.minimize(astronaut_shift_error,
                            0,
                            args=(shifted, ),
                            method='Powell')
    print(f'The optimal shift for correction is: {res.x}')
    # 根据MSE优化失败的例子
    ncol_2 = astronaut.shape[1]
    shifts_2 = np.linspace(-0.9 * ncol_2, 0.9 * ncol_2, 181)
    mse_cost_2 = []
    for shift in shifts_2:
        shifted_back_2 = ndi.shift(astronaut, (0, shift))
        mse_cost_2.append(mse(astronaut, shifted_back_2))
    axes[1][1].set_xlabel('Shift')
    axes[1][1].set_ylabel('MSE')
    axes[1][1].set_title('Failure example of MSE')
    axes[1][1].plot(shifts_2, mse_cost_2)
    axes[1][1].annotate("Local minimum", (-380, 0.24),
                        xytext=(-380, 0.18),
                        color='red',
                        size='small',
                        arrowprops=dict(width=0.5,
                                        headwidth=3,
                                        headlength=4,
                                        fc='k',
                                        shrink=0.1))
    shifted_2 = ndi.shift(astronaut, (0, -340))
    # 这时优化函数可能找到-380左右的局部极小值，而无法找到真正的最小值(全局极小)0
    res = optimize.minimize(astronaut_shift_error,
                            0,
                            args=(shifted_2, ),
                            method='Powell')
    # 理论上这里得到的结果应该是340，而不是-38
    print(f'the optimal shift for correction is {res.x}')
    # 解决这种问题的一般方法是对图像进行平滑和缩放
    astronaut_smooth = filters.gaussian(astronaut, sigma=20)
    mse_cost_smooth = []
    shifts_smooth = np.linspace(-0.9 * ncol_2, 0.9 * ncol_2, 181)
    for shift in shifts_smooth:
        shifted_back_smooth = ndi.shift(astronaut_smooth, (0, shift))
        mse_cost_smooth.append(mse(astronaut_smooth, shifted_back_smooth))
    axes[2][0].set_xlabel('Shift')
    axes[2][0].set_ylabel('MSE')
    axes[2][0].set_title('MSE with Gaussian smooth')
    axes[2][0].plot(shifts_smooth, mse_cost_2, label='Original')
    axes[2][0].plot(shifts_smooth, mse_cost_smooth, label='Smooth')
    axes[2][0].legend()

    # 高斯金字塔对齐
    shifts_pyramid = np.linspace(-0.9 * ncol, 0.9 * ncol, 181)
    n_level = 8
    costs = np.zeros(shape=(n_level, len(shifts_pyramid)), dtype=float)
    astronaut_pyramid = gaussian_pyramid(astronaut, levels=n_level)
    for col, shift in enumerate(shifts_pyramid):
        shifted_back_pyramid = ndi.shift(astronaut, (0, shift))
        shifted_pyramid = gaussian_pyramid(shifted_back_pyramid,
                                           levels=n_level)
        for row, image in enumerate(shifted_pyramid):
            costs[row, col] = mse(astronaut_pyramid[row], image)
    for level, cost in enumerate(costs):
        axes[2][1].plot(shifts_pyramid, cost, label=f'Level={n_level-level}')
    axes[2][1].legend(loc='lower right', frameon=True, framealpha=0.9)
    axes[2][1].set_xlabel('Shift')
    axes[2][1].set_ylabel('MSE')
    # 保存绘图结果
    plt.tight_layout()
    plt.savefig('./7_1_shift.png')


def plot_rotation():
    # 原图与旋转效果
    theta = 45
    astronaut = color.rgb2gray(data.astronaut())
    rotated = transform.rotate(astronaut, theta)
    _, axes = plt.subplots(4, 2, figsize=(2.4 * 2, 2.4 * 4))
    axes[0][0].imshow(astronaut)
    axes[0][0].set_title('Original')
    axes[0][1].imshow(rotated)
    axes[0][1].set_title(f'Rotated(theta={theta})')
    # 配准成功的案例
    theta = 60
    rotated = transform.rotate(astronaut, theta)
    rotated = util.random_noise(rotated,
                                mode='gaussian',
                                seed=0,
                                mean=0,
                                var=1e-3)
    tf = align(astronaut, rotated)
    corrected = transform.warp(rotated, tf, order=3)
    axes[1][0].imshow(rotated)
    axes[1][0].set_title(f'Rotated(theta={theta})')
    axes[1][1].imshow(corrected)
    axes[1][1].set_title('Registered')
    # 配准失败的案例，陷入局部极小值
    theta = 50
    rotated = transform.rotate(astronaut, theta)
    rotated = util.random_noise(rotated,
                                mode='gaussian',
                                seed=0,
                                mean=0,
                                var=1e-3)
    tf = align(astronaut, rotated)
    corrected = transform.warp(rotated, tf, order=3)
    axes[2][0].imshow(rotated)
    axes[2][0].set_title(f'Rotated(theta={theta})')
    axes[2][1].imshow(corrected)
    axes[2][1].set_title('Registered (Failure)')
    # 使用basinhopping算法再次配准Powell失败的例子
    tf = align(astronaut, rotated, cost=cost_mse, BasinHopping=True)
    corrected = transform.warp(rotated, tf, order=3)
    axes[3][0].imshow(rotated)
    axes[3][0].set_title(f'Rotated(theta={theta})')
    axes[3][1].imshow(corrected)
    axes[3][1].set_title('Registered (BH)')
    # 保存绘图
    plt.tight_layout()
    plt.savefig('./7_1_rotation.png')


def cost_nmi(param, reference_image, target_image):
    transformation = make_rigid_transform(param)
    transformed = transform.warp(target_image, transformation, order=3)
    return -normalized_mutual_information(reference_image, transformed)


def plot_modal():
    stained_glass = io.imread('../data/00998v.jpg') / 255.0
    print(f'shape of original image: {stained_glass.shape}')
    step = stained_glass.shape[0] // 3
    channels = (stained_glass[:step, :], stained_glass[step:step * 2, :],
                stained_glass[step * 2:step * 3, :])
    channels_name = ['blue', 'green', 'red']
    _, axes = plt.subplots(2, 3, figsize=(4.8 * 3, 4.8 * 2))
    for i in range(3):
        axes[0][i].imshow(channels[i])
        axes[0][i].set_title(channels_name[i])
        axes[0][i].axis('off')
    # 简单的RGB叠加
    b, g, r = channels
    original = np.dstack((r, g, b))
    axes[1][0].imshow(original)
    axes[1][0].set_title('RGB(Simple superposition)')
    axes[1][0].axis('off')
    # 基于MSE的对齐，以green为基准，将red,blue向blue对齐，对齐效果不明显，没有完全消除光晕
    print('*** Aligning blue to green based-MSE ***')
    tf = align(g, b)
    cblue = transform.warp(b, tf, order=3)
    print('*** Aligning red to green based-MSE ***')
    tf = align(g, r)
    cred = transform.warp(r, tf, order=3)
    corrected = np.dstack((cred, g, cblue))
    axes[1][1].imshow(corrected)
    axes[1][1].set_title('RGB(alignment based-MSE)')
    axes[1][1].axis('off')
    # 基于NMI(Normalized Mutual Information)的对齐，以green为基准，将red,blue向blue对齐，对齐效果不明显，没有完全消除光晕
    print('*** Aligning blue to green based-NMI ***')
    tf = align(g, b,cost=cost_nmi)
    cblue = transform.warp(b, tf, order=3)
    print('*** Aligning red to green based-NMI ***')
    tf = align(g, r,cost=cost_nmi)
    cred = transform.warp(r, tf, order=3)
    corrected = np.dstack((cred, g, cblue))
    axes[1][2].imshow(corrected)
    axes[1][2].set_title('RGB(alignment based-NMI)')
    axes[1][2].axis('off')
    # 保存绘图
    plt.tight_layout()
    plt.savefig('./7_1_modals.png')


if __name__ == "__main__":
    plot_shift()
    plot_rotation()
    plot_modal()