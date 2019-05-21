'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-12 19:57:38
@LastEditors: shifaqiang
@LastEditTime: 2019-05-15 10:20:02
@Software: Visual Studio Code
@Description: 
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')
from skimage import io


def random_image(filename, img_size=(500, 500)):
    img = np.random.rand(*img_size)
    plt.imsave(filename, img, format == 'png', cmap=plt.cm.gray_r)


def image_is_numpy_ndarray():
    '''
    gray image, https://raw.githubusercontent.com/scikit-image/scikit-image/v0.10.1/skimage/data/coins.png
    color image, https://raw.githubusercontent.com/scikit-image/scikit-image/master/skimage/data/astronaut.png
    '''
    print("gray image:")
    img = io.imread('../data/coins.png', as_grey=True)
    print('type:{}'.format(type(img)))
    print('shape:{}'.format(img.shape))
    print('data type:{}'.format(img.dtype))
    print('img:')
    print(img)
    print("color image:")
    img = io.imread('../data/astronaut.png')
    print('type:{}'.format(type(img)))
    print('shape:{}'.format(img.shape))
    print('data type:{}'.format(img.dtype))
    print('img:')
    print(img)

def change_image():
    '''
    @description: change image by two kind of methods, slice and mask
    '''
    astro=io.imread('../data/astronaut.png')
    astro_sq,astro_mask=np.copy(astro),np.copy(astro)
    astro_sq[50:100,50:100]=[0,255,0] # [R,G,B]
    mask=np.zeros(astro.shape[:2],dtype=bool)
    mask[300:350,300:350]=True
    astro_mask[mask]=[0,255,0]
    img=np.concatenate([astro_sq,astro_mask],axis=1)
    io.imsave('./3_1_change_image.png',arr=img)

def overlay_grid(filename,spacing=128):
    '''
    @description: overlay grid on a iamge with fixed spacing 
    '''
    img=io.imread(filename)
    rows,cols,_=img.shape
    line_rows,line_cols=np.arange(0,rows,spacing)[1:],np.arange(0,cols,spacing)[1:]
    for row,col in zip(line_rows,line_cols):
        img[row,:]=[0,0,255]
        img[:,col]=[0,0,255]
    io.imsave('./3_1_overlay_grid.png',img)
    

def main():
    '''
    @description: entrance bof the program
    '''
    random_image(filename='./3_1_random_image.png')
    image_is_numpy_ndarray()
    change_image()
    overlay_grid('../data/astronaut.png',spacing=128)


if __name__ == "__main__":
    main()