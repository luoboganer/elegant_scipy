'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-15 22:14:20
@LastEditors: shifaqiang
@LastEditTime: 2019-05-15 22:23:56
@Software: Visual Studio Code
@Description: practice of image convolution by FTT
'''

import numpy as np
from scipy import signal

if __name__ == "__main__":
    x=np.random.random(size=(50,50))
    filter_kernel=np.ones(shape=(5,5))
    z_np=signal.convolve2d(x,filter_kernel)
    L=x.shape[0]+filter_kernel.shape[0]-1
    Px=L-x.shape[0]
    Py=L-filter_kernel.shape[0]
    xx=np.pad(x,((0,Px),(0,Px)),mode='constant')
    yy=np.pad(filter_kernel,((0,Py),(0,Py)),mode='constant')
    z_fft=np.fft.ifft2(np.fft.fft2(xx)*np.fft.fft2(yy)).real
    print('Resulting shape: {}'.format(z_fft.shape))
    print('Results are equal? {}'.format(np.allclose(z_fft,z_np)))