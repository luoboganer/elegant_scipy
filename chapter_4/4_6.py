'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-15 10:27:23
@LastEditors: shifaqiang
@LastEditTime: 2019-05-15 15:26:41
@Software: Visual Studio Code
@Description: 
'''

# packages for Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')
# packages for computing
from skimage import io
from scipy import fftpack
from sympy import factorint
import numpy as np

import time

def exec_time_of_fft():
    '''
    @Description: 
        观察不同平滑度输入数组的FFT计算时间，理解Cooley-Tukey算法的时间复杂度
        最优情况下n(long(n))，退化到最坏情况下n^2
    '''
    K=1000
    lengths=range(250,260)
    # 计算所有输入长度的平滑度
    smoothness=[max(factorint(i).keys()) for i in lengths]
    exec_time_of_fft_array=[]
    for i in lengths:
        '''traverse all possible input lengths, and record the minimum time in multiple times calculation'''
        z=np.random.random(i)
        times=[]
        for k in range(K):
            tic=time.monotonic()
            fftpack.fft(z)
            toc=time.monotonic()
            times.append(toc-tic)
        exec_time_of_fft_array.append(min(times))
    _,(ax0,ax1)=plt.subplots(2,1,sharex=True)
    ax0.stem(lengths,np.array(exec_time_of_fft_array)*1e6)
    ax0.set_xlabel('Length of input array')
    ax0.set_ylabel('Execution time [us]')
    ax1.stem(lengths,np.array(smoothness))
    ax1.set_ylabel('Smoothness of input length\n(lower is better)')
    plt.savefig('./4_6_execution_time_of_fft.png')

def frequency():
    '''
    @Description: 
        观察FFT输出的频率计算结果 
    '''
    N=10
    # 输入不变时输出为0
    x=np.ones(N)
    y=fftpack.fft(x)
    print('x={}'.format(x))
    print('fft(x)={}'.format(y))
    # 输入交替变化时输出中存在高频分量
    z=np.ones(N)
    z[::2]=-1
    print('Applying FFT to {}'.format(z))
    print('fft(z)={}'.format(fftpack.fft(z)))
    # fftreq函数显示那个频率需要特别关注
    print('fftpack.fftfreq(10)={}'.format(fftpack.rfftfreq(10)))
    # 输入实数序列时输出频谱是共轭对称的
    x=np.array([1,5,12,7,3,0,4,3,2,8])
    yx=fftpack.fft(x)
    np.set_printoptions(precision=2)
    print("Applying FFT to {}".format(x))
    print('Real part:\t{}'.format(yx.real))
    print('Imaginary part:\t{}'.format(yx.imag))

def fft_in_image(filename):
    '''
    @Description:
        观察图片中像素值随着x、y轴的空间变化频率
    '''
    image=io.imread(filename)
    M,N=image.shape
    print('shape of the image: {}, dtype: {}'.format(image.shape,image.dtype))
    F=fftpack.fftn(image)
    F_magnitude=np.abs(F)
    F_magnitude=fftpack.fftshift(F_magnitude)
    fig,axes=plt.subplots(2,2,figsize=(8,7))
    # left upper
    axes[0][0].imshow(image)
    axes[0][0].set_title('Original image')
    # left lower
    axes[1][0].imshow(np.log(1+F_magnitude),cmap='viridis',extent=(-N//2,N//2,-M//2,M//2))
    axes[1][0].set_title('Spectrum magnitude')
    # right upper, 将频谱中心的一块儿归零，当做过滤高频噪声，同时过滤高于98%分位的峰值
    K=40
    F_magnitude[M//2-K:M//2+K,N//2-K:N//2+K]=0
    peaks=F_magnitude<np.percentile(F_magnitude,98)
    peaks=fftpack.ifftshift(peaks)
    F_magnitude_dim=F.copy()
    F_magnitude_dim=F_magnitude_dim*peaks.astype(int)
    # 执行反向傅里叶变换还原图像
    image_filtered=np.real(fftpack.ifftn(F_magnitude_dim))
    axes[0][1].imshow(np.log10(1+np.abs(F_magnitude_dim)),cmap='viridis')
    axes[0][1].set_title('Spectrum after suppression')
    # right lower
    axes[1][1].imshow(image_filtered)
    axes[1][1].set_title('Reconstructed image')
    # save image
    fig.suptitle('result of fft filter for image reconstruct')
    plt.savefig('./4_6_result_of_fft_filter.png')

def slide_lobes_in_fft():
    '''
    @Description: 在矩形脉冲的傅里叶变换中，可以看到频谱中有许多旁瓣 
    '''
    x=np.zeros(500)
    x[100:150]=1
    fft_x=fftpack.fft(x)
    f,(ax0,ax1,ax2)=plt.subplots(3,1,sharex=True,figsize=(4.8,4.8))
    ax0.plot(x)
    ax0.set_ylim(-0.1,1.1)
    ax0.set_ylabel('Original Signal')
    ax1.plot(np.abs(fft_x))
    ax1.set_ylim(-5,55)
    ax1.set_ylabel('FFT of the signal')
    ax2.plot(fftpack.fftshift(np.abs(fft_x)))
    ax2.set_ylabel('shifted FFT of the signal')
    f.suptitle('slide_lobes_in_fft',size=18)
    plt.savefig('./4_6_slide_lobes_in_fft.png')

def kaiser(N=10,beta_max=5):
    '''观察kaiser窗口函数'''
    colormap=plt.cm.plasma
    norm=plt.Normalize(vmin=0,vmax=beta_max)
    _=plt.figure()
    lines=[plt.plot(np.kaiser(100,beta),color=colormap(norm(beta))) for beta in np.linspace(0,beta_max,N)]
    sm=plt.cm.ScalarMappable(cmap=colormap,norm=norm)
    sm._A=[]
    plt.colorbar(sm).set_label(r'kaiser $\beta$')
    plt.title('Kaiser window function',size='large')
    plt.savefig('./4_6_Kaiser_window funciton.png')

def acyclic():
    t=np.linspace(0,1,500)
    x=np.sin(49*np.pi*t)
    X=fftpack.fft(x)
    f,(ax0,ax1,ax2)=plt.subplots(3,1,figsize=(4.8,4.8))
    ax0.plot(x)
    ax0.set_ylim(-1.1,1.1)
    ax0.set_ylabel('Original signal')
    ax1.plot(fftpack.fftfreq(len(t)),np.abs(X))
    ax1.set_ylim(0,190)
    ax1.set_ylabel('FFT of signal')
    # 给傅里叶变换加上kaiser窗口函数
    win=np.kaiser(len(t),5)
    X_win=fftpack.fft(x*win)
    ax2.plot(fftpack.fftfreq(len(t)),np.abs(X_win))
    ax2.set_ylim(0,190)
    ax2.set_ylabel('FFT of signal\n'+r'with kaiser window($\beta$=5)')
    plt.suptitle('FFT on acyclic signal',size='large')
    plt.savefig('./4_6_fft_on_acyclic_signal.png')

def fft_with_window():
    slide_lobes_in_fft()
    kaiser()
    acyclic()


if __name__ == "__main__":
    exec_time_of_fft()
    frequency()
    # image_url = https://raw.githubusercontent.com/elegant-scipy/elegant-scipy/master/images/moonlanding.png
    fft_in_image('../data/moonlanding.png')
    fft_with_window()