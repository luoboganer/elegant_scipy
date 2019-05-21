'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-13 01:15:20
@LastEditors: shifaqiang
@LastEditTime: 2019-05-15 01:15:34
@Software: Visual Studio Code
@Description: 
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')
import numpy as np
from scipy import fftpack
from scipy.io import wavfile
from skimage import util
from scipy import signal


def plot_sin_wave():
    f, f_s = 10, 100  # frequency, frequency_of_samples
    t = np.linspace(0, 2, 2 * f_s, endpoint=False)
    x = np.sin(f * 2 * np.pi * t)
    _, ax = plt.subplots(2, 1, figsize=(6.4, 7.2))
    # original signal
    ax[0].plot(t, x)
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Signal amplitude')
    # fft
    X = fftpack.fft(x)
    freqs = fftpack.fftfreq(len(x)) * f_s
    ax[1].stem(freqs, np.abs(X))
    ax[1].set_xlabel('Frequency in Hertz [Hz]')
    ax[1].set_ylabel('Frequency Domain (spectrum) Magnitude')
    ax[1].set_xlim(-f_s / 2, f_s / 2)
    ax[1].set_ylim(-5, 110)
    plt.savefig('./4_1_sin_wave.png')


def audio_process(filename):
    # 载入音频数据并画出波形
    rate, audio = wavfile.read(filename)
    audio = np.mean(audio, axis=1)  # 左右声道取平均值变为单声道
    N = audio.shape[0]
    L = N / rate
    print('audio length {:.2f} seconds'.format(L))
    _ = plt.figure(figsize=(6.4, 3.6))
    plt.plot(np.arange(N) / rate, audio)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [Unknown]')
    plt.savefig('./4_1_audio_wave.png')

    # 短时间傅里叶变换
    M = 1024
    slices = util.view_as_windows(audio, window_shape=(M, ), step=100)
    print('Audio shape:{}, Sliced audio shape:{}'.format(
        audio.shape, slices.shape))
    win = np.hamming(M + 1)[:-1]  # 加窗函数
    slices = slices * win
    slices = slices.T
    print('Shape of slices:', slices.shape)
    spectrum = np.abs(np.fft.fft(slices, axis=0)[:M // 2 + 1:-1])
    _ = plt.figure()
    S = 20 * np.log10(spectrum / np.max(spectrum))  # 计算分贝数
    plt.imshow(S,
               origin='lower',
               cmap='viridis',
               extent=(0, L, 0, rate / 2 / 1e3))
    plt.axis('tight')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [KHz]')
    plt.savefig('./4_1_audio_spectrum_manual.png')

    # scipy中的短视傅里叶变换
    freqs, times, sx = signal.spectrogram(audio,
                                          fs=rate,
                                          window='hanning',
                                          nperseg=M,
                                          noverlap=M - 100,
                                          detrend=False,
                                          scaling='spectrum')
    _ = plt.figure()
    plt.pcolormesh(times, freqs / 1e3, 10 * np.log10(sx), cmap='viridis')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [KHz]')
    plt.savefig('./4_1_audio_spectrum_sicpy.png')


if __name__ == "__main__":
    plot_sin_wave()
    audio_process('../data/nightingale.wav')

#%%
# player for wav file
from IPython.display import Audio
Audio('../data/nightingale.wav')
