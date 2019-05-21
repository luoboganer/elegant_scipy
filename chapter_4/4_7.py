'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-15 15:36:12
@LastEditors: shifaqiang
@LastEditTime: 2019-05-17 13:05:45
@Software: Visual Studio Code
@Description: 傅里叶变换在雷达数据分析中的实际应用
'''

# packages for Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')

import numpy as np
from scipy import spatial
from mpl_toolkits.mplot3d import Axes3D


class Ladar(object):
    '''
    @Description: 
        plot three different kinds of radar signals in a figure
    @Parameters: 
        actual_signal, actual radar signal from external data file
    '''

    def __init__(self, actual_signal_filename):
        data = np.load(actual_signal_filename)
        # 这里注意.npz文件为结构化numpy数组
        self.actual_signal = data['scan']['samples'][5, 14, :] * (2.5 / 8192)
        # parameters of radar hardware
        self.c = 3e8  # 光速
        fs = 78125
        ts = 1.0 / fs
        self.Teff, self.Beff = 2048.0 * ts, 100e6
        S = self.Beff / self.Teff
        # some random object
        time_lenght = 2048
        R = np.array([100, 137, 154, 159,
                      180])  # distance from the objects to radar
        M = np.array([0.33, 0.2, 0.9, 0.02, 0.1])  # size of the objects
        P = np.array([0, np.pi / 2, np.pi / 3, np.pi / 5, np.pi / 6])
        t = np.arange(time_lenght) * ts  # time of sampling
        self.t_ms = t * 1e3  # from s to ms
        fd = 2 * S * R / 3e8  # frequency difference of this five objects
        self.signals = np.cos(2 * np.pi * fd * t[:, np.newaxis] + P)
        self.v_signal_object_0 = self.signals[:,
                                              0]  # radar signal of the first objects
        self.v_sim = np.sum(
            M * self.signals, axis=1
        )  # compute signal sum weighted by size of the different object
        self.length = len(self.v_signal_object_0) // 2
        self.w = np.kaiser(len(self.v_signal_object_0), 6.1)

    def plot_signal(self):
        _, axes = plt.subplots(3, 1, sharex=True, figsize=(6.4, 4.8))
        axes[0].plot(self.t_ms, self.v_signal_object_0)
        axes[0].set_ylabel(r'$V_{\mathrm{object0}}(t) [V]$')
        axes[1].plot(self.t_ms, self.v_sim)
        axes[1].set_ylabel(r'$V_{\mathrm{sim}}(t) [V]$')
        axes[2].plot(self.t_ms, self.actual_signal)
        axes[2].set_xlabel('Time [ms]')
        axes[2].set_ylabel(r'$V_{\mathrm{actual}} [V]$')
        # axes[2].set_xlim(0,t_ms[-1])
        plt.suptitle('different kinds of radar signals', size='x-large')
        plt.savefig('./4_7_radar_signals.png')

    def distance_tracking(self):
        # FFT
        V_signal_object_0 = np.fft.fft(self.v_signal_object_0)
        V_sim = np.fft.fft(self.v_sim)
        V_actual = np.fft.fft(self.actual_signal)
        # plot figure
        _, axes = plt.subplots(3, 1, sharex=True, figsize=(4.8, 3.2))
        with plt.style.context('../style/thinner.mplstyle'):
            axes[0].plot(np.abs(V_signal_object_0[:self.length]))
            axes[0].set_ylabel(r'$|V_{\mathrm{object0}}|$')
            axes[0].set_xlim(0, self.length)
            axes[0].set_ylim(0, 1100)
            axes[1].plot(np.abs(V_sim[:self.length]))
            axes[1].set_ylabel(r'$|V_{\mathrm{sim}}|$')
            axes[1].set_ylim(0, 1000)
            axes[2].plot(np.abs(V_actual[:self.length]))
            axes[2].set_ylim(0, 750)
            axes[2].set_xlabel(r'FFT component $n$')
            axes[2].set_ylabel(r'$|V_{\mathrm{actual}}|$')
            for ax in axes:
                ax.grid()
        # axes[2].set_xlim(0,t_ms[-1])
        plt.suptitle('distance tracking by fft of radar signals', size='large')
        plt.savefig('./4_7_distance_tracking_by_radar_signals_fft.png')

    def plot_signal_with_win(self):
        # plot figure
        data = [(self.v_signal_object_0, r'$|V_{\mathrm{object0}} [V]|$'),
                (self.v_sim, r'$|V_{\mathrm{sim}} [V]|$'),
                (self.actual_signal, r'$|V_{\mathrm{actual}} [V]|$')]
        _, axes = plt.subplots(3, 1, sharex=True, figsize=(4.8, 3.2))
        for n, (signal, label) in enumerate(data):
            with plt.style.context('../style/thinner.mplstyle'):
                axes[n].plot(self.t_ms, self.w * signal)
                axes[n].set_ylabel(label)
                axes[n].grid()
        axes[2].set_xlim(0, self.t_ms[-1])
        axes[2].set_xlabel('Time [ms]')
        plt.suptitle('different kinds of radar signals with Kaiser window',
                     size='large')
        plt.savefig('./4_7_radar_signals_win.png')

    def _dB(self, y):
        y = np.abs(y)
        y /= y.max()
        return 20 * np.log10(y)

    def _log_plot_normalized(self, ax, x, y, ylabel):
        ax.plot(x, self._dB(y))
        ax.set_ylabel(ylabel)
        ax.grid()

    def distance_tracking_log(self):
        # FFT
        V_signal_object_0 = np.fft.fft(self.v_signal_object_0)
        V_sim = np.fft.fft(self.v_sim)
        V_actual = np.fft.fft(self.actual_signal)
        # plot figure
        rng = np.arange(self.length) * self.c / 2 / self.Beff
        _, axes = plt.subplots(3, 1, figsize=(4.8, 4.8))
        with plt.style.context('../style/thinner.mplstyle'):
            self._log_plot_normalized(axes[0], rng,
                                      V_signal_object_0[:self.length],
                                      r'$|V_{\mathrm{object0}} [dB]|$')
            self._log_plot_normalized(axes[1], rng, V_sim[:self.length],
                                      r'$|V_{\mathrm{sim}} [dB]|$')
            self._log_plot_normalized(axes[2], rng, V_actual[:self.length],
                                      r'$|V_{\mathrm{actual}} [dB]|$')

        axes[0].set_xlim(0, 300)
        axes[1].set_xlim(0, 300)
        axes[2].set_xlim(0, len(V_actual) // 2)
        axes[2].set_xlabel('range')

        plt.suptitle('log range by fft of radar signals', size='large')
        plt.savefig('./4_7_log_range_by_radar_signals_fft.png')

    def distance_tracking_log_win(self):
        v_signal_win = np.fft.fft(self.w * self.v_signal_object_0)
        v_sim_wim = np.fft.fft(self.w * self.v_sim)
        v_actual_win = np.fft.fft(self.w * self.actual_signal)
        # plot figure
        rng = np.arange(self.length) * self.c / 2 / self.Beff
        _, axes = plt.subplots(3, 1, figsize=(4.8, 4.8))
        with plt.style.context('../style/thinner.mplstyle'):
            self._log_plot_normalized(axes[0], rng, v_signal_win[:self.length],
                                      r'$|V_{\mathrm{object0,win}} [dB]|$')
            self._log_plot_normalized(axes[1], rng, v_sim_wim[:self.length],
                                      r'$|V_{\mathrm{sim,win}} [dB]|$')
            self._log_plot_normalized(axes[2], rng, v_actual_win[:self.length],
                                      r'$|V_{\mathrm{actual,win}} [dB]|$')

        axes[0].set_xlim(0, 300)
        axes[1].set_xlim(0, 300)
        axes[1].annotate("New, previously unseen!", (160, -35),
                         xytext=(10, 15),
                         textcoords=("offset points"),
                         color='red',
                         size='small',
                         arrowprops=dict(width=0.5,
                                         headwidth=3,
                                         headlength=4,
                                         fc='k',
                                         shrink=0.1))
        axes[2].set_xlabel('range')
        plt.suptitle('log range by fft with Kaiser window of radar signals',
                     size='large')
        plt.savefig('./4_7_log_range_by_radar_signals_fft_with_win.png')


class Lader_2D(object):
    def __init__(self, filename):
        data = np.load(filename)
        # 这里注意.npz文件为结构化numpy数组
        self.v = data['scan']['samples'] * (2.5 / 8192)
        self.length = self.v.shape[-1]
        self.w = np.hanning(self.length + 1)[:-1]
        self.V = np.fft.fft(self.v * self.w,
                            axis=2)[::-1, :, :self.length // 2]
        self.contours = np.arange(-40, 1, 2)

    def _dB(self, y):
        y = np.abs(y)
        y /= y.max()
        return 20 * np.log10(y)

    def _plot_slice(self, ax, radar_slice, title, xlabel, ylabel):
        ax.contour(self._dB(radar_slice), self.contours, cmap='magma_r')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_facecolor(plt.cm.magma_r(-40))

    def plot_fft(self):
        _, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
        labels = ('Range', 'Azimuth', 'Elevation')
        with plt.style.context('../style/thinner.mplstyle'):
            self._plot_slice(axes[0], self.V[:, :, 250], 'Range=250',
                             'Azimuth', 'Elevation')
            self._plot_slice(axes[1], self.V[:, 3, :], 'Azimuth=3', 'Range',
                             'Elevation')
            self._plot_slice(axes[2], self.V[6, :, :].T, 'Elevation=3',
                             'Azimuth', 'Range')
        # plt.suptitle('FFT of radar data in different slice')
        plt.tight_layout()
        plt.savefig('./4_7_fft_of_radar_data_in_different_slice.png')

    def plt_3d(self):
        r = np.argmax(self.V, axis=2)
        el, az = np.meshgrid(*[np.arange(s) for s in r.shape], indexing='ij')
        axis_labels = ['Range', 'Azimuth', 'Elevation']
        coords = np.column_stack((el.flat, az.flat, r.flat))
        d = spatial.Delaunay(coords[:, :2])
        simplexes = coords[d.vertices]
        coords = np.roll(coords, shift=-1, axis=1)
        axis_labels = np.roll(axis_labels, shift=-1)
        _, ax = plt.subplots(1,
                             1,
                             figsize=(4.8, 4.8),
                             subplot_kw=dict(projection='3d'))
        with plt.style.context('../style/thinner.mplstyle'):
            ax.plot_trisurf(*coords.T, triangles=d.vertices, cmap='magma_r')
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_zlabel(axis_labels[2], labelpad=-3)
            ax.set_xticks([0, 5, 10, 15])
        ax.view_init(azim=-50)
        plt.savefig('./4_7_3d_fft.png')


if __name__ == "__main__":
    # 1-d data
    ladar = Ladar(actual_signal_filename='../data/radar_scan_0.npz')
    ladar.plot_signal()
    ladar.plot_signal_with_win()
    ladar.distance_tracking()
    ladar.distance_tracking_log()
    ladar.distance_tracking_log_win()
    # 2-d data
    ladar_2d = Lader_2D('../data/radar_scan_1.npz')
    ladar_2d.plot_fft()
    ladar_2d.plt_3d()