'''
@Filename: 3_4.py
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-13 13:53:27
@LastEditors: shifaqiang
@LastEditTime: 2019-05-15 10:20:05
@Software: Visual Studio Code
@Description: 
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')
import numpy as np
from skimage import io, morphology
from scipy import ndimage as ndi


def get_tax_rate_map():
    def tax(prices):
        return 1e4 + 0.05 * np.percentile(prices, 90)

    house_price_map = (np.random.rand(100, 100) + 0.5) * 1e6
    footprint = morphology.disk(radius=10)
    tax_rate_map = ndi.generic_filter(house_price_map,
                                      tax,
                                      footprint=footprint)
    plt.imshow(tax_rate_map)
    p = plt.colorbar()
    plt.savefig('./3_4_tax_rate_map.png')


class Conways_game_of_life(object):
    def __init__(self, board_size=(50, 50), n_generation=100, circle=False):
        self.board = np.random.randint(0, 2, size=board_size)
        self.n_generation = n_generation
        self.circle = circle

    def _next_generation_numpy(self):
        tmp = self.board[:-2, :-2]+self.board[:-2, 1:-1]+self.board[:-2, 2:] + \
            self.board[1:-1, :-2]+self.board[1:-1, 1:-1]+self.board[1:-1, 2:] + \
            self.board[2:, :-2]+self.board[2:, 1:-1]+self.board[2:, 2:]
        birth = (tmp == 3) & (self.board[1:-1, 1:-1] == 0)
        survive = ((tmp == 2) | (tmp == 3)) & (self.board[1:-1, 1:-1] == 1)
        self.board = np.zeros_like(self.board)
        self.board[1:-1, 1:-1][birth | survive] = 1

    def _next_generator_filter(self, values):
        center = values[len(values) // 2]
        neighbors_count = np.sum(values) - center
        if neighbors_count == 3 or (center and neighbors_count == 2):
            return 1
        else:
            return 0

    def _next_generation_convolution(self):
        if self.circle:
            mode_specific = 'wrap'
        else:
            mode_specific = 'constant'
        return ndi.generic_filter(self.board,
                                  self._next_generator_filter,
                                  size=3,
                                  mode=mode_specific)

    def go(self, method):
        if method == 'convolution':
            method_function = self._next_generation_convolution
        elif method == 'numpy':
            method_function = self._next_generation_numpy
        for n in range(self.n_generation):
            method_function()

    def __str__(self):
        return str(self.board)


def sobel_magnitude():
    hsobel = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]])
    vsobel = hsobel.T
    hsobel_r = np.ravel(hsobel)
    vsobel_r = np.ravel(vsobel)

    def sobel_magnitude_filter(values):
        h_edge = values @ hsobel_r
        v_edge = values @ vsobel_r
        return np.hypot(h_edge, v_edge)

    coins = io.imread('../data/coins.png')
    sobel_mag = ndi.generic_filter(coins, sobel_magnitude_filter, size=3)
    plt.imsave('./3_4_sobeled_coins.png', arr=sobel_mag, cmap='viridis')


if __name__ == "__main__":

    get_tax_rate_map()

    game = Conways_game_of_life()
    print('original life board:')
    print(game)
    game.go(method='numpy')
    print('numpy:')
    print(game)
    game.go(method='convolution')
    print('convolution:')
    print(game)

    sobel_magnitude()
