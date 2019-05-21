'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-20 22:30:25
@LastEditors: shifaqiang
@LastEditTime: 2019-05-21 09:48:36
@Software: Visual Studio Code
@Description: 
'''

# packages for Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')

import numpy as np
import toolz as tz
from toolz import curried as cur
from sklearn import datasets, decomposition


def is_sequence(line: str) -> bool:
    line = line.rstrip()
    return len(line) > 0 and not line.startswith('>')


def reads_to_kmers(reads_iter, k=7):
    for read in reads_iter:
        for start in range(0, len(read) - k):
            yield read[start:start + k]  # this is a generator


def kmer_counter(kmer_iter):
    counts = {}
    for kmer in kmer_iter:
        if kmer not in counts:
            counts[kmer] = 0
        counts[kmer] += 1
    return counts


def reads_from_filename_with_pure_python(filename):
    with open(filename) as fin:
        reads = filter(is_sequence, fin)
        kmers = reads_to_kmers(reads)
        counts = kmer_counter(kmers)
    return counts


def integer_histogram(counts,
                      ax,
                      normed=True,
                      title=None,
                      xlim=[],
                      ylim=[],
                      *args,
                      **kwargs):
    hist = np.bincount(counts)
    if normed:
        hist = hist / np.sum(hist)
    ax.plot(np.arange(len(hist)), hist, *args, **kwargs)
    ax.set_xlabel('counts')
    ax.set_ylabel('frequency')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if not title is None:
        ax.set_title(title)


def kmer():
    counts = reads_from_filename_with_pure_python('../data/sample.fasta')
    counts_arr = np.fromiter(counts.values(), dtype=int, count=len(counts))
    _, axes = plt.subplots(2, 1, figsize=(5.2, 3.2 * 2))
    integer_histogram(counts_arr, axes[0], title='Pure Python', xlim=[-1, 250])
    # help information of tz.slide_window
    print(tz.sliding_window.__doc__)
    # 调用toolz.curried.c实现counts统计
    k = 7
    counts_tz = tz.pipe('../data/sample.fasta', open, cur.filter(is_sequence),
                        cur.map(str.rstrip), cur.map(cur.sliding_window(k)),
                        tz.concat, cur.map(''.join), tz.frequencies)
    counts_arr_tz = np.fromiter(counts.values(), dtype=int, count=len(counts))
    integer_histogram(counts_arr_tz, axes[1], title='Toolz', xlim=[-1, 250])
    plt.tight_layout()
    plt.savefig('./8_3_frequency_counts.png')


def streaming_PCA(samples, n_components=2, batch_size=100):
    ipca = decomposition.IncrementalPCA(n_components=n_components,
                                        batch_size=batch_size)
    tz.pipe(samples, cur.partition(batch_size), cur.map(np.array),
            cur.map(ipca.partial_fit), tz.last)
    return ipca


def array_from_txt(line, sep=',', dtype=np.float):
    return np.array(line.rstrip().split(sep), dtype=dtype)


def iris():
    reshape = tz.curry(np.reshape)
    with open('../data/iris.csv') as fin:
        pca_obj = tz.pipe(fin, cur.map(array_from_txt), streaming_PCA)

    with open('../data/iris.csv') as fin:
        components = tz.pipe(fin, cur.map(array_from_txt),
                             cur.map(reshape(newshape=(1, -1))),
                             cur.map(pca_obj.transform), np.vstack)
    print(f'The shape of components is {components.shape}')
    iris_types = np.loadtxt('../data/iris-target.csv')
    _, axes = plt.subplots(2, 1, figsize=(4.8, 3.2 * 2))
    axes[0].scatter(*components.T, c=iris_types, cmap='viridis')
    axes[0].set_title('Streaming PCA')
    iris = np.loadtxt('../data/iris.csv', delimiter=',')
    components_stardanr = decomposition.PCA(n_components=2).fit_transform(iris)
    axes[1].scatter(*components_stardanr.T, c=iris_types, cmap='viridis')
    axes[1].set_title('Stardand PCA')
    plt.tight_layout()
    plt.savefig('./8_3_iris_PCA.png')


if __name__ == "__main__":
    kmer()
    iris()