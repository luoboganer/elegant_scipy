'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-20 20:26:43
@LastEditors: shifaqiang
@LastEditTime: 2019-05-20 23:24:11
@Software: Visual Studio Code
@Description: 
'''

import numpy as np
import toolz as tz


def demo_of_file_row(filename):
    '''
    @Description:
        demonstration of load a file from disk by row
    '''
    with open(filename, 'r') as f:
        sum_of_means = 0
        for line in f:
            sum_of_means += np.mean(np.fromstring(line, dtype=int, sep='\t'))
        print(f"The sum of mean: {sum_of_means}")


def load_from_txt_and_compute(filename):
    expr = np.loadtxt(filename)
    logexpr = np.log(expr + 1)
    print(f'The sum of the mean(axis=0):\n{np.mean(logexpr,axis=0)}')


def log_all_standard(input):
    output = []
    for elem in input:
        output.append(np.log(elem))
    return output


def log_all_streaming(input_stream):
    for elem in input_stream:
        yield np.log(elem)


def demo_of_yeild():
    '''
    @Description: 
        yeild 关键字使得Python可以流式处理大数据问题，有效节省内存 
    '''
    # 设定随机数种子以求稳定结果
    np.random.seed(seed=7)
    # 设置输出显示精度
    np.set_printoptions(precision=3, suppress=True)

    arr = np.random.rand(1000) + 0.5
    result_batch = sum(log_all_standard(arr))
    print(f'Batch result: {result_batch}')
    result_stream = sum(log_all_streaming(arr))
    print(f'Stream result: {result_stream}')


def tsv_line_to_arry(line):
    lst = [float(elem) for elem in line.rstrip().split('\t')]
    return np.array(lst)


def readtsv(filename):
    print('Start read tsv filename...')
    with open(filename) as fin:
        for i, line in enumerate(fin):
            print(f'reading line {i}')
            yield tsv_line_to_arry(line)
    print('Finished read tsv filename!')


def add1(array_iter):
    'Starting adding 1 ...'
    for i, arr in enumerate(array_iter):
        print(f'adding 1 to line {i}')
        yield arr + 1
    print('Finished add 1')


def log(array_iter):
    print('Starting log ...')
    for i, arr in enumerate(array_iter):
        print(f'taking log of array {i}')
        yield np.log(arr)
    print('Finished log!')


def running_mean(arrays_iter):
    print('Starting running mean ...')
    for i, arr in enumerate(arrays_iter):
        if i == 0:
            mean = arr
        else:
            mean += (arr - mean) / (i + 1)  # 均值mean的增量算法-
        print(f'adding line {i} to the running mean')
    print('Returning mean ...')
    return mean


def experiment_of_yeild(filename):
    print('Creating lines iterator ...')
    lines = readtsv(filename)
    print('Creating loglines iterator ...')
    loglines = log(add1(lines))
    print('Computing mean ...')
    mean = running_mean(loglines)
    print(f'The mean log-row is {mean}')


def demo_of_toolz(filename):
    print('Demonstration of compute mean by using toolz ...')
    mean = tz.pipe(filename, readtsv, add1, log, running_mean)
    # 这等价于 running_mean(log(add1(readtsv(filename))))
    print(f'Done, the sum of mean is\n{mean}')


def demo_of_curry():
    @tz.curry
    def add(x, y):
        return x + y

    add_partial = add(2)
    result = add_partial(5)
    print(f'curried function(add(x,y))\n\tp_add=add(2)\n\tp_add(5)={result}')


if __name__ == "__main__":
    expr_filename = '../data/expr.tsv'
    demo_of_file_row(expr_filename)
    load_from_txt_and_compute(expr_filename)
    demo_of_yeild()
    experiment_of_yeild(expr_filename)
    demo_of_toolz(expr_filename)
    demo_of_curry()