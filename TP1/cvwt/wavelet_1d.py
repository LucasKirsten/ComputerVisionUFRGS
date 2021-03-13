# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:36:22 2021

@author: kirstenl
"""

import numpy as np
from .utils import _upsample, _get_haar


def wavelet_1d(x, wavelet='haar'):
    # TODO: add doc

    if wavelet == 'haar':
        c, d, f, g = _get_haar()
    elif type(wavelet) == list:
        c, d = wavelet
    else:
        raise Exception('Not a valid wavelet!')

    # convolve filters
    y0 = np.convolve(x, c)
    y1 = np.convolve(x, d)

    # downsize
    v0 = y0[1::2]
    v1 = y1[1::2]

    return v0, v1


def inv_wavelet_1d(x, D, wavelet='haar'):
    # TODO: add doc

    if wavelet == 'haar':
        c, d, f, g = _get_haar()
    elif type(wavelet) == list:
        f, g = wavelet
    else:
        raise Exception('Not a valid wavelet!')

    # eliminate extra zero due convolution
    if len(x) > len(D):
        x = x[:-1]

    # upsample
    u0 = _upsample(x)
    u1 = _upsample(D)

    # convolve filters
    w0 = np.convolve(u0, f)[:-1]
    w1 = np.convolve(u1, g)[:-1]

    w = w0 + w1

    return w


if __name__ == '__main__':
    # define iterations
    J = 3

    # input signal
    x = np.array([-2, 1, 3, 2, -3, 4])

    # going forward in the wavelet transform
    print('Wavelet:')
    Aj, Dj = [], []
    cA = np.copy(x)
    for j in range(J):
        cA, cD = wavelet_1d(cA)
        Aj.append(cA);
        Dj.append(cD)
        print(f'A{j + 1}: ', cA)
        print(f'D{j + 1}: ', cD)
        inv = inv_wavelet_1d(cA, cD)
        print(f'Inverse(A{j + 1}): ', inv)

    # returning to the original signal
    print('\nInverse Wavelet:')
    for j, dj in enumerate(reversed(Dj)):
        cA = inv_wavelet_1d(cA, dj)
        print(f'Inverse(A{len(Aj) - j}): ', cA)