# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:30:16 2021

@author: kirstenl
"""

import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from .wavelet_1d import wavelet_1d, inv_wavelet_1d
from .utils import _get_rows, _get_cols, _plot_wavelet2d


def wavelet_2d(x, wavelet='haar'):
    # TODO: add doc

    # convert to gray scale
    if len(x.shape) > 2:
        x = np.mean(x, axis=-1)

    # function to apply wavelet 1d
    _wavelet1d = lambda inp: wavelet_1d(inp, wavelet)

    # operate in lines
    lines = _get_rows(x)
    wl = np.stack(map(_wavelet1d, lines), axis=-1)

    al, dl = wl[0], wl[1]

    # operate in columns
    a_cols = _get_rows(al)
    d_cols = _get_rows(dl)

    wac = np.stack(map(_wavelet1d, a_cols), axis=-1)
    wdc = np.stack(map(_wavelet1d, d_cols), axis=-1)

    # split componenets
    HL, HH = wdc[0], wdc[1]
    LL, LH = wac[0], wac[1]

    return LL, (LH, HL, HH)


def inv_wavelet_2d(x, D, wavelet='haar'):
    # TODO: add doc

    # aliases
    LL, (LH, HL, HH) = x, D

    # function to apply inverse wavelet 1d
    _inv_wavelet1d = lambda xx, dd: inv_wavelet_1d(xx, dd, wavelet)

    # columns inverse
    # first branch: LL + LH
    colsLL = _get_cols(LL)
    colsLH = _get_cols(LH)

    c1 = np.array([_inv_wavelet1d(cLL, cLH) for cLL, cLH in zip(colsLL, colsLH)])

    # second branch: HL + HH
    colsHL = _get_cols(HL)
    colsHH = _get_cols(HH)

    c2 = np.array([_inv_wavelet1d(cHL, cHH) for cHL, cHH in zip(colsHL, colsHH)])

    # lines inverse
    rowsC1 = _get_cols(c1)
    rowsC2 = _get_cols(c2)

    inverse = np.array([_inv_wavelet1d(cc1, cc2) for cc1, cc2 in zip(rowsC1, rowsC2)])

    return inverse


if __name__ == '__main__':
    # define input image
    x = cv2.imread('./barbara.jpg')
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    # define iterations
    J = 3

    # going forward in the wavelet transform
    Aj, Dj = [], []
    cA = np.copy(x)
    for j in range(J):
        cA, cD = wavelet_2d(cA)
        Aj.append(cA);
        Dj.append(cD)
        _plot_wavelet2d(cA, cD, f'Own: Wavelet {j + 1}')

    # returning to the original image
    for j, dj in enumerate(reversed(Dj)):
        cA = inv_wavelet_2d(cA, dj)
        plt.figure(figsize=(10, 10))
        plt.imshow(cA)
        plt.title(f'Own: Inverse Wavelet {len(Aj) - j}')

    # %% comparing with Pywavelet

    # going forward in the wavelet transform
    Aj, Dj = [], []
    cA = np.copy(np.mean(x, axis=-1))
    for j in range(J):
        cA, cD = pywt.dwt2(cA, 'haar')
        Aj.append(cA);
        Dj.append(cD)
        _plot_wavelet2d(cA, cD, f'PyWavelet: Wavelet {j + 1}')

    # returning to the original image
    for j, dj in enumerate(reversed(Dj)):
        cA = pywt.idwt2((cA, dj), 'haar')
        plt.figure(figsize=(10, 10))
        plt.imshow(cA)
        plt.title(f'PyWavelet: Inverse Wavelet {len(Aj) - j}')