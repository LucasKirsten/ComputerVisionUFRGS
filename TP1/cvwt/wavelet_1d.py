# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:36:22 2021

@author: kirstenl
"""

import numpy as np
from .utils import _upsample, _get_filters, _remove_borders


def wavelet_1d(x, wavelet='haar'):
    """
    Applies the 1D Wavelet Transform based on a given signal.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal.
    wavelet : str or list<numpy.ndarray>, optional
        Wavelet filter to be used. The default is haar.

    Returns
    -------
    cA : numpy.ndarray
        Approximation coefficients
    cD : numpy.ndarray
        Detail coefficients

    """
    if type(wavelet) == str:
        c, d, f, g = _get_filters(wavelet)
    elif type(wavelet) == list:
        c, d = wavelet
    else:
        raise Exception('Not a valid wavelet!')

    # convolve filters
    y0 = np.convolve(x, c)
    y1 = np.convolve(x, d)
    
    # downsize
    cA = y0[1::2]
    cD = y1[1::2]

    return cA, cD


def inv_wavelet_1d(x, D, wavelet='haar'):
    """
    Applies the 1D Inverse Wavelet Transform based on the coefficients.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal (approximation coefficients)
    D :
        Detail coefficients
    wavelet : str or list<numpy.ndarray>, optional
        Wavelet filter to be used. The default is haar.

    Returns
    -------
    w : numpy.ndarray
        The reconstructed signal

    """
    if type(wavelet) == str:
        c, d, f, g = _get_filters(wavelet)
    elif type(wavelet) == list:
        f, g = wavelet
    else:
        raise Exception('Not a valid wavelet!')
        
    # remove borders due convolution to match sizes
    if len(x)!=len(D):
        x = _remove_borders(x, size=len(D))

    # upsample
    u0 = _upsample(x)
    u1 = _upsample(D)

    # convolve filters
    w0 = np.convolve(u0, f, 'same')
    w1 = np.convolve(u1, g, 'same')

    w = w0 + w1
    
    # remove border to match expected output shape
    w = _remove_borders(w, num=len(f)-2)

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