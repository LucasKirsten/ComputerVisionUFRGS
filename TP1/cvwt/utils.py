# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:35:28 2021

@author: kirstenl
"""

import pywt
import matplotlib.pyplot as plt
import numpy as np
from math import log10, sqrt


def _remove_borders(x, size=None, num=None):
    """
    Remove border on the input signal.
    If size is given, the function return x with the shape=size
    If num is given, it is removed (num/2) from both sides of x

    """
    if size:
        num = len(x) - int(size)

    init = int(np.floor(num / 2))
    end = int(-np.ceil(num / 2))
    if end == 0: end = len(x)

    return x[init:end]


def _get_rows(x):
    """
    Return 2D signal rows.

    """
    return [x[i, :] for i in range(x.shape[0])]


def _get_cols(x):
    """
    Return 2D signal columns.

    """
    return [x[:, i] for i in range(x.shape[1])]


def _plot_wavelet2d(A, D, cmap=None):
    """
    Plot 2D Wavelet Transform results.

    Parameters
    ----------
    A : list<numpy.ndarray>
        List of smooth signals.
    D : list<numpy.ndarray>
        List of detail coefficients.

    """

    if len(A.shape) > 2:
        channels = A.shape[-1]
    else:
        channels = 1
        A = A[..., np.newaxis]

    for ch in range(channels):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f'Channel: {ch}')
        axs[0][0].imshow(A[..., ch], cmap=cmap)
        axs[0][0].set_title('LL')

        axs[0][1].imshow(D[0][..., ch], cmap=cmap)
        axs[0][1].set_title('LH')

        axs[1][0].imshow(D[1][..., ch], cmap=cmap)
        axs[1][0].set_title('HL')

        axs[1][1].imshow(D[2][..., ch], cmap=cmap)
        axs[1][1].set_title('HH')


def _upsample(x, ratio=2):
    """
    Parameters
    ----------
    x : numpy.ndarray
        Input signal.
    ratio : int, optional
        Upsample ratio. The default is 2.

    Returns
    -------
    y : numpy.ndarray
        Upsampled signal.

    """
    y = np.zeros((len(x) * ratio,))
    y[::ratio] = x

    return y


def _get_filters(wavelet):
    """
    Return filters for wavelet family.

    """
    try:
        wavelet = pywt.Wavelet(wavelet)
    except:
        raise Exception('Not a valid wavelet! Choose among: ', pywt.wavelist())

    return [np.array(f) for f in wavelet.filter_bank]


def psnr(original_img, noise_img):
    """
    Get the Peak Signal tp Noise Ratio (PSNR) value between two images

    Parameters
    ----------
    original_img : numpy.ndarray
        Original image
    noise_img : numpy.ndarray
        Noisy image

    Returns
    -------
    psnr : float
        The psnr value that represents the ratio between the maximum possible power of an image
        and the power of corrupting noise that affects the quality of its representation

    """
    mse = np.mean((np.array(original_img, dtype=np.float32) - np.array(noise_img, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
