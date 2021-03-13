# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:31:12 2021

@author: kirstenl
"""

import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from .wavelet_2d import wavelet_2d, inv_wavelet_2d


def _soft_thresholding(x, t):
    """
    Apply soft thresholding
    T(x) = sign(x) * (|x| - t)+

    """
    x = np.abs(x) - t
    return np.sign(x) * np.where(x >= 0, x, 0)


def apply_soft_thresholding(x, wavelet='haar', J=1, t=0, use_pywavelet=False):
    """
    Apply soft thresholding in 2D input image using Wavelet Transform.

    Parameters
    ----------
    x : np.array
        Input signal.
    wavelet : str or list<np.array>, optional
        Wavelet filter to be used. The default is haar.
    J : int, optional
        Maximum decomposition level. The default is 1.
    t : int, optional
        Valeu to be subtracted from soft thresholding function. The default value is 0.
    use_pywavelet : bool, optional
        If to use PyWavelet python package. The default is False.

    Returns
    -------
    cA : np.array
        Smooth Thresholded signal.

    """
    assert J > 0, 'J should be greater than 0'

    # going forward in the wavelet transform
    Dj = []
    cA = np.copy(x)
    for j in range(J):
        if use_pywavelet:
            cA, cD = pywt.dwt2(cA, wavelet)
        else:
            cA, cD = wavelet_2d(cA, wavelet)

        # apply soft threhsolding and save detail coefficients
        cD = _soft_thresholding(cD, t)
        Dj.append(cD)

    # returning to the filtered image
    for j, dj in enumerate(reversed(Dj)):
        if use_pywavelet:
            cA = pywt.idwt2((cA, dj), wavelet)
        else:
            cA = inv_wavelet_2d(cA, dj, wavelet)

    return cA


if __name__ == '__main__':
    # number of iterations
    J = 1

    # parameter for soft thersholding
    t = 50

    # define input image
    x = cv2.imread('./barbara.jpg')
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    # add noise to the image
    mean = 0
    var = 100
    sigma = np.sqrt(var)
    noise = x.astype('float32') + np.random.normal(mean, sigma, x.shape)
    noise = np.uint8(noise)

    # soft thresholding
    smooth = apply_soft_thresholding(noise, J=J, t=t)

    fig, axs = plt.subplots(1, 3)
    fig.suptitle(f'Own: Soft thresholding J={J}')
    axs[0].imshow(x)
    axs[0].set_title('Original image')

    axs[1].imshow(noise)
    axs[1].set_title('Noise image')

    axs[2].imshow(smooth)
    axs[2].set_title('Smooth image')

    # soft thresholding with PyWavelet
    smooth = apply_soft_thresholding(noise, J=J, t=t, use_pywavelet=True)

    fig, axs = plt.subplots(1, 3)
    fig.suptitle(f'PyWavelet: Soft thresholding J={J}')
    axs[0].imshow(x)
    axs[0].set_title('Original image')

    axs[1].imshow(noise)
    axs[1].set_title('Noise image')

    axs[2].imshow(smooth)
    axs[2].set_title('Smooth image')