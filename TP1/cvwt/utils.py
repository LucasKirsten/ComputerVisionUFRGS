# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:35:28 2021

@author: kirstenl
"""

import matplotlib.pyplot as plt
import numpy as np


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


def _plot_wavelet2d(A, D, title='', cmap=None):
    """
    Plot 2D Wavelet Transform results.

    Parameters
    ----------
    A : list<numpy.ndarray>
        List of smooth signals.
    D : list<numpy.ndarray>
        List of detail coefficients.
    title : str, optional
        Plot title. The default is \'\'.

    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title)
    axs[0][0].imshow(A, cmap=cmap)
    axs[0][0].set_title('LL')

    axs[0][1].imshow(D[0], cmap=cmap)
    axs[0][1].set_title('LH')

    axs[1][0].imshow(D[1], cmap=cmap)
    axs[1][0].set_title('HL')

    axs[1][1].imshow(D[2], cmap=cmap)
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


def _get_haar():
    """
    Return haar filters.

    """
    c = 1 / np.sqrt(2) * np.array([1., 1.])
    d = 1 / np.sqrt(2) * np.array([-1., 1.])
    f = 1 / np.sqrt(2) * np.array([1., 1.])
    g = 1 / np.sqrt(2) * np.array([1., -1.])

    return c, d, f, g