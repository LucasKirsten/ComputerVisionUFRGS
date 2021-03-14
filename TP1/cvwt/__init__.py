from .wavelet_1d import wavelet_1d, inv_wavelet_1d
from .wavelet_2d import wavelet_2d, inv_wavelet_2d
from .compact_img import compact_image
from .soft_thresholding import apply_soft_thresholding
from .utils import _plot_wavelet2d as plot_dwt2

import numpy as np


def dwt(x, J=1, c=None, d=None, wavelet='haar'):
    """
    1D Wavelet Transform.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal.
    J : int, optional
        Maximum decomposition level. The default is 1.
    c : numpy.ndarray, optional
        Low pass filter to be used. The default is None.
    d : numpy.ndarray, optional
        High pass filter to be used. The default is None.
    wavelet : str, optional
        Wavelet filter to be used. The default is haar.

    Returns
    -------
    cA : numpy.ndarray
        Smooth signal.
    Dj : list<numpy.ndarray>
        Detail coefficients.

    """
    assert J > 0, 'J should be greater than 0!'

    if c and d:
        wavelet = [c, d]

    Dj = []
    cA = np.copy(x)
    for j in range(J):
        cA, cD = wavelet_1d(cA, wavelet=wavelet)
        Dj.append(cD)

    return cA, Dj


def idwt(cA, cD, f=None, g=None, wavelet='haar'):
    """
    1D Inverse Wavelet Transform.

    Parameters
    ----------
    cA : numpy.ndarray
        Input signal.
    cD : list<numpy.ndarray>
        List of detail coefficient for each level of decomposition.
    f : numpy.ndarray, optional
        Low pass filter to be used. The default is None.
    g : numpy.ndarray, optional
        High pass filter to be used. The default is None.
    wavelet : str, optional
        Wavelet filter to be used. The default is haar.

    Returns
    -------
    cA : numpy.ndarray
        Inverse signal.

    """
    if f and g:
        wavelet = [f, g]

    for j, dj in enumerate(reversed(cD)):
        cA = inv_wavelet_1d(cA, dj, wavelet=wavelet)

    return cA


def dwt2(x, J=1, c=None, d=None, wavelet='haar'):
    """
    2D Wavelet Transform.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal.
    J : int, optional
        Maximum decomposition level. The default is 1.
    c : numpy.ndarray, optional
        Low pass filter to be used. The default is None.
    d : numpy.ndarray, optional
        High pass filter to be used. The default is None.
    wavelet : str, optional
        Wavelet filter to be used. The default is haar.

    Returns
    -------
    cA : numpy.ndarray
        Smooth signal.
    Dj : list<numpy.ndarray>
        Detail coefficients.

    """
    assert J > 0, 'J should be greater than 0!'

    if c and d:
        wavelet = [c, d]

    Dj = []
    cA = np.copy(x)
    for j in range(J):
        cA, cD = wavelet_2d(cA, wavelet=wavelet)
        Dj.append(cD)

    return cA, Dj


def idwt2(cA, cD, f=None, g=None, wavelet='haar'):
    """
    2D Inverse Wavelet Transform.

    Parameters
    ----------
    cA : numpy.ndarray
        Input signal.
    cD : list<numpy.ndarray>
        List of detail coefficient for each level of decomposition.
    f : numpy.ndarray, optional
        Low pass filter to be used. The default is None.
    g : numpy.ndarray, optional
        High pass filter to be used. The default is None.
    wavelet : str, optional
        Wavelet filter to be used. The default is haar.

    Returns
    -------
    cA : numpy.ndarray
        Inverse signal.

    """
    if f and g:
        wavelet = [f, g]

    for j, dj in enumerate(reversed(cD)):
        cA = inv_wavelet_2d(cA, dj, wavelet=wavelet)

    return cA