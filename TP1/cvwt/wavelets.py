from .wavelet_1d import wavelet_1d, inv_wavelet_1d
from .wavelet_2d import wavelet_2d, inv_wavelet_2d
from .utils import _remove_borders, _get_filters

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
        Approximation signal.
    Dj : list<numpy.ndarray>
        Detail coefficients.

    """
    assert J > 0, 'J should be greater than 0!'

    # get filters if given individually
    if c and d:
        wavelet = [c, d]

    # go forward in the wavelet transform
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
    
    # get filters if given individually
    if f and g:
        wavelet = [f, g]
    
    # go backwards in the wavelet transform
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
        Approximation signal.
    Dj : list<numpy.ndarray>
        Detail coefficients.

    """
    assert J > 0, 'J should be greater than 0!'
    
    # get filters if given individually
    if c and d:
        wavelet = [c, d]
    
    # verify if the image has color channels
    if len(x.shape)>2:
        channels = x.shape[-1]
    else:
        channels = 1
        x = x[..., np.newaxis]
        
    def _dwt(x):
        # function to go fowards in the wavelet transform
        Dj = []
        cA = np.copy(x)
        for j in range(J):
            cA, cD = wavelet_2d(cA, wavelet=wavelet)
            Dj.append(cD)
        return cA, Dj
    
    # apply wavelet transform for each image channel
    cA, Dj = [],[]
    for ch in range(channels):
        xx = x[..., ch]
        cA_ch, cD_ch = _dwt(xx)
        cA.append(cA_ch)
        Dj.append(cD_ch)
    
    # merge the channels
    cA = np.stack(cA, axis=-1)
    cD = []
    for j in range(J):
        dj = []
        for ch in range(channels):
            dj.append(Dj[ch][j])
        cD.append(np.stack(dj, axis=-1))

    return cA, cD


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
    
    # get filters if given individually
    if f and g:
        wavelet = [f, g]
        
    # verify if the image has color channels
    if len(cA.shape)>2:
        channels = cA.shape[-1]
    else:
        channels = 1
        cA = cA[..., np.newaxis]
    
    # go backwards in the wavelet transform
    inv = []
    for ch in range(channels):
        aa = cA[..., ch]
        for j, dj in enumerate(reversed(cD)):
            aa = inv_wavelet_2d(aa, dj[...,ch], wavelet=wavelet)
        inv.append(aa)
        
    inv = np.stack(inv, axis=-1)
    
    # remove last channel if single color
    if inv.shape[-1]==1:
        inv = inv[...,0]
    return inv