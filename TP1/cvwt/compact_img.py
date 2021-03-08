# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:31:12 2021

@author: kirstenl
"""

import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from .wavelet_2d import wavelet_2d, inv_wavelet_2d

@njit
def _remove_low_values(x, alpha):
    
    '''
    Remove low values for array.
    '''
    
    # TODO: maybe improve speed
    
    # flatten x and store shape
    shape = x.shape
    x = x.reshape(-1)
    
    # iterate over sorted values and indexes of |x| to eliminate low values
    for i,xx in zip(np.argsort(np.abs(x)),np.sort(np.abs(x))):
        # verify if the ratio is as expected
        if (np.count_nonzero(x)/len(x))<=alpha:
            break
        
        # if not already zero, set to zero
        if xx!=0:
            x[i] = 0
    
    # reshape to original shape and return
    return x.reshape(*shape)

def compact_image(x, alpha, wavelet='haar', J=1, use_pywavelet=False):
    
    '''
    Compact 2D input image using Wavelet Transform.
    
    Parameters
    ----------
    x : np.array
        Input signal.
    alpha : float ranging from (0, 1]
        Percentage of low gradients to remove from detail coefficients.
    wavelet : str or list<np.array>, optional
        Wavelet filter to be used. The default is haar.
    J : int, optional
        Maximum decomposition level. The default is 1.
    use_pywavelet : bool, optional
        If to use PyWavelet python package. The default is False.

    Returns
    -------
    [cA] : list<np.array>
        Intermidiate smooth signals.
    Dj : list<np.array>
        Detail coefficients.
    cA : np.array
        Final smooth signal.
    
    '''
    
    assert J>0, 'J should be greater than 0!'
    assert 0<alpha<=1, 'alpha should be between [0,1]!'
    
    # going forward in the wavelet transform
    Dj = []
    cA = np.copy(x)
    for j in range(J):
        if use_pywavelet:
            cA,cD = pywt.dwt2(cA, wavelet=wavelet)
        else:
            cA,cD = wavelet_2d(cA, wavelet=wavelet)
        
        # remove low values
        Dj.append([_remove_low_values(d,alpha) for d in cD])
    
    # store values to be returned
    returns = (cA, Dj)
        
    # returning to the filtered image
    for j, dj in enumerate(reversed(Dj)):
        if use_pywavelet:
            cA = pywt.idwt2((cA, dj), wavelet=wavelet)
        else:
            cA = inv_wavelet_2d(cA, dj, wavelet=wavelet)
    
    return returns + (cA,)
    
if __name__=='__main__':
    # number of iterations
    J = 3
    
    # parameter for compacting
    alpha = 0.1
    
    # define input image
    x = cv2.imread('./barbara.jpg')
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    
    # copmact image
    (cA, Dj, reconstructed) = compact_image(x, J=J, alpha=alpha)
    
    fig, axs = plt.subplots(1,2)
    fig.suptitle(f'Own: Compacted image J={J}')
    axs[0].imshow(x)
    axs[0].set_title('Original image')
    
    axs[1].imshow(reconstructed)
    axs[1].set_title('Reconstructed image')
    
    # compact images with PyWavelet
    (cA, Dj, reconstructed) = compact_image(x, J=J, alpha=alpha, use_pywavelet=True)
    
    fig, axs = plt.subplots(1,2)
    fig.suptitle(f'PyWavelet: Compacted image J={J}')
    axs[0].imshow(x)
    axs[0].set_title('Original image')
    
    axs[1].imshow(reconstructed)
    axs[1].set_title('Reconstructed image')
    
    
    