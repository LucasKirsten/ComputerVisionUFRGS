# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:15:09 2021

@author: kirstenl
"""

import numpy as np

def dice_score(y_true, y_pred, smooth=1):
    
    y_true = y_true.astype('float32')
    y_pred = y_pred.astype('float32')
    
    y_true = 1. - y_true/(np.max(y_true)+1e-3)
    y_pred = 1. - y_pred/(np.max(y_pred)+1e-3)
    
    intersection = np.sum(y_true*y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    
    dice = (2. * intersection + smooth)/(union + smooth)
    
    return dice

def binary_crossentropy(y_true, y_pred):
    
    y_true = y_true.astype('float32')
    y_pred = y_pred.astype('float32')
    
    y_true = y_true/(np.max(y_true)+1e-3)
    y_pred = y_pred/(np.max(y_pred)+1e-3)
    
    log = -y_true*np.log(y_pred+1e-3) - (1-y_true)*np.log(1-y_pred+1e-3)
    
    return log