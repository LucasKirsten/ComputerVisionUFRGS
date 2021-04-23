# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:39:12 2021

@author: kirstenl
"""

import cv2
import numpy as np
from skimage import filters
from skimage.feature import hog, corner_harris, daisy
from skimage.exposure import equalize_adapthist

def hog_extractor(image):
    return hog(image, orientations=8, pixels_per_cell=(8,8),
               cells_per_block=(1,1), visualize=False, multichannel=True)

def lab_extractor(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab = equalize_adapthist(image)
    
    #image = np.concatenate([image, lab], axis=-1)
    
    return hog(lab, orientations=8, pixels_per_cell=(6,6),
               cells_per_block=(1,1), visualize=False, multichannel=True)

def daisy_extractor(image):
    r,g,b = cv2.split(image)
    fr = daisy(r, normalization='daisy', rings=3, histograms=3)
    fg = daisy(g, normalization='daisy', rings=3, histograms=3)
    fb = daisy(b, normalization='daisy', rings=3, histograms=3)
    
    return np.concatenate([fr, fg, fb]).reshape(-1)

def hist_extractor(image):
    
    # color histograms
    img_area = image.shape[0]*image.shape[1]
    hist0 = np.histogram(image[...,0].reshape(-1), bins=np.linspace(0,255,10))[0]/img_area
    hist1 = np.histogram(image[...,1].reshape(-1), bins=np.linspace(0,255,10))[0]/img_area
    hist2 = np.histogram(image[...,1].reshape(-1), bins=np.linspace(0,255,10))[0]/img_area
    
    # gradient histograms
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[...,0]
    grads = filters.sobel(lab)
    histg = np.histogram(grads.reshape(-1), bins=np.linspace(np.min(grads),np.max(grads),30))[0]/img_area
    
    hist = np.concatenate([hist0, hist1, hist2, histg])
    
    return hist