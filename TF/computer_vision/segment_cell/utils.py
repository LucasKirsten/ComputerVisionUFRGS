""" Utils functions for plotting """

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot(x, *args, **kwargs):
    plt.figure(*args, **kwargs)
    plt.imshow(x)

def show_ann(image, mask):

    '''
    Display annotations for a given image.
    '''
    
    plt.figure(figsize=(15,10))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.merge([mask[...,0], mask[...,1], np.zeros_like(mask[...,0])]))
    plt.title('Segmentation mask')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(image)
    mask0 = cv2.merge([np.zeros_like(mask[...,0]), mask[...,0], np.zeros_like(mask[...,0])])
    mask1 = cv2.merge([mask[...,1], np.zeros_like(mask[...,1]), np.zeros_like(mask[...,1])])
    plt.imshow(mask0, cmap='jet', alpha=0.3)
    plt.imshow(mask1, cmap='jet_r', alpha=0.3)
    plt.title('Mask applied on the image')
    plt.axis('off')