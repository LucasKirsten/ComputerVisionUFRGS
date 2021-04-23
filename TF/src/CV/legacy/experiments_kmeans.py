# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:29:07 2021

@author: kirstenl
"""

import os
import sys
sys.path.append(os.getcwd())

from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, segmentation, feature, future
from skimage.exposure import equalize_adapthist
from functools import partial

from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from metrics import *
from utils import *
from mask_proposal import *

folder_images = '../data/images/train'
folder_masks  = '../data/masks/train'

#%% define path to images

path_images = sorted(glob(os.path.join(folder_images, '*.png')))
path_masks  = sorted(glob(os.path.join(folder_masks, '*.png')))

#%%

i = np.random.randint(0, len(path_images))
pi = path_images[i]
pm = path_masks[i]

image = imread(pi)
mask  = get_mask(pm)

plt.figure()
plt.imshow(image)

#%%

lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
lab = equalize_adapthist(lab, clip_limit=0.01)

sigma_min = 1
sigma_max = 16
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=True, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        multichannel=True)
features = features_func(lab)

#%%

# Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
pixel_vals = features.reshape((-1,features.shape[-1])) 
# Convert to float type only for supporting cv2.kmean
pixel_vals = np.float32(pixel_vals)

#%%

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) #criteria
k = 5 # Choosing number of cluster
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

#%%

centers = np.uint8(centers) # convert data into 8-bit values 
segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)

#%%

for cluster in range(k):
    mask = np.where(labels == cluster, 255, 0).astype('uint8')
    plt.figure()
    plt.imshow(mask.reshape(*image.shape[:2]))

#%%

mask = np.where(labels == 1, 255, 0).astype('uint8')
plt.imshow(mask.reshape(*image.shape[:2]))

















