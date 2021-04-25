# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:18:49 2021

@author: kirstenl
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
from skimage.io import imread, imsave

import multiprocessing
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()
from sklearn.feature_extraction.image import extract_patches_2d
import tensorflow as tf

#%%

max_disp_steps = 50  # maximum disparity to consider
window_size = 11  # size of the window to consider around the scan line point

teddy_images_dir = "../data/saved_images/teddy"
cones_images_dir = "../data/saved_images/cones"
pl.Path(teddy_images_dir).mkdir(exist_ok=True, parents=True)
pl.Path(cones_images_dir).mkdir(exist_ok=True, parents=True)

left_img_teddy_path = "../../data/teddy/im2.png"
right_img_teddy_path = "../../data/teddy/im6.png"

img_teddy_left = imread(left_img_teddy_path)
img_teddy_right = imread(right_img_teddy_path)

left_img_cones_path = "../../data/cones/im2.png"
right_img_cones_path = "../../data/cones/im6.png"

img_cones_left = imread(left_img_cones_path)
img_cones_right = imread(right_img_cones_path)

#%%
from time import time

def _ssd(disparity, patches_left, patches_right, apply_dist, penalty, shape):
    penalty = tf.cast(penalty**2, 'float32')
    patches_right = np.roll(patches_right, shift=disparity, axis=1)
    patches_right[:,:disparity] = 0
    
    diff_square = tf.square(patches_left-patches_right)
    if apply_dist:
        dist_metric = diff_square/(diff_square+penalty)
    else:
        dist_metric = diff_square
    costs = tf.reduce_sum(dist_metric, axis=(2,3)).numpy()
    return costs.reshape(shape)

def compute_ssd(left_image, right_image, disparities, window_size,
                apply_dist=False, penalty=20):
    
    assert (left_image.shape == right_image.shape)
    assert (len(left_image.shape)>2)
    assert (disparities > 0)
    assert (window_size > 0 and window_size%2!=0)
    if apply_dist:
        assert penalty>0
    
    pad = window_size//2
    
    padded = cv2.copyMakeBorder(left_image, pad,pad,pad,pad, cv2.BORDER_CONSTANT, None, 0)
    patches_left = extract_patches_2d(padded, (window_size, window_size)).astype('float32')
    patches_left = np.reshape(patches_left, (*left_image.shape[:2], *patches_left.shape[1:]))
    
    padded = cv2.copyMakeBorder(right_image, pad,pad,pad,pad, cv2.BORDER_CONSTANT, None, 0)
    patches_right = extract_patches_2d(padded, (window_size, window_size)).astype('float32')
    patches_right = np.reshape(patches_right, (*right_image.shape[:2], *patches_right.shape[1:]))
    
    params = dict(
        patches_left=patches_left,
        patches_right=patches_right,
        apply_dist=apply_dist,
        penalty=penalty,
        shape=img_teddy_left.shape
    )
    with Parallel(n_jobs=NUM_CORES, prefer="threads") as parallel:
        costs = parallel(delayed(_ssd)(disparity=d, **params) \
                      for d in range(disparities))

    return np.stack(costs,axis=-1)

#%%
from time import time

init = time()
costs0 = compute_ssd(left_image=img_teddy_left,
                    right_image=img_teddy_right,
                    disparities=max_disp_steps,
                    window_size=1,
                    apply_dist=True, penalty=100)
print(time()-init)

mincost = np.mean(np.argmin(costs0, axis=-1), axis=-1)
plt.imshow(mincost)

#%% Aggregation

def __compute_aggregation(costs0, window_size, mode):
    
    H,W,C,D = costs0.shape
    
    costs = np.zeros_like(costs0)
    
    def __compute_y(y):
        for x in range(W):
            if mode=='mean':
                h = y-window_size
                w = x-window_size
                
                costh = 0 if h<0 else costs0[h,x]
                costw = 0 if w<0 else costs0[y,w]
                costhw = 0 if w<0 and h<0 else costs0[h,w]
                
                costs[y,x] = (costs0[y,x] - costh - costw + costhw)/window_size**2.
            
            else:
                miny = max(0, y-window_size//2)
                maxy = min(H, y+window_size//2)
                minx = max(0, x-window_size//2)
                maxx = min(W, x+window_size//2)
                
                cost = costs0[miny:maxy, minx:maxx]
                costs[y,x] = np.median(cost, axis=(0,1))
    
    with Parallel(n_jobs=NUM_CORES, prefer="threads") as parallel:
        parallel(delayed(__compute_y)(y) for y in range(H))
    
    return costs

def compute_aggregation(costs0, window_size, mode='mean'):
    
    assert mode in ('mean', 'median'), 'Invalid mode to compute costs with aggregation!'
    assert window_size%2!=0, 'Window size should be an odd value!'
    
    if mode=='mean':
        integral = np.stack([cv2.integral(costs0[...,i])[1:,1:] \
                             for i in range(costs0.shape[-1])], axis=-1)
        costs0 = np.copy(integral)
    costs0 = np.float32(costs0)
    
    return __compute_aggregation(costs0, window_size, mode)

#%%
init = time()
agg = compute_aggregation(costs0, 11, 'median')
print('Aggregation time: ', time()-init)

mincost = np.mean(np.argmin(agg, axis=-1), axis=-1)
plt.imshow(mincost)


