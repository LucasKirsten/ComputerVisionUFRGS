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
import tqdm
from utils import plot_utils, block_matching

#%%

max_disp_steps = 50  # maximum disparity to consider
window_size = 2  # size of the window to consider around the scan line point

teddy_images_dir = "../data/saved_images/teddy"
cones_images_dir = "../data/saved_images/cones"
pl.Path(teddy_images_dir).mkdir(exist_ok=True, parents=True)
pl.Path(cones_images_dir).mkdir(exist_ok=True, parents=True)

left_img_teddy_path = "./images/teddy/im2.png"
right_img_teddy_path = "./images/teddy/im6.png"

img_teddy_left = imread(left_img_teddy_path)
img_teddy_right = imread(right_img_teddy_path)

left_img_cones_path = "./images/cones/im2.png"
right_img_cones_path = "./images/cones/im6.png"

img_cones_left = imread(left_img_cones_path)
img_cones_right = imread(right_img_cones_path)

#%%

costs0 = block_matching.__compute__ssd_rgb(left_image=img_teddy_left,
                                            right_image=img_teddy_right,
                                            disparities=max_disp_steps,
                                            window=window_size)

#%%

integral = np.stack([cv2.integral(costs0[...,i])[1:,1:] \
                     for i in range(costs0.shape[-1])], axis=-1)

#%% mean

M = 3
costs = np.zeros(integral.shape[:-1])

for y in range(costs.shape[0]):
    for x in range(costs.shape[1]):
        for c in range(costs.shape[2]):
            h = max(0, y-M)
            w = max(0, x-M)
            cost = (integral[y,x,c] - integral[h, x, c] - integral[y, w, c] + integral[h, w, c])/M**2
            costs[y,x,c] = np.min(cost)

#%% median

M = 3

assert M%2!=0, 'M should be odd!'
M = M//2
costs = np.zeros(costs0.shape[:-1])
h,w = costs.shape[:2]

for y in range(costs.shape[0]):
    for x in range(costs.shape[1]):
        for c in range(costs.shape[2]):
            
            miny = max(0, y-M)
            maxy = min(h, y+M)
            minx = max(0, x-M)
            maxx = min(w, x+M)
            
            cost = costs0[miny:maxy, minx:maxx, c]
            costs[y,x,c] = np.median(cost)




