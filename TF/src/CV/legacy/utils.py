# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:33:49 2021

@author: kirstenl
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from tqdm import tqdm

from feature_extractor import *

def plot(x):
    plt.imshow(x)

def get_mask(path_mask, resize=(252,252)):
    
    mask = cv2.imread(path_mask)[...,0]
    
    ery = np.where(mask==1, 255, 0).astype('uint8')
    spi = np.where(mask==2, 255, 0).astype('uint8')
    
    ery = cv2.resize(ery, resize)
    spi = cv2.resize(spi, resize)
    
    return np.stack([ery, spi], axis=-1)

def show_ann(image, mask):
    
    plt.figure(figsize=(15,10))
    plt.subplot(121)
    plt.imshow(image)
    
    plt.subplot(122)
    plt.imshow(image)
    mask0 = cv2.merge([np.zeros_like(mask[...,0]), mask[...,0], np.zeros_like(mask[...,0])])
    mask1 = cv2.merge([mask[...,1], np.zeros_like(mask[...,1]), np.zeros_like(mask[...,1])])
    plt.imshow(mask0, cmap='jet', alpha=0.3)
    plt.imshow(mask1, cmap='jet_r', alpha=0.3)
    
def imread(path_img, resize=(252,252)):
    image = cv2.imread(path_img)[...,::-1]
    image = cv2.resize(image, resize)
    return image 

def get_feats_labels(image, mask, label, return_contours=False):
    
    contour = cv2.findContours(mask.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]
    
    features, labels, cnts_return = [],[],[]
    for cnt in contour:
        if len(cnt)<2:
            continue
        
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area<1 or perimeter<1:
            continue
        
        xmin = int(min(cnt[:,:,0]))
        xmax = int(max(cnt[:,:,0]))
        ymin = int(min(cnt[:,:,1]))
        ymax = int(max(cnt[:,:,1]))
        
        if xmax-xmin<32:
            aug = (xmax-xmin)//2
            xmin = max(xmin-aug, 0)
            xmax = min(xmax+aug, image.shape[1])
        if ymax-ymin<32:
            aug = (ymax-ymin)//2
            xmin = max(ymin-aug, 0)
            xmax = min(ymax+aug, image.shape[0])
        
        mask = cv2.drawContours(np.zeros_like(image), [cnt], 0, (1,1,1), -1)
        crop = image * mask
        crop = crop[ymin:ymax, xmin:xmax]
        crop = cv2.resize(crop, (32,32))
        
        feats = hog_extractor(crop)
        
        features.append(feats)
        labels.append(label)
        cnts_return.append(cnt)
    
    if return_contours:
        return features, labels, cnts_return
    else:
        return features, labels
    
def features_fromclusters(image, labels):
    features, contours = [],[]
    
    for lb in np.unique(labels):
        if lb==0:
            continue
        
        mask = np.where(labels==lb, 1, 0)
        
        # enclose cluster and get hog features
        contour = cv2.findContours(mask.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]
        if len(contour)<1:
            continue
        cnt = np.concatenate(contour, axis=0)
            
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area<1 or perimeter<1:
            continue
        
        xmin = int(min(cnt[:,:,0]))
        xmax = int(max(cnt[:,:,0]))
        ymin = int(min(cnt[:,:,1]))
        ymax = int(max(cnt[:,:,1]))
        
        if xmax-xmin<32:
            aug = (xmax-xmin)//2
            xmin = min(xmin-aug, 0)
            xmax = max(xmax+aug, image.shape[1])
        if ymax-ymin<32:
            aug = (ymax-ymin)//2
            xmin = min(ymin-aug, 0)
            xmax = max(ymax+aug, image.shape[0])
        
        crop = image[ymin:ymax, xmin:xmax]
        crop = cv2.resize(crop.astype('uint8'), (32,32))
        
        feats = hog_extractor(crop)
        
        features.append(feats)
        contours.append(cnt)

    return features, contours
    
def features_labels_fromclusters(image, mask_true, segments):
    features, labels = [],[]
    
    for cluster in range(np.max(segments)):
        mask = np.where(segments==cluster, 255, 0)
        
        # enclose cluster and get hog features
        contour = cv2.findContours(mask.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]
        if len(contour)<1:
            continue
        cnt = np.concatenate(contour, axis=0)
            
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area<1 or perimeter<1:
            continue
        
        xmin = int(min(cnt[:,:,0]))
        xmax = int(max(cnt[:,:,0]))
        ymin = int(min(cnt[:,:,1]))
        ymax = int(max(cnt[:,:,1]))
        
        crop = image[ymin:ymax, xmin:xmax]
        crop = cv2.resize(crop, (12,12))
        
        feats = hog(crop, orientations=8, pixels_per_cell=(6,6),
                    cells_per_block=(1,1), visualize=False, multichannel=True)
        
        label = mask_true[ymin:ymax, xmin:xmax]
        label = np.bincount(label.reshape(-1)).argmax()
        
        features.append(feats)
        labels.append(label)

    return features, labels

class DataLoader():
    def __init__(self, path_root):
        
        train_images = open(os.path.join(path_root,'train/train_images.txt'), 'r').read().split('\n')
        val_images = open(os.path.join(path_root,'val/val_images.txt'), 'r').read().split('\n')
        test_images = open(os.path.join(path_root,'test/test_images.txt'), 'r').read().split('\n')

        train_images = np.concatenate([train_images, val_images], axis=-1)
        
        train_images = [os.path.join(path_root,p) for p in train_images]
        test_images = [os.path.join(path_root,p) for p in test_images]
        
        print('Loading train data...')
        self.train_data = []
        for path_img in tqdm(train_images):
            image = imread(path_img)
            masks = get_mask(path_img.replace('images','masks'))
            
            self.train_data.append([image, masks])
        
        print('Loading test data...')
        self.test_data = []
        for path_img in tqdm(test_images):
            image = imread(path_img)
            masks = get_mask(path_img.replace('images','masks'))
            
            self.test_data.append([image, masks])
            
    def set_data(self, data):
        if data=='train':
            self.data = self.train_data
        else:
            self.data = self.test_data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]