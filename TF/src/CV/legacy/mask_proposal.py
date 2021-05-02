# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:34:39 2021

@author: kirstenl
"""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

from utils import get_feats_labels, features_fromclusters

def single_mask_proposal(image):
    # define kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    # augment contrast of objects in scene
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[...,0]
    blur = equalize_adapthist(lab, clip_limit=0.01)
    blur = np.uint8(blur*255)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # remove noises
    contour = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    segmented = np.copy(th)
    for cnt in contour:
        perimeter = cv2.arcLength(cnt,True)
        if perimeter>8:
            cv2.drawContours(segmented, [cnt], 0, (255,255,255), -1)
            continue
        
        # remove noise
        draw = np.ones_like(segmented)
        cv2.drawContours(draw, [cnt], 0, (0,0,0), -1)
        
        segmented = segmented * draw
    return np.uint8(segmented)

def get_mask_proposal(image):
    # define kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    # augment contrast of objects in scene
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[...,0]
    blur = equalize_adapthist(lab, clip_limit=0.01)
    blur = np.uint8(blur*255)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,0)
    
    # remove noises
    contour = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    segmented = np.copy(th)
    for cnt in contour:
        perimeter = cv2.arcLength(cnt,True)
        if perimeter>8:
            #cv2.drawContours(segmented, [cnt], 0, (255,255,255), -1)
            continue
        
        # remove noise
        draw = np.ones_like(segmented)
        cv2.drawContours(draw, [cnt], 0, (0,0,0), -1)
        
        segmented = segmented * draw
    mask = np.uint8(segmented)
        
    # split possible ery x spi classes
    kernel = np.ones((3,3),np.uint8)
    unknown = cv2.erode(mask, kernel, iterations=3)
    
    sure_ery = cv2.dilate(unknown, kernel, iterations=3)
    # fill contours on sure_ery
    contour = cv2.findContours(sure_ery, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    sure_ery = np.zeros_like(sure_ery)
    # for cnt in contour:
    #     cv2.drawContours(sure_ery, [cnt], 0, (255,255,255), -1)
    
    # unknown labels
    unknown = cv2.subtract(mask, sure_ery)
    
    # probably spi
    contour = cv2.findContours(unknown, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[1]
    spi = np.zeros_like(sure_ery)
    for cnt in contour:
            
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        if area<1 or perimeter<1:
            continue
        # if perimeter/area>2:
        #     continue
        
        _ = cv2.drawContours(spi, [cnt], 0, 255, -1)
    
    return sure_ery, spi

def get_watershed_labels(segmented):
    distance = cv2.distanceTransform(segmented, cv2.DIST_L2, 3)
    
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=segmented)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(distance, markers, mask=segmented)

    return labels

def watershed_mask(image):
    segmented = single_mask_proposal(image)
    labels = get_watershed_labels(segmented)
    
    return labels

def predict_masks(image, clf):
    y_pred  = list(get_mask_proposal(image))
    #plt.figure()
    #plt.imshow(np.hstack([y_pred[0], y_pred[1]]))
    
    feats,_,contours = get_feats_labels(image, y_pred[0], -1, return_contours=True)
    
    ery_pred = np.zeros_like(y_pred[0])
    spi_pred = np.zeros_like(y_pred[1])
    
    if len(feats)>0:
        class_pred = clf.predict(feats)
        
        for cnt, cp in zip(contours, class_pred):
            if cp==1:
                _ = cv2.drawContours(ery_pred, [cnt], 0, (255,255,255), -1)
            elif cp==2:
                _ = cv2.drawContours(spi_pred, [cnt], 0, (255,255,255), -1)
    
    feats,_,contours = get_feats_labels(image, y_pred[1], -1, return_contours=True)
    if len(feats)>0:
        class_pred = clf.predict(feats)
        for cnt, cp in zip(contours, class_pred):
            if cp==1:
                _ = cv2.drawContours(ery_pred, [cnt], 0, (255,255,255), -1)
            elif cp==2:
                _ = cv2.drawContours(spi_pred, [cnt], 0, (255,255,255), -1)
    
    return ery_pred, spi_pred

def predict_masks_watershed(image, clf):
    y_pred = watershed_mask(image)
    feats, contours = features_fromclusters(image, y_pred)
    
    ery_pred = np.zeros_like(y_pred)
    spi_pred = np.zeros_like(y_pred)
    
    if len(feats)>0:
        class_pred = clf.predict(feats)
        
        for cnt, cp in zip(contours, class_pred):
            if cp==1:
                _ = cv2.drawContours(ery_pred, [cnt], 0, (255,255,255), -1)
            elif cp==2:
                _ = cv2.drawContours(spi_pred, [cnt], 0, (255,255,255), -1)
    
    feats,_,contours = get_feats_labels(image, y_pred[1], -1, return_contours=True)
    if len(feats)>0:
        class_pred = clf.predict(feats)
        for cnt, cp in zip(contours, class_pred):
            if cp==1:
                _ = cv2.drawContours(ery_pred, [cnt], 0, (255,255,255), -1)
            elif cp==2:
                _ = cv2.drawContours(spi_pred, [cnt], 0, (255,255,255), -1)
    
    return ery_pred, spi_pred







