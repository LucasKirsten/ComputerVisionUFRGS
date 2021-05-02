# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:13:56 2021

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

from sklearn.metrics import classification_report, precision_score, recall_score, jaccard_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

import multiprocessing
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()

from metrics import *
from utils import *
from mask_proposal import *

#%% define path to images TRAIN

folder_images = '../data/images/train'
folder_masks  = '../data/masks/train'

path_images = sorted(glob(os.path.join(folder_images, '*.png')))
path_masks  = sorted(glob(os.path.join(folder_masks, '*.png')))

#%%

def get_features_labels(dataset):
    features, labels = [],[]
    def _get_features_labels(pi, pm):

        image = imread(pi)
        mask  = get_mask(pm)

        y_true = list(cv2.split(mask))
        y_pred = list(get_mask_proposal(image))

        f,l = get_feats_labels(image, y_true[0], 1)
        features.extend(f); labels.extend(l)
        f,l = get_feats_labels(image, y_true[1], 2)
        features.extend(f); labels.extend(l)

        # background
        background = y_pred[1].astype('float') - y_true[1].astype('float')
        background[background<0] = 0
        f,l = get_feats_labels(image, background.astype('uint8'), 0)
        features.extend(f); labels.extend(l)

    with Parallel(n_jobs=NUM_CORES, prefer="threads") as parallel:
        _ = parallel(delayed(_get_features_labels)(dataset[i]) for i in tqdm(range(dataset)))
        
    return np.array(features), np.array(labels)
    
#%%

# kf = StratifiedKFold(n_splits=5)

# for train_index, test_index in kf.split(features, labels):
#     X_train, X_test = features[train_index], features[test_index]
#     y_train, y_test = labels[train_index], labels[test_index]

#     clf = make_pipeline(StandardScaler(), SVC(C=30, gamma='auto', class_weight='balanced'))
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)

#     print(classification_report(y_test, y_pred))
    
#     del clf

#%%

clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='rbf', gamma='auto',\
                                          class_weight=None, decision_function_shape='ovr',\
                                          verbose=True))
clf.fit(features, labels)

#%% evaluate on metrics

# define path to images TEST
folder_images = '../data/images/test'
folder_masks  = '../data/masks/test'

path_images = sorted(glob(os.path.join(folder_images, '*.png')))
path_masks  = sorted(glob(os.path.join(folder_masks, '*.png')))

all_true, all_pred = [],[]

def make_predicts(pi, pm):
    
    image = imread(pi)
    mask  = get_mask(pm)
    
    mask_true = list(cv2.split(mask))
    mask_pred = list(predict_masks(image, clf))
    
    y_true = np.where(mask_true[0]==255, 1, 0)
    y_true = np.where(mask_true[1]==255, 2, y_true)
    
    y_pred = np.where(mask_pred[0]==255, 1, 0)
    y_pred = np.where(mask_pred[1]==255, 2, y_pred)
    
    all_true.append(y_true)
    all_pred.append(y_pred)
    
with Parallel(n_jobs=NUM_CORES, prefer="threads") as parallel:
    _ = parallel(delayed(make_predicts)(pi,pm) \
                      for pi, pm in tqdm(zip(path_images, path_masks), total=len(path_images)))

y_true = np.array(all_true).reshape(-1,1)
y_pred = np.array(all_pred).reshape(-1,1)

print(classification_report(y_true, y_pred))
print('Jaccard (macro avg): ', jaccard_score(y_true, y_pred, average='macro'))
print('Jaccard (weighted avg): ', jaccard_score(y_true, y_pred, average='weighted'))

#%%

i = np.random.randint(0, len(path_images))
pi = path_images[i]
pm = path_masks[i]

image = imread(pi)
mask  = get_mask(pm)

#%%
mask_true = list(cv2.split(mask))
proposal = single_mask_proposal(image)
mask_pred = list(predict_masks(image, clf))

ery_true = np.where(mask_true[0]==255, 255, 0)
spi_true = np.where(mask_true[1]==255, 255, 0)
y_true = cv2.merge([ery_true, spi_true, np.zeros_like(ery_true)])

ery_pred = np.where(mask_pred[0]==255, 255, 0)
spi_pred = np.where(mask_pred[1]==255, 255, 0)
y_pred = cv2.merge([ery_pred, spi_pred, np.zeros_like(ery_pred)])

plt.figure()
plt.imshow(np.hstack([image, y_true, y_pred, cv2.merge([proposal]*3)]))















