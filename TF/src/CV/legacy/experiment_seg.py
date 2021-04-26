# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:49:22 2021

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
from skimage.exposure import equalize_adapthist
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries

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
mask_true = imread(pm)

#%%

lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
lab = equalize_adapthist(lab, clip_limit=0.01)

#%%

img = np.copy(image.astype('double'))

segments_fz = felzenszwalb(lab, scale=100, sigma=0.8, min_size=10, multichannel=True)
segments_slic = slic(img, n_segments=100, compactness=10, sigma=1,
                     start_label=1)

#%%

fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)

ax[0].imshow(mark_boundaries(image, segments_fz))
ax[0].set_title("Felzenszwalbs's method")
ax[1].imshow(mark_boundaries(image, segments_slic))
ax[1].set_title('SLIC')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()


#%%

features, labels = [],[]
for pi, pm in tqdm(zip(path_images, path_masks), total=len(path_images)):
    
    image = imread(pi)
    mask_true = imread(pm)
    
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab = equalize_adapthist(lab, clip_limit=0.01)
    
    segments = felzenszwalb(lab, scale=100, sigma=0.8, min_size=10, multichannel=True)
    
    ft,lb = features_labels_fromclusters(image, mask_true, np.copy(segments))
    
    features.extend(ft)
    labels.extend(lb)
    
features = np.array(features)
labels = np.array(labels)

#%%

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

#%%
    
#%%

kf = StratifiedKFold(n_splits=5)

for train_index, test_index in kf.split(features, labels):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    clf = make_pipeline(StandardScaler(), SVC(C=30, gamma='auto', class_weight='balanced'))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    
    del clf
    
    
    
    
    
    
    
    