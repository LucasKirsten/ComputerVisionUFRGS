''' Data loader script '''

import os
import cv2
import numpy as np
from tqdm import tqdm

def imread(path_img, resize=(252,252)):
    '''
    Read images in RGB format and resizing it.
    '''
    image = cv2.imread(path_img)[...,::-1]
    image = cv2.resize(image, resize)
    return image

def get_mask(path_mask, resize=(252,252)):
    '''
    Get ground truth masks.
    '''
    
    mask = cv2.imread(path_mask)[...,0]
    
    ery = np.where(mask==1, 255, 0).astype('uint8')
    spi = np.where(mask==2, 255, 0).astype('uint8')
    
    ery = cv2.resize(ery, resize)
    spi = cv2.resize(spi, resize)
    
    return np.stack([ery, spi], axis=-1)

class DataLoader():
    def __init__(self, path_root):
        """
        Data loader for handling images and masks from dataset.

        Parameters
        ----------
        path_root : string
            Root folder where to find the data.

        """
        
        # get annotations
        train_images = open(os.path.join(path_root,'train/train_images.txt'), 'r').read().split('\n')
        val_images = open(os.path.join(path_root,'val/val_images.txt'), 'r').read().split('\n')
        test_images = open(os.path.join(path_root,'test/test_images.txt'), 'r').read().split('\n')
        
        # join training and validation
        train_images = np.concatenate([train_images, val_images], axis=-1)
        
        # defines full path
        train_images = [os.path.join(path_root,p) for p in train_images]
        test_images = [os.path.join(path_root,p) for p in test_images]
        
        # read all training images
        print('Loading train data...')
        self.train_data = []
        for path_img in tqdm(train_images):
            image = imread(path_img)
            masks = get_mask(path_img.replace('images','masks'))
            
            self.train_data.append([image, masks])
        
        # read all test images
        print('Loading test data...')
        self.test_data = []
        for path_img in tqdm(test_images):
            image = imread(path_img)
            masks = get_mask(path_img.replace('images','masks'))
            
            self.test_data.append([image, masks])
            
    def set_data(self, data):
        """
        Apply soft thresholding in 2D input image using Wavelet Transform.

        Parameters
        ----------
        data : string
            Data to be used. Either train or test.

        """
        if data=='train':
            self.data = self.train_data
        else:
            self.data = self.test_data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]