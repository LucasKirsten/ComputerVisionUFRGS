import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import preprocessing
from utils.converters import Converters


converters = Converters()


class BacteriaDataset(Dataset):
    def __init__(self, base_path, is_train, resolution=None, augmentation=False):
        if resolution is None:
            resolution = [128, 128]
        self.images_list = sorted(glob.glob("/".join([base_path, "images"]) + "/*.png"))
        self.masks_list = sorted(glob.glob("/".join([base_path, "masks"]) + "/*.png"))

        self.is_train = is_train
        self.resolution = resolution
        self.augmentation = augmentation

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        filename = self.images_list[idx].split('/')[-1].split('.')[0]
        img_np = cv2.imread(self.images_list[idx], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        label_np = cv2.imread(self.masks_list[idx], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)

        # ori_shape = img_np.shape
        img_np = cv2.resize(img_np, (self.resolution[0], self.resolution[1]))[..., ::-1]
        label_np = cv2.resize(label_np, (self.resolution[0], self.resolution[1]))
        img_np = np.ascontiguousarray(img_np)
        label_np = np.ascontiguousarray(label_np)

        if self.is_train and self.augmentation:
            # Added gaussian noise
            if np.random.rand() < 0.5:
                img_np = preprocessing.gaussian_noise_image(img_np, 0, np.random.randint(1, 10))
            # Random rotation
            if np.random.rand() < 0.75:
                img_np, label_np = preprocessing.random_rotation_image_max_size(img_np, label_np, 10)

            label_np = np.round(label_np).astype(np.int32)
        else:
            label_np = np.round(label_np).astype(np.int32)

        img_pt = img_np.astype(np.float32) / 255.0
        for i in range(3):
            img_pt[..., i] -= converters.get_mean()[i]
            img_pt[..., i] /= converters.get_std()[i]
        img_pt = img_pt.transpose(2, 0, 1)

        img_pt = torch.from_numpy(img_pt)
        label_pt = torch.from_numpy(label_np).long()
        sample = {'image': img_pt, 'gt': label_pt, 'image_original': img_np, 'filename': filename}

        return sample
