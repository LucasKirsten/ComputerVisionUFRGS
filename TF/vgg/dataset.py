import glob
import numpy as np
import cv2
import preprocessing
import torch
from torch.utils.data import Dataset


class BacteriaDataset(Dataset):
    def __init__(self, base_path, is_train, resolution=[128, 128], max_side=None, augmentation=False, transform=None):
        self.images_list = sorted(glob.glob("/".join([base_path, "images"]) + "/*.png"))
        self.masks_list = sorted(glob.glob("/".join([base_path, "masks"]) + "/*.png"))

        self.is_train = is_train
        self.resolution = resolution
        self.max_side = max_side
        self.augmentation = augmentation
        self.transform = transform

        self.class_to_id = {"background": 0, "erythrocytes": 1, "spirochaete": 2}
        self.class_to_color = {"background": [0, 0, 0], "erythrocytes": [255, 0, 0], "spirochaete": [255, 255, 0]}
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        self.num_classes = len(self.class_to_id)

        # Mean and std are needed because we start from a pre trained net
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        # image = np.array(Image.open(self.images_list[idx]).convert("RGB"))
        # mask = np.array(Image.open(self.masks_list[idx]).convert("RGB"))
        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"].permute(2, 0, 1)
        # sample = {'image': image, 'mask': mask}

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
                img_np, label_np = preprocessing.random_rotation_image_max_size(img_np, label_np, 10, self.max_side)
            # Elastic distortion
            if np.random.rand() < 0.75:
                img_np, label_np = preprocessing.elastic_distortion_image(img_np, label_np)
            label_np = np.round(label_np).astype(np.int32)
        else:
            label_np = np.round(label_np).astype(np.int32)

        img_pt = img_np.astype(np.float32) / 255.0
        for i in range(3):
            img_pt[..., i] -= self.mean[i]
            img_pt[..., i] /= self.std[i]
        img_pt = img_pt.transpose(2, 0, 1)

        img_pt = torch.from_numpy(img_pt)
        label_pt = torch.from_numpy(label_np).long()
        sample = {'image': img_pt, 'gt': label_pt, 'image_original': img_np}

        if self.transform:
            sample = self.transform(sample)

        return sample
