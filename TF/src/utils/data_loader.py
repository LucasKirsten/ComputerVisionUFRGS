import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
from skimage import io
from PIL import Image
import cv2
import numpy as np
import glob
import random


def transform(image, mask, res):
    # Resize
    resize = transforms.Resize(size=res)
    image = resize(image)
    mask = resize(mask)

    # # Random crop
    # i, j, h, w = transforms.RandomCrop.get_params(
    #     image, output_size=res)
    # image = tf.crop(image, i, j, h, w)
    # mask = tf.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = tf.hflip(image)
        mask = tf.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = tf.vflip(image)
        mask = tf.vflip(mask)

    # Transform to tensor
    image = tf.to_tensor(image)
    mask_np = np.array(mask).transpose(2, 0, 1)
    mask = torch.from_numpy(mask_np).long()

    return image, mask


class BacteriaDataset(Dataset):
    """Bacteria dataset."""

    def __init__(self, base_path, resolution=(256, 256)):
        self.images_list = sorted(glob.glob("/".join([base_path, "images"]) + "/*.png"))
        self.masks_list = sorted(glob.glob("/".join([base_path, "masks"]) + "/*.png"))
        self.resolution = resolution

        self.class_to_id = {"background": 0, "erythrocytes": 1, "spirochaete": 2}
        self.class_to_color = {"background": [0, 0, 0], "erythrocytes": [255, 0, 0], "spirochaete": [255, 255, 0]}
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}

        # self.inputs_dtype = torch.float32
        # self.targets_dtype = torch.long
        self.num_classes = len(self.class_to_id)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        ori_image = io.imread(self.images_list[idx])
        ori_mask = io.imread(self.masks_list[idx])

        image = Image.fromarray(np.uint8(ori_image)).convert('RGB')
        mask = Image.fromarray(np.uint8(ori_mask)).convert('RGB')
        image, mask = transform(image, mask, self.resolution)
        
        ori_image_res = cv2.resize(ori_image, (self.resolution[0], self.resolution[1]))
        ori_mask_res = cv2.resize(ori_mask, (self.resolution[0], self.resolution[1]))

        sample = {
            'image': image, 
            'mask': mask, 
            'original_img': ori_image_res,
            'original_mask': ori_mask_res
            }
        return sample
