import matplotlib.pyplot as plt
import numpy as np
from utils.converters import Converters


converters = Converters()
__ID_TO_CLASS = converters.get_id_to_class()
__CLASS_TO_COLOR = converters.get_class_to_color()


def plot_data(loader, limit_imgs, set_title=""):
    batch_dict = next(iter(loader))
    images_batch = batch_dict['image_original']
    gt_batch = batch_dict['gt']

    idx_imgs = 0
    for i in range(images_batch.shape[0]):
        if idx_imgs == limit_imgs:
            break
        idx_imgs += 1

        curr_img = images_batch[i, ...].cpu().numpy()
        current_gt = gt_batch[i, ...].cpu().numpy()

        color_label = np.zeros((curr_img.shape[0], curr_img.shape[1], 3))
        for key, val in __ID_TO_CLASS.items():
            color_label[current_gt == key, :] = __CLASS_TO_COLOR[val]

        fig = plt.figure(figsize=(20, 20))
        fig.suptitle(set_title, fontsize=50)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("Original Image", fontsize=50)
        ax1.imshow(curr_img)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("GT", fontsize=50)
        ax2.imshow(color_label)
        plt.show()


def get_board_image(curr_img, curr_label):
    color_label = np.zeros((curr_img.shape[0], curr_img.shape[1], 3))
    for key, val in __ID_TO_CLASS.items():
        color_label[curr_label == key, :] = __CLASS_TO_COLOR[val]
    return (curr_img / 255) * 0.5 + (color_label / 255) * 0.5
