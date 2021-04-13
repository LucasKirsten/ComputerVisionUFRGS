import matplotlib.pyplot as plt
import numpy as np


def plot_data(loader, limit_imgs):
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
        for key, val in loader.dataset.id_to_class.items():
            color_label[current_gt == key, :] = loader.dataset.class_to_color[val]
            # color_label[np.where((current_gt == key).all(axis=2))] = loader.dataset.class_to_color[val]

        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("Original Image", fontsize=50)
        ax1.imshow(curr_img)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("GT", fontsize=50)
        ax2.imshow(color_label)
        plt.show()


def plot_val_data(curr_img, curr_label, id_to_class, class_to_color):
    color_label = np.zeros((curr_img.shape[0], curr_img.shape[1], 3))
    for key, val in id_to_class.items():
        color_label[curr_label == key, :] = class_to_color[val]
    plt.figure()
    plt.imshow((curr_img / 255) * 0.5 + (color_label / 255) * 0.5)
    plt.show()
