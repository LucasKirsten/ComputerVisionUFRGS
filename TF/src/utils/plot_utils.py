import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals._pilutil import bytescale


def re_normalize(inp: np.ndarray,
                 low: int = 0,
                 high: int = 255
                 ):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out


def plot_data(batch_dict, dataset):
    images_batch = batch_dict['image']
    masks_batch = batch_dict['mask']

    for i in range(images_batch.shape[0]):
        curr_img = images_batch[i, ...].numpy().transpose(1, 2, 0)
        curr_msk = masks_batch[i, ...].numpy().transpose(1, 2, 0)

        mask_cp = curr_msk.copy()
        mask_cp = re_normalize(mask_cp)
        for label, color in dataset.class_to_color.items():
            mask_cp[np.any(curr_msk == dataset.class_to_id[label], axis=-1)] = color

        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(curr_img)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(mask_cp)
        plt.show()
