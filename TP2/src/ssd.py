from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from skimage.color import rgb2gray
from skimage.io import imread

matplotlib.use('TkAgg')


def plot_images(left, right, disp, cmap=None):
    # Plot everything
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Left Image", fontsize=30)
    ax1.imshow(left, cmap=cmap)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Right Image", fontsize=30)
    ax2.imshow(right, cmap=cmap)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Disparity Image", fontsize=30)
    ax3.imshow(disp, cmap=cmap)
    plt.show()


@jit(nopython=False, parallel=True, cache=True)
def __compute__ssd_gray(left_image, right_image, disparities, window):
    """
    Compute a cost volume with maximum disparity steps considering a neighbourhood window with Sum of Squared Differences (SSD)

        @param left_image:  left input image of size (H,W)
        @param right_image: right input image of size (H,W)
        @param disparities:       maximum disparity
        @param window:      radius of the filter

        @return:            cost volume of size (H,W,steps)

    """
    assert (left_image.shape == right_image.shape)
    assert (len(left_image.shape) == 2)
    assert (disparities > 0)
    assert (window > 0)

    H, W = left_image.shape
    deth_map = np.zeros((H, W, disparities))

    # Loop over internal image
    for row in range(window, H - window):
        for col in range(window, W - window):
            for v in range(-window, window + 1):
                for u in range(-window, window + 1):
                    left = left_image[row + v, col + u]
                    # Loop over all possible disparities
                    for d in range(0, disparities):
                        right = right_image[row + v, col + u - d]
                        deth_map[row, col, d] += (int(left) - int(right)) ** 2
    return deth_map


@jit(nopython=False, parallel=True, cache=True)
def __compute__ssd_rgb(left_image, right_image, disparities, window):
    """
    Compute a cost volume with maximum disparity steps considering a neighbourhood window with Sum of Squared Differences (SSD)

        @param left_image:  left input image of size (H,W)
        @param right_image: right input image of size (H,W)
        @param disparities:       maximum disparity
        @param window:      radius of the filter

        @return:            cost volume of size (H,W,steps)

    """
    assert (left_image.shape == right_image.shape)
    assert (len(left_image.shape) == 3)
    assert (disparities > 0)
    assert (window > 0)

    H, W, C = left_image.shape
    disp_map = np.zeros((H, W, C, disparities))

    # Loop over internal image
    for row in range(window, H - window):
        for col in range(window, W - window):
            # Loop over channels
            for c in range(0, C):
                # Loop over window
                # v and u are the x,y of our local window search, used to ensure a good match
                # by the squared differences of the neighbouring pixels
                for v in range(-window, window + 1):
                    for u in range(-window, window + 1):
                        left = left_image[row + v, col + u, c]
                        # Loop over all possible disparities
                        for d in range(0, disparities):
                            right = right_image[row + v, col + u - d, c]
                            disp_map[row, col, c, d] += np.square(left - right)
    return disp_map


def get_min_disparity_ssd(l_img, r_img, d_steps, w_size):
    if len(l_img.shape) > 2:
        return np.mean(np.argmin(__compute__ssd_rgb(l_img, r_img, d_steps, w_size), axis=-1), axis=-1)
    else:
        return np.argmin(__compute__ssd_gray(l_img, r_img, d_steps, w_size), axis=-1)


if __name__ == "__main__":
    left_img_path = "../data/Teddy/teddy-png-2/im2.png"
    right_img_path = "../data/Teddy/teddy-png-2/im6.png"

    max_disp_steps = 60  # maximum disparity to consider
    window_size = 1  # size of the window to consider around the scan line point

    img_left = imread(left_img_path)
    img_right = imread(right_img_path)
    disp_map = get_min_disparity_ssd(l_img=img_left,
                                     r_img=img_right,
                                     d_steps=max_disp_steps,
                                     w_size=window_size)

    fig = plt.figure()
    plt.imshow(disp_map, cmap='gray')
    plt.show()
