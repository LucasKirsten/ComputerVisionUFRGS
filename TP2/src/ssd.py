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


@jit(nopython=True, parallel=True, cache=True)
def __compute_costvolume_ssd(left_image, right_image, steps, window):
    """
    Compute a cost volume with maximum disparity steps considering a neighbourhood window with Sum of Squared Differences (SSD)

        @param left_image:  left input image of size (H,W)
        @param right_image: right input image of size (H,W)
        @param steps:       maximum disparity
        @param window:      radius of the filter

        @return:            cost volume of size (H,W,steps)

    """
    assert (left_image.shape == right_image.shape)
    assert (steps > 0)
    assert (window > 0)

    (H, W) = left_image.shape
    cv = np.zeros((H, W, steps))

    # Loop over internal image
    for y in range(window, H - window):
        for x in range(window, W - window):
            # Loop over window
            # v and u are the x,y of our local window search, used to ensure a good match
            # by the squared differences of the neighbouring pixels
            for v in range(-window, window + 1):
                for u in range(-window, window + 1):
                    # Loop over all possible disparities
                    for d in range(0, steps):
                        cv[y, x, d] += (left_image[y + v, x + u] - right_image[y + v, x + u - d]) ** 2
    return cv


def get_min_disparity_ssd(l_img, r_img, d_steps, w_size):
    return np.argmin(__compute_costvolume_ssd(l_img, r_img, d_steps, w_size), axis=-1)


if __name__ == "__main__":
    left_img_path = "../data/Teddy/teddy-png-2/im2.png"
    right_img_path = "../data/Teddy/teddy-png-2/im6.png"

    img_left = rgb2gray(imread(left_img_path))
    img_right = rgb2gray(imread(right_img_path))
    max_disp_steps = 60  # maximum disparity to consider
    window_size = 11  # size of the window to consider around the scan line point
    disp_res = get_min_disparity_ssd(l_img=img_left,
                                     r_img=img_right,
                                     d_steps=max_disp_steps,
                                     w_size=window_size)

    # plot_images(img_left, img_right, disp_res, 'gray')
    plt.figure()
    plt.imshow(disp_res, cmap='gray')
    plt.show()
