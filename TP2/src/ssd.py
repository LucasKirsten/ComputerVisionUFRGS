import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from skimage.io import imread

matplotlib.use('TkAgg')


@jit(nopython=False, parallel=True, cache=True)
def __compute__ssd_gray(left_image, right_image, disparities, window):
    """
    Compute a cost volume with maximum disparity steps considering a
    neighbourhood window with Sum of Squared Differences (SSD)

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
    disp_map = np.zeros((H, W, disparities))

    # Loop over internal image
    for row in range(window, H - window):
        for col in range(window, W - window):
            for v in range(-window, window + 1):
                for u in range(-window, window + 1):
                    left = left_image[row + v, col + u]
                    # Loop over all possible disparities
                    for d in range(0, disparities):
                        right = right_image[row + v, col + u - d]
                        disp_map[row, col, d] += np.square(left - right)
    return disp_map


@jit(nopython=False, parallel=True, cache=True)
def __compute__ssd_gray_optim(left_image, right_image, disparities, window, apply_dist, penalty):
    """
    Compute a cost volume with maximum disparity steps considering a
    neighbourhood window with Sum of Squared Differences (SSD)

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
    if apply_dist:
        assert (penalty is not None)

    H, W = left_image.shape
    disp_map = np.zeros((H, W, disparities))

    # Loop over internal image
    for row in range(window, H - window):
        for col in range(window, W - window):
            for v in range(-window, window + 1):
                for u in range(-window, window + 1):
                    left = left_image[row + v, col + u]
                    # Loop over all possible disparities
                    for d in range(0, disparities):
                        right = right_image[row + v, col + u - d]
                        dist_metric = np.square(left - right) / (np.square(left - right) + penalty)
                        disp_map[row, col, d] += dist_metric
    return disp_map


@jit(nopython=False, parallel=True, cache=True)
def __compute__ssd_rgb(left_image, right_image, disparities, window):
    """
    Compute a cost volume with maximum disparity steps considering
    a neighbourhood window with Sum of Squared Differences (SSD)

        @param left_image:  left input image of size (H,W)
        @param right_image: right input image of size (H,W)
        @param disparities: maximum disparity
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


@jit(nopython=False, parallel=True, cache=True)
def __compute__ssd_rgb_optim(left_image, right_image, disparities, window, apply_dist, penalty):
    """
    Compute a cost volume with maximum disparity steps considering
    a neighbourhood window with Sum of Squared Differences (SSD)

        @param left_image:  left input image of size (H,W)
        @param right_image: right input image of size (H,W)
        @param disparities: maximum disparity
        @param window:      radius of the filter

        @return:            cost volume of size (H,W,steps)

    """
    assert (left_image.shape == right_image.shape)
    assert (len(left_image.shape) == 3)
    assert (disparities > 0)
    assert (window > 0)
    if apply_dist:
        assert (penalty is not None)

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
                            dist_metric = np.square(left - right) / (np.square(left - right) + penalty)
                            disp_map[row, col, c, d] += dist_metric
    return disp_map


def get_min_disparity_ssd(l_img, r_img, d_steps, w_size, apply_dist=False, penalty=None):
    if len(l_img.shape) > 2:
        if apply_dist:
            return np.mean(np.argmin(__compute__ssd_rgb_optim(left_image=l_img,
                                                              right_image=r_img,
                                                              disparities=d_steps,
                                                              window=w_size,
                                                              apply_dist=apply_dist,
                                                              penalty=penalty), axis=-1), axis=-1)
        else:
            return np.mean(np.argmin(__compute__ssd_rgb(left_image=l_img,
                                                        right_image=r_img,
                                                        disparities=d_steps,
                                                        window=w_size), axis=-1), axis=-1)
    else:
        if apply_dist:
            return np.argmin(__compute__ssd_gray_optim(left_image=l_img,
                                                       right_image=r_img,
                                                       disparities=d_steps,
                                                       window=w_size,
                                                       apply_dist=apply_dist,
                                                       penalty=penalty), axis=-1)
        else:
            return np.argmin(__compute__ssd_gray(left_image=l_img,
                                                 right_image=r_img,
                                                 disparities=d_steps,
                                                 window=w_size), axis=-1)


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
                                     w_size=window_size,
                                     apply_dist=False,
                                     penalty=None)

    disp_map_pen = get_min_disparity_ssd(l_img=img_left,
                                         r_img=img_right,
                                         d_steps=max_disp_steps,
                                         w_size=window_size,
                                         apply_dist=True,
                                         penalty=5000000000000000)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("SSD")
    ax1.imshow(disp_map, cmap='gray')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("SSD with Penalty")
    ax2.imshow(disp_map_pen, cmap='gray')
    plt.show()
