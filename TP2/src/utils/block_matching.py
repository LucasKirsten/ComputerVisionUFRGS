import cv2
import numpy as np
from numba import jit


def get_min_disparity_ssd(l_img, r_img, d_steps, w_size, apply_dist=False, penalty=None):
    """
    Main method that calls the corresponding block matching algorithm to be used

    Parameters
    ----------
    l_img : numpy.ndarray
        left input image of size (H, W) or (H, W, C)
    r_img : numpy.ndarray
        right input image of size (H, W) or (H, W, C)
    d_steps: int
        maximum disparity number
    w_size: int
        radius of the filter (it will be a square window (window x window))
    apply_dist : bool
        flag used to select whether the optimized block matching algorithm or the standard one
    penalty : int or float
        numeric value used as a penalty to the distance metric

    Returns
    -------
    disp_map : numpy.ndarray
        the best depth map of size (H, W) based on the computed disparities

    """
    if len(l_img.shape) > 2:
        if apply_dist:
            return np.mean(np.argmin(__compute_ssd_rgb_optim(left_image=l_img,
                                                             right_image=r_img,
                                                             disparities=d_steps,
                                                             window=w_size,
                                                             penalty=penalty), axis=-1), axis=-1)
        else:
            return np.mean(np.argmin(__compute_ssd_rgb(left_image=l_img,
                                                       right_image=r_img,
                                                       disparities=d_steps,
                                                       window=w_size), axis=-1), axis=-1)
    else:
        if apply_dist:
            return np.argmin(__compute_ssd_gray_optim(left_image=l_img,
                                                      right_image=r_img,
                                                      disparities=d_steps,
                                                      window=w_size,
                                                      penalty=penalty), axis=-1)
        else:
            return np.argmin(__compute_ssd_gray(left_image=l_img,
                                                right_image=r_img,
                                                disparities=d_steps,
                                                window=w_size), axis=-1)


@jit(nopython=False, parallel=True, cache=True)
def __compute_ssd_gray(left_image, right_image, disparities, window):
    """
    Compute a cost volume with maximum disparity steps considering a
    neighbourhood window with Sum of Squared Differences (SSD)

    Parameters
    ----------
    left_image : numpy.ndarray
        left input image of size (H, W)
    right_image : numpy.ndarray
        right input image of size (H, W)
    disparities: int
        maximum disparity number
    window: int
        radius of the filter (it will be a square window (window x window))

    Returns
    -------
    disp_map : numpy.ndarray
        cost volume of size (H, W, disparities)

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
def __compute_ssd_gray_optim(left_image, right_image, disparities, window, penalty):
    """
    Compute a cost volume with maximum disparity steps considering a
    neighbourhood window with a robust distance metric together with the
    Sum of Squared Differences (SSD) calculation

    Parameters
    ----------
    left_image : numpy.ndarray
        left input image of size (H, W)
    right_image : numpy.ndarray
        right input image of size (H, W)
    disparities: int
        maximum disparity number
    window: int
        radius of the filter (it will be a square window (window x window))
    penalty : int or float
        numeric value used as a penalty to the distance metric

    Returns
    -------
    disp_map : numpy.ndarray
        cost volume of size (H, W, disparities)

    """
    assert (left_image.shape == right_image.shape)
    assert (len(left_image.shape) == 2)
    assert (disparities > 0)
    assert (window > 0)
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
                        dist_metric = np.square(left - right) / (np.square(left - right) + np.square(penalty))
                        disp_map[row, col, d] += dist_metric
    return disp_map


@jit(nopython=False, parallel=True, cache=True)
def __compute_ssd_rgb(left_image, right_image, disparities, window):
    """
    Compute a cost volume with maximum disparity steps considering a
    neighbourhood window with Sum of Squared Differences (SSD)

    Parameters
    ----------
    left_image : numpy.ndarray
        left input image of size (H, W, C)
    right_image : numpy.ndarray
        right input image of size (H, W, C)
    disparities: int
        maximum disparity number
    window: int
        radius of the filter (it will be a square window (window x window))

    Returns
    -------
    disp_map : numpy.ndarray
        cost volume of size (H, W, disparities, C)

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
def __compute_ssd_rgb_optim(left_image, right_image, disparities, window, penalty):
    """
    Compute a cost volume with maximum disparity steps considering a
    neighbourhood window with a robust distance metric together with the
    Sum of Squared Differences (SSD) calculation

    Parameters
    ----------
    left_image : numpy.ndarray
        left input image of size (H, W, C)
    right_image : numpy.ndarray
        right input image of size (H, W, C)
    disparities: int
        maximum disparity number
    window: int
        radius of the filter (it will be a square window (window x window))
    penalty : int or float
        numeric value used as a penalty to the distance metric

    Returns
    -------
    disp_map : numpy.ndarray
        cost volume of size (H, W, disparities, C)

    """
    assert (left_image.shape == right_image.shape)
    assert (len(left_image.shape) == 3)
    assert (disparities > 0)
    assert (window > 0)
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
                            dist_metric = np.square(left - right) / (np.square(left - right) + np.square(penalty))
                            disp_map[row, col, c, d] += dist_metric
    return disp_map


@jit(nopython=False, parallel=True, cache=True)
def __compute_ssd_rgb_agg(left_image, right_image, disparities, window, penalty):
    """
    References:
        https://www.ipol.im/pub/art/2014/57/article_lr.pdf
        https://www.csd.uwo.ca/~oveksler/Courses//Winter2016/CS4442_9542b/L11-CV-stereo.pdf
        https://core.ac.uk/download/pdf/286357783.pdf
    TODO: Work in progress here, not finished yet!!!
    """
    assert (left_image.shape == right_image.shape)
    assert (len(left_image.shape) == 3)
    assert (disparities > 0)
    assert (window > 0)
    assert (penalty is not None)

    H, W, C = left_image.shape
    disp_map = np.zeros((H, W, C, disparities))

    # Loop over internal image
    for row in range(window, H - window):
        for col in range(window, W - window):
            # Loop over channels
            for c in range(0, C):
                # Loop over all possible disparities
                for d in range(0, disparities):
                    for v in range(-window, window + 1):
                        for u in range(-window, window + 1):
                            left = left_image[row + v, col + u, c]
                            right = right_image[row + v, col + u - d, c]
                            disp_map[row, col, c, d] += np.square(left - right)
                    int_image = cv2.integral(disp_map)
    return disp_map
