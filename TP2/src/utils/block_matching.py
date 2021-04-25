import cv2
import numpy as np
from numba import jit


def get_disparity_map(l_img, r_img, d_steps, w_size, method="default", penalty=None):
    """
    Main method that calls the corresponding block matching algorithm to be used

    Parameters
    ----------
    :param l_img : numpy.ndarray
        left input image of size (H, W) or (H, W, C)
    :param r_img : numpy.ndarray
        right input image of size (H, W) or (H, W, C)
    :param d_steps: int
        maximum disparity number
    :param w_size: int
        radius of the filter (it will be a square window (window x window))
    :param method: str
        flag used to select whether the optimized block matching algorithm or the standard one
    :param penalty : int or float
        numeric value used as a penalty to the distance metric

    Returns
    -------
    :return disp_map : numpy.ndarray
        the best depth map of size (H, W) based on the computed disparities

    """
    if method == "default":
        if len(l_img.shape) > 2:
            return np.mean(np.argmin(__compute_ssd_rgb(left_image=l_img,
                                                       right_image=r_img,
                                                       disparities=d_steps,
                                                       window=w_size,
                                                       penalty=penalty), axis=-1), axis=-1)
        else:
            return np.argmin(__compute_ssd_gray(left_image=l_img,
                                                right_image=r_img,
                                                disparities=d_steps,
                                                window=w_size,
                                                penalty=penalty), axis=-1)
    elif method == "mean":
        return __compute_ssd_agg_mean(left_image=l_img,
                                      right_image=r_img,
                                      disparities=d_steps,
                                      window=w_size,
                                      penalty=penalty)
    elif method == "median":
        return __compute_ssd_agg_median(left_image=l_img,
                                        right_image=r_img,
                                        disparities=d_steps,
                                        window=w_size,
                                        penalty=penalty,
                                        kernel_median=3)
    else:
        raise Exception("Unknown method!")


@jit(nopython=False, parallel=True, cache=True)
def __compute_ssd_gray(left_image, right_image, disparities, window, penalty):
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
                        dist_metric = np.square(left - right) if penalty is None else np.square(left - right) / (
                                np.square(left - right) + np.square(penalty))
                        disp_map[row, col, d] += dist_metric
    return disp_map


@jit(nopython=False, parallel=True, cache=True)
def __compute_ssd_rgb(left_image, right_image, disparities, window, penalty):
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
                            dist_metric = np.square(left - right) if penalty is None else np.square(left - right) / (
                                    np.square(left - right) + np.square(penalty))
                            disp_map[row, col, c, d] += dist_metric
    return disp_map


def __compute_ssd_agg_mean(left_image, right_image, disparities, window, penalty):
    if len(left_image.shape) > 2:
        costs0 = __compute_ssd_rgb(left_image, right_image, disparities, window, penalty)
        integral_img = np.stack([cv2.integral(costs0[..., i])[1:, 1:] for i in range(costs0.shape[-1])], axis=-1)
        M = len(integral_img.shape)
        costs = np.zeros(integral_img.shape[:-1])
        for y in range(costs.shape[0]):
            for x in range(costs.shape[1]):
                for c in range(costs.shape[2]):
                    h = max(0, y - M)
                    w = max(0, x - M)
                    cost = (integral_img[y, x, c] - integral_img[h, x, c] - integral_img[y, w, c] + integral_img[
                        h, w, c]) / M ** 2
                    costs[y, x, c] = np.argmin(cost)
        return np.mean(costs, axis=-1)
    else:
        costs0 = __compute_ssd_gray(left_image, right_image, disparities, window, penalty)
        integral_img = np.stack([cv2.integral(costs0[..., i])[1:, 1:] for i in range(costs0.shape[-1])], axis=-1)
        M = len(integral_img.shape)
        costs = np.zeros(integral_img.shape[:-1])
        for y in range(costs.shape[0]):
            for x in range(costs.shape[1]):
                h = max(0, y - M)
                w = max(0, x - M)
                cost = (integral_img[y, x] - integral_img[h, x] - integral_img[y, w] + integral_img[h, w]) / M ** 2
                costs[y, x] = np.argmin(cost)
        return costs


def __compute_ssd_agg_median(left_image, right_image, disparities, window, penalty, kernel_median=3):
    assert kernel_median % 2 != 0, 'M should be odd!'
    kernel_median = kernel_median // 2
    if len(left_image.shape) > 2:
        costs0 = __compute_ssd_rgb(left_image, right_image, disparities, window, penalty)
        costs = np.zeros(costs0.shape[:-1])
        h, w = costs.shape[:2]

        for y in range(costs.shape[0]):
            for x in range(costs.shape[1]):
                for c in range(costs.shape[2]):
                    miny = max(0, y - kernel_median)
                    maxy = min(h, y + kernel_median)
                    minx = max(0, x - kernel_median)
                    maxx = min(w, x + kernel_median)

                    cost = costs0[miny:maxy, minx:maxx, c]
                    costs[y, x, c] = np.median(cost)
        return np.mean(costs, axis=-1)
    else:
        costs0 = __compute_ssd_gray(left_image, right_image, disparities, window, penalty)
        costs = np.zeros(costs0.shape[:-1])
        h, w = costs.shape[:2]

        for y in range(costs.shape[0]):
            for x in range(costs.shape[1]):
                    miny = max(0, y - kernel_median)
                    maxy = min(h, y + kernel_median)
                    minx = max(0, x - kernel_median)
                    maxx = min(w, x + kernel_median)

                    cost = costs0[miny:maxy, minx:maxx]
                    costs[y, x] = np.median(cost)
        return costs
