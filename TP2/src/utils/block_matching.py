import numpy as np
from numba import jit


def get_min_disparity_ssd(l_img, r_img, d_steps, w_size, apply_dist=False, penalty=None):
    if len(l_img.shape) > 2:
        if apply_dist:
            return np.mean(np.min(__compute__ssd_rgb_optim(left_image=l_img,
                                                              right_image=r_img,
                                                              disparities=d_steps,
                                                              window=w_size,
                                                              apply_dist=apply_dist,
                                                              penalty=penalty), axis=-1), axis=-1)
        else:
            return np.mean(np.min(__compute__ssd_rgb(left_image=l_img,
                                                        right_image=r_img,
                                                        disparities=d_steps,
                                                        window=w_size), axis=-1), axis=-1)
    else:
        if apply_dist:
            return np.min(__compute__ssd_gray_optim(left_image=l_img,
                                                       right_image=r_img,
                                                       disparities=d_steps,
                                                       window=w_size,
                                                       apply_dist=apply_dist,
                                                       penalty=penalty), axis=-1)
        else:
            return np.min(__compute__ssd_gray(left_image=l_img,
                                                 right_image=r_img,
                                                 disparities=d_steps,
                                                 window=w_size), axis=-1)


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
                        dist_metric = np.square(left - right) / (np.square(left - right) + np.square(penalty))
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
                            dist_metric = np.square(left - right) / (np.square(left - right) + np.square(penalty))
                            disp_map[row, col, c, d] += dist_metric
    return disp_map


@jit(nopython=False, parallel=True, cache=True)
def __compute__ssd_rgb_agg(left_image, right_image, disparities, window, apply_dist, penalty):
    """
    References:
        https://www.csd.uwo.ca/~oveksler/Courses//Winter2016/CS4442_9542b/L11-CV-stereo.pdf
        https://imagej.net/Integral_Image_Filters.html#Block_Matching_with_Integral_Images
        https://www.ipol.im/pub/art/2014/57/article_lr.pdf
        https://www.researchgate.net/profile/Alexander-Toet/publication/259658777_Speed-up_Template_Matching_through_Integral_Image_based_Weak_Classifiers/links/00b7d52d3b53510e13000000/Speed-up-Template-Matching-through-Integral-Image-based-Weak-Classifiers.pdf

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
    # for row in range(window, H - window):
    #     for col in range(window, W - window):
    #         # Loop over channels
    #         for c in range(0, C):
    #             # Loop over window
    #             # v and u are the x,y of our local window search, used to ensure a good match
    #             # by the squared differences of the neighbouring pixels
    #             for v in range(-window, window + 1):
    #                 for u in range(-window, window + 1):
    #                     left = left_image[row + v, col + u, c]
    #                     # Loop over all possible disparities
    #                     for d in range(0, disparities):
    #                         right = right_image[row + v, col + u - d, c]
    #                         dist_metric = np.square(left - right) / (np.square(left - right) + np.square(penalty))
    #                         disp_map[row, col, c, d] += dist_metric
    # return disp_map
