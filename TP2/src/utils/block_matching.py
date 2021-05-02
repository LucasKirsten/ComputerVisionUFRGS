import multiprocessing

import cv2
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.feature_extraction.image import extract_patches_2d

NUM_CORES = multiprocessing.cpu_count()


def _ssd(disparity, patches_left, patches_right, apply_dist, penalty, shape):
    '''
    Aux function to calc disparities in threads.
    '''
    
    penalty = tf.cast(penalty ** 2, 'float32')
    # move patches to match disparity
    patches_right = np.roll(patches_right, shift=disparity, axis=1)
    # zero values in the borders
    patches_right[:, :disparity] = 0
    
    # square differences
    diff_square = tf.square(patches_left - patches_right)
    
    # if to calc using a more robust function
    if apply_dist:
        dist_metric = diff_square / (diff_square + penalty)
    else:
        dist_metric = diff_square
    # sum of differences using tensorflow (over 10x faster than numpy)
    costs = tf.reduce_sum(dist_metric, axis=(2, 3)).numpy()
    return costs.reshape(shape)


def compute_ssd(left_image, right_image, disparities, window_size,
                apply_dist=False, penalty=0.000001):
    """
    Compute disparity map using SSD criterion.

    Parameters
    ----------
    left_image : numpy.ndarray
        Input stereo left image.
    right_image : numpy.ndarray
        Input stereo right image.
    disparities : int
        Maximum disparity (in pixels) to be considered between the input images.
    window_size : int
        Window size to be used in the SSD criterion. Must be an odd value!
    apply_dist : bool (optional, default False)
        If to apply a more robust penalty function, defined as: p(d) = d^2/(d^2 + penalty^2)
    penalty : float (optional, default 0.000001)
        Value to be used in the penalty function.

    Returns
    -------
    costs : numpy.ndarray
        Disparity map.

    """
    assert (left_image.shape == right_image.shape)
    assert (len(left_image.shape) > 2)
    assert (disparities > 0)
    assert (window_size > 0 and window_size % 2 != 0)
    if apply_dist:
        assert penalty > 0
    
    # padding to the input image to extract patches
    pad = window_size // 2
    
    # add padding to the input image
    padded = cv2.copyMakeBorder(left_image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, 0)
    # extract patches of the image to improve speed
    # large images may use a lot of memory, but it makes the cost calcule way faster
    patches_left = extract_patches_2d(padded, (window_size, window_size)).astype('float32')
    patches_left = np.reshape(patches_left, (*left_image.shape[:2], *patches_left.shape[1:]))

    padded = cv2.copyMakeBorder(right_image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, 0)
    patches_right = extract_patches_2d(padded, (window_size, window_size)).astype('float32')
    patches_right = np.reshape(patches_right, (*right_image.shape[:2], *patches_right.shape[1:]))
    
    # default parameters to calc SSD costs
    params = dict(
        patches_left=patches_left,
        patches_right=patches_right,
        apply_dist=apply_dist,
        penalty=penalty,
        shape=left_image.shape
    )
    # calc SSD costs for each disparity in a thread
    with Parallel(n_jobs=NUM_CORES, prefer="threads") as parallel:
        costs = parallel(delayed(_ssd)(disparity=d, **params) for d in range(disparities))

    return np.stack(costs, axis=-1)


def __compute_aggregation(costs0, window_size, mode):
    '''
    Aux function to compute aggregation in threads.
    '''
    # diparity map shape
    H, W, C, D = costs0.shape

    costs = np.zeros_like(costs0)

    # aux function to compute aggregation in Y in threads
    def __compute_y(y):
        # iterate over X axis
        for x in range(W):
            if mode == 'mean':
                # get points of integral image to be used
                h = y - window_size
                w = x - window_size
                
                # validate values of integral image
                costh = 0 if h < 0 else costs0[h, x]
                costw = 0 if w < 0 else costs0[y, w]
                costhw = 0 if w < 0 and h < 0 else costs0[h, w]
                
                # compute mean using integral image
                costs[y, x] = (costs0[y, x] - costh - costw + costhw) / window_size ** 2.

            else:
                # define window points of the input disparity map
                miny = max(0, y - window_size // 2)
                maxy = min(H, y + window_size // 2)
                minx = max(0, x - window_size // 2)
                maxx = min(W, x + window_size // 2)
                
                # compute median value
                cost = costs0[miny:maxy, minx:maxx]
                costs[y, x] = np.median(cost, axis=(0, 1))
    
    # run function in threads over the Y axis
    with Parallel(n_jobs=NUM_CORES, prefer="threads") as parallel:
        parallel(delayed(__compute_y)(y) for y in range(H))

    return costs


def compute_aggregation(costs0, window_size, mode='mean'):
    """
    Compute aggregation for an input disparity map.

    Parameters
    ----------
    costs0 : numpy.ndarray
        Input disparity map.
    window_size : int
        Window size to be used to compute aggregation. Must be an odd value!
    mode : string (optional, default=mean)
        Choose between: mean OR mode. Criterion to be used to compute the aggregation.

    Returns
    -------
    costs : numpy.ndarray
        Disparity map with aggregation.

    """
    assert mode in ('mean', 'median'), 'Invalid mode to compute costs with aggregation!'
    assert window_size % 2 != 0, 'Window size should be an odd value!'

    if mode == 'mean':
        # if to use mean mode, calc the integral image for each disparity value
        integral = np.stack([cv2.integral(costs0[..., i])[1:, 1:] for i in range(costs0.shape[-1])], axis=-1)
        costs0 = np.copy(integral)
    costs0 = np.float32(costs0)
    
    # use aux function to compute aggregation in threads
    return __compute_aggregation(costs0, window_size, mode)
