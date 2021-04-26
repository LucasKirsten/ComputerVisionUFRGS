import multiprocessing

import cv2
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.feature_extraction.image import extract_patches_2d

NUM_CORES = multiprocessing.cpu_count()


def _ssd(disparity, patches_left, patches_right, apply_dist, penalty, shape):
    penalty = tf.cast(penalty ** 2, 'float32')
    patches_right = np.roll(patches_right, shift=disparity, axis=1)
    patches_right[:, :disparity] = 0

    diff_square = tf.square(patches_left - patches_right)
    if apply_dist:
        dist_metric = diff_square / (diff_square + penalty)
    else:
        dist_metric = diff_square
    costs = tf.reduce_sum(dist_metric, axis=(2, 3)).numpy()
    return costs.reshape(shape)


def compute_ssd(left_image, right_image, disparities, window_size,
                apply_dist=False, penalty=0.000001):
    assert (left_image.shape == right_image.shape)
    assert (len(left_image.shape) > 2)
    assert (disparities > 0)
    assert (window_size > 0 and window_size % 2 != 0)
    if apply_dist:
        assert penalty > 0

    pad = window_size // 2

    padded = cv2.copyMakeBorder(left_image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, 0)
    patches_left = extract_patches_2d(padded, (window_size, window_size)).astype('float32')
    patches_left = np.reshape(patches_left, (*left_image.shape[:2], *patches_left.shape[1:]))

    padded = cv2.copyMakeBorder(right_image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, 0)
    patches_right = extract_patches_2d(padded, (window_size, window_size)).astype('float32')
    patches_right = np.reshape(patches_right, (*right_image.shape[:2], *patches_right.shape[1:]))

    params = dict(
        patches_left=patches_left,
        patches_right=patches_right,
        apply_dist=apply_dist,
        penalty=penalty,
        shape=left_image.shape
    )
    with Parallel(n_jobs=NUM_CORES, prefer="threads") as parallel:
        costs = parallel(delayed(_ssd)(disparity=d, **params) for d in range(disparities))

    return np.stack(costs, axis=-1)


def __compute_aggregation(costs0, window_size, mode):
    H, W, C, D = costs0.shape

    costs = np.zeros_like(costs0)

    def __compute_y(y):
        for x in range(W):
            if mode == 'mean':
                h = y - window_size
                w = x - window_size

                costh = 0 if h < 0 else costs0[h, x]
                costw = 0 if w < 0 else costs0[y, w]
                costhw = 0 if w < 0 and h < 0 else costs0[h, w]

                costs[y, x] = (costs0[y, x] - costh - costw + costhw) / window_size ** 2.

            else:
                miny = max(0, y - window_size // 2)
                maxy = min(H, y + window_size // 2)
                minx = max(0, x - window_size // 2)
                maxx = min(W, x + window_size // 2)

                cost = costs0[miny:maxy, minx:maxx]
                costs[y, x] = np.median(cost, axis=(0, 1))

    with Parallel(n_jobs=NUM_CORES, prefer="threads") as parallel:
        parallel(delayed(__compute_y)(y) for y in range(H))

    return costs


def compute_aggregation(costs0, window_size, mode='mean'):
    assert mode in ('mean', 'median'), 'Invalid mode to compute costs with aggregation!'
    assert window_size % 2 != 0, 'Window size should be an odd value!'

    if mode == 'mean':
        integral = np.stack([cv2.integral(costs0[..., i])[1:, 1:] for i in range(costs0.shape[-1])], axis=-1)
        costs0 = np.copy(integral)
    costs0 = np.float32(costs0)

    return __compute_aggregation(costs0, window_size, mode)
