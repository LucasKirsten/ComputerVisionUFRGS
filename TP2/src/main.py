from time import time
import numpy as np

import matplotlib.pyplot as plt
from skimage.io import imread

from utils.block_matching import compute_ssd, compute_aggregation
from utils.plot_utils import plot_images

if __name__ == "__main__":
    max_disp_steps = 50  # maximum disparity to consider
    window_size = 7  # size of the window to consider around the scan line point
    apply_dist = True
    penalty = 100

    # left_img_path = "../data/Teddy/teddy-png-2/im2.png"
    # right_img_path = "../data/Teddy/teddy-png-2/im6.png"
    left_img_path = "../data/Cones/cones-png-2/im2.png"
    right_img_path = "../data/Cones/cones-png-2/im6.png"

    left_img = imread(left_img_path)
    right_img = imread(right_img_path)

    # Computing Block Matching using SSD
    init = time()
    costs0 = compute_ssd(left_image=left_img,
                         right_image=right_img,
                         disparities=max_disp_steps,
                         window_size=window_size,
                         apply_dist=apply_dist,
                         penalty=penalty)

    print(f'SSD time: {(time() - init):.2f} s')
    min_cost = np.mean(np.argmin(costs0, axis=-1), axis=-1)

    # Computing Block Matching using SSD + Mean Aggregation
    init = time()
    mean_agg = compute_aggregation(costs0=costs0, window_size=window_size, mode='mean')
    print(f'Mean Aggregation time: {(time() - init):.2f} s')

    min_cost_mean_agg = np.mean(np.argmin(mean_agg, axis=-1), axis=-1)
    plot_images(imageL=left_img,
                imageR=right_img,
                disp_map1=min_cost,
                disp_map2=min_cost_mean_agg,
                disp_map1_title=f"SSD with penalty of {penalty}",
                disp_map2_title=f"SSD + Mean Agg with penalty of {penalty}")

    # Computing Block Matching using SSD + Median Aggregation
    init = time()
    median_agg = compute_aggregation(costs0=costs0, window_size=window_size, mode='median')
    print(f'Median Aggregation time: {(time() - init):.2f} s')

    min_cost_median_agg = np.mean(np.argmin(median_agg, axis=-1), axis=-1)
    plot_images(imageL=left_img,
                imageR=right_img,
                disp_map1=min_cost,
                disp_map2=min_cost_median_agg,
                disp_map1_title=f"SSD with penalty of {penalty}",
                disp_map2_title=f"SSD + Median Agg with penalty of {penalty}")

    plt.show()
    if apply_dist:
        print(f"Using penalty of {penalty}")
