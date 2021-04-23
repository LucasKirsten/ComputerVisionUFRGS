import matplotlib.pyplot as plt
import pathlib as pl
from skimage.io import imread, imsave
import tqdm
from utils import plot_utils, block_matching

#%%
if __name__ == "__main__":
    is_plot_images = True
    save_images = False
    max_disp_steps = 50  # maximum disparity to consider
    # window_size = 10  # size of the window to consider around the scan line point
    # penalty = 50

    window_sizes = [1, 2, 4, 8, 16, 32]
    penalties = [2, 4, 8, 16, 32, 64, 128, 256]

    teddy_images_dir = "../data/saved_images/teddy"
    cones_images_dir = "../data/saved_images/cones"
    pl.Path(teddy_images_dir).mkdir(exist_ok=True, parents=True)
    pl.Path(cones_images_dir).mkdir(exist_ok=True, parents=True)

    left_img_teddy_path = "./images/teddy/im2.png"
    right_img_teddy_path = "./images/teddy/im6.png"

    img_teddy_left = imread(left_img_teddy_path)
    img_teddy_right = imread(right_img_teddy_path)
    
    left_img_cones_path = "./images/cones/im2.png"
    right_img_cones_path = "./images/cones/im6.png"

    img_cones_left = imread(left_img_cones_path)
    img_cones_right = imread(right_img_cones_path)
    
    window_size = 1
    penalty = 2

    for window_size in tqdm.tqdm(window_sizes):
        for idx, penalty in enumerate(penalties):
            teddy_disp_map = block_matching.get_min_disparity_ssd(l_img=img_teddy_left,
                                                                  r_img=img_teddy_right,
                                                                  d_steps=max_disp_steps,
                                                                  w_size=window_size)

            teddy_disp_map_pen = block_matching.get_min_disparity_ssd(l_img=img_teddy_left,
                                                                      r_img=img_teddy_right,
                                                                      d_steps=max_disp_steps,
                                                                      w_size=window_size,
                                                                      apply_dist=True,
                                                                      penalty=penalty)

            cones_disp_map = block_matching.get_min_disparity_ssd(l_img=img_cones_left,
                                                                  r_img=img_cones_right,
                                                                  d_steps=max_disp_steps,
                                                                  w_size=window_size)

            cones_disp_map_pen = block_matching.get_min_disparity_ssd(l_img=img_cones_left,
                                                                      r_img=img_cones_right,
                                                                      d_steps=max_disp_steps,
                                                                      w_size=window_size,
                                                                      apply_dist=True,
                                                                      penalty=penalty)

            if is_plot_images:
                plot_utils.plot_images(img_teddy_left, img_teddy_right, teddy_disp_map, teddy_disp_map_pen, penalty)
                plot_utils.plot_images(img_cones_left, img_cones_right, cones_disp_map, cones_disp_map_pen, penalty)
                plt.show()

            if save_images:
                saved_images_teddy = f"{teddy_images_dir}/disp_{idx}_w_{window_size}.png"
                saved_images_teddy_optim = f"{teddy_images_dir}/disp_{idx}_w_{window_size}_pen_{penalty}.png"
                saved_images_cones = f"{cones_images_dir}/disp_{idx}_w_{window_size}.png"
                saved_images_cones_optim = f"{cones_images_dir}/disp_{idx}_w_{window_size}_pen_{penalty}.png"

                imsave(saved_images_teddy, teddy_disp_map)
                imsave(saved_images_teddy_optim, teddy_disp_map_pen)
                imsave(saved_images_cones, cones_disp_map)
                imsave(saved_images_cones_optim, cones_disp_map_pen)
