import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread
from utils import block_matching

matplotlib.use('TkAgg')

if __name__ == "__main__":
    save_images = False

    window_size = 5  # size of the window to consider around the scan line point
    max_disp_steps = 50  # maximum disparity to consider
    penalty = 0.00001

    left_img_teddy_path = "../data/Teddy/teddy-png-2/im2.png"
    right_img_teddy_path = "../data/Teddy/teddy-png-2/im6.png"

    img_teddy_left = imread(left_img_teddy_path)
    img_teddy_right = imread(right_img_teddy_path)

    d_map_default = block_matching.get_disparity_map(
        l_img=img_teddy_left,
        r_img=img_teddy_right,
        d_steps=max_disp_steps,
        w_size=window_size,
        method="default",
        penalty=penalty
    )

    d_map_mean = block_matching.get_disparity_map(
        l_img=img_teddy_left,
        r_img=img_teddy_right,
        d_steps=max_disp_steps,
        w_size=window_size,
        method="mean",
        penalty=penalty
    )

    d_map_median = block_matching.get_disparity_map(
        l_img=img_teddy_left,
        r_img=img_teddy_right,
        d_steps=max_disp_steps,
        w_size=window_size,
        method="median",
        penalty=penalty
    )

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.set_title("Left Image")
    ax1.imshow(img_teddy_left)
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.set_title("Right Image")
    ax2.imshow(img_teddy_right)
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.set_title("SSD Default")
    ax3.imshow(d_map_default, cmap='gray')
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.set_title(f"SSD Mean")
    ax4.imshow(d_map_mean, cmap='gray')
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.set_title(f"SSD Median")
    ax5.imshow(d_map_median, cmap='gray')
    plt.show()
