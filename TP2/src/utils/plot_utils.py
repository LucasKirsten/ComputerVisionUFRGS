from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def plot_images(imageL, imageR, disp_map1, disp_map2, disp_map1_title=None, disp_map2_title=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Left Image")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Right Image")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("SSD 1") if disp_map1_title is None else ax3.set_title(disp_map1_title)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title(f"SSD 2") if disp_map2_title is None else ax4.set_title(disp_map2_title)
    if len(imageL.shape) > 2:
        ax1.imshow(imageL)
        ax2.imshow(imageR)
        ax3.imshow(disp_map1, cmap='gray')
        ax4.imshow(disp_map2, cmap='gray')
    else:
        ax1.imshow(rgb2gray(imageL), cmap='gray')
        ax2.imshow(rgb2gray(imageR), cmap='gray')
        ax3.imshow(disp_map1, cmap='gray')
        ax4.imshow(disp_map2, cmap='gray')
