from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib

def plot_images(imageL, imageR, disp_map, disp_map_optim, penalty):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Left Image")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Right Image")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("SSD")
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title(f"SSD with Penalty of {penalty}")
    if len(imageL.shape) > 2:
        ax1.imshow(imageL)
        ax2.imshow(imageR)
        ax3.imshow(disp_map, cmap='gray')
        ax4.imshow(disp_map_optim, cmap='gray')
    else:
        ax1.imshow(rgb2gray(imageL), cmap='gray')
        ax2.imshow(rgb2gray(imageR), cmap='gray')
        ax3.imshow(disp_map, cmap='gray')
        ax4.imshow(disp_map_optim, cmap='gray')
