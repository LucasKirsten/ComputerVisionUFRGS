import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def plot_images(imageL, imageR, disp_map1, disp_map2=None, disp_map1_title=None, disp_map2_title=None):
    fig = plt.figure(figsize=(15, 15))
    if disp_map2 is not None:
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title("Left Image", size=20)
        ax1.imshow(imageL)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title("Right Image", size=20)
        ax2.imshow(imageR)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title("SSD 1", size=20) if disp_map1_title is None else ax3.set_title(disp_map1_title, size=20)
        ax3.imshow(disp_map1, cmap='gray')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title("SSD 2", size=20) if disp_map2_title is None else ax4.set_title(disp_map2_title, size=20)
        ax4.imshow(disp_map2, cmap='gray')
    else:
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title("Left Image", size=20)
        ax1.imshow(imageL)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title("Right Image", size=20)
        ax2.imshow(imageR)
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title("SSD", size=20) if disp_map1_title is None else ax3.set_title(disp_map1_title, size=20)
        ax3.imshow(disp_map1, cmap='gray')
