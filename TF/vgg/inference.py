import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import pt_utils, plot_utils
from utils.converters import Converters
from vgg_unet_aspp_detection import UNetVgg

if __name__ == "__main__":
    WORKING_PATH = "/home/diegosevero/Projects/UFRGS/ComputerVisionUFRGS/TF"
    IMAGES_PATH = "/".join([WORKING_PATH, "data", "test", "images"])
    CHECKPOINT_PATH = "/".join([WORKING_PATH, "checkpoints", "unet_vgg_v2.pth"])
    N_CLASSES = 3

    converters = Converters()

    model, device, resolution = pt_utils.load_model_with_meta(UNetVgg, CHECKPOINT_PATH, N_CLASSES)

    print(model)
    print(device)
    print(resolution)

    images_paths_list = sorted(glob.glob(IMAGES_PATH + "/*.png"))

    for img_p in images_paths_list:
        image = cv2.imread(img_p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image_res = cv2.resize(image, (resolution[0], resolution[1]))[..., ::-1]
        img_np = np.ascontiguousarray(image_res)
        img_np = img_np.astype(np.float32) / 255.0
        for i in range(img_np.shape[2]):
            img_np[..., i] -= converters.get_mean()[i]
            img_np[..., i] /= converters.get_std()[i]
        img_np = img_np.transpose(2, 0, 1)
        img_pt = torch.from_numpy(img_np)
        img_pt = img_pt.unsqueeze(0).to(device)

        outputs, _ = model(img_pt)
        label_out = torch.nn.functional.softmax(outputs, dim=1)
        label_out = label_out.cpu().detach().numpy()
        labels = np.argmax(label_out, axis=1)

        current_labels = labels[0, ...]
        image_plot = plot_utils.get_board_image(image_res, current_labels)

        plt.imshow(image_plot)
        plt.show()
