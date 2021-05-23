import cv2
import numpy as np
import torch
from sklearn.metrics import classification_report, jaccard_score
from torch.utils.data import DataLoader
from torchsummary import summary

from dataset import BacteriaDataset
from utils import pt_utils
from utils.converters import Converters
# from vgg_unet import UNetVgg
from vgg_unet_aspp_detection import UNetVgg


def _init_loader_seed():
    np.random.seed(torch.initial_seed() % 2 ** 32)


TEST_DIR = "../../data/test"
SAVE_IMAGES_DIR = "../../data/saved_images"
CHECKPOINT_PATH = "./checkpoints/unet_vgg_aspp.pth"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
RESOLUTION = [IMAGE_HEIGHT, IMAGE_WIDTH]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = 0
NUM_EPOCHS = 1000
CLASS_WEIGHTS = [0.1, 0.55, 1]
SAVE_IMAGES = False

if __name__ == "__main__":
    converters = Converters()

    test_ds = BacteriaDataset(base_path=TEST_DIR, is_train=False, resolution=RESOLUTION)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        worker_init_fn=_init_loader_seed())

    nClasses = len(CLASS_WEIGHTS)
    model, _, _ = pt_utils.load_model_with_meta(UNetVgg, CHECKPOINT_PATH, nClasses, DEVICE)

    summary(model, (3, IMAGE_HEIGHT, IMAGE_WIDTH))
    model.to(DEVICE)
    model.eval()

    # Initialize the prediction and label lists(tensors)
    y_pred_list = []
    y_true_list = []

    for i_batch, sample_batched in enumerate(test_loader):
        filenames = sample_batched['filename']
        target_np = sample_batched['gt'].cpu().detach().view(-1).numpy()
        output, _ = model.eval_net_with_loss(model=model,
                                             batch=sample_batched,
                                             class_weights=CLASS_WEIGHTS,
                                             device=DEVICE)
        softmax = torch.nn.functional.softmax(output, dim=1)
        preds = torch.argmax(softmax, 1)
        y_pred_np = preds.cpu().detach().view(-1).view(-1).numpy()
        y_pred_list.append(y_pred_np)
        y_true_list.append(target_np)

        if SAVE_IMAGES:
            for sample_i in range(preds.shape[0]):
                current_pred = preds[sample_i, ...].cpu().detach().numpy()
                color_label = np.zeros((current_pred.shape[0], current_pred.shape[1], 3))
                color_label[current_pred == 0, :] = [0, 0, 0]
                color_label[current_pred == 1, :] = [0, 0, 255]
                color_label[current_pred == 2, :] = [0, 255, 0]
                save_path = f"{SAVE_IMAGES_DIR}/{filenames[sample_i]}_pred.png"
                cv2.imwrite(save_path, color_label)

    y_true_flat = np.array(y_true_list).reshape(-1, 1)
    y_pred_flat = np.array(y_pred_list).reshape(-1, 1)
    target_names = ["background", "erythrocytes", "spirochaete"]
    # target_names = ["Background", "Red Blood Cell", "Bacteria"]

    print(classification_report(y_true_flat, y_pred_flat, target_names=target_names))
    print('Jaccard (macro avg): ', jaccard_score(y_true_flat, y_pred_flat, average='macro'))
    print('Jaccard (weighted avg): ', jaccard_score(y_true_flat, y_pred_flat, average='weighted'))

    for class_id in np.unique(y_true_flat):
        yt = np.where(y_true_flat == class_id, 1, 0)
        yp = np.where(y_pred_flat == class_id, 1, 0)
        class_name = converters.get_id_to_class()[class_id]
        print(f'Jaccard {class_name}: ', jaccard_score(yt, yp, average='macro'))
