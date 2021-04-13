import numpy as np
import matplotlib.pyplot as plt
from vgg_unet_aspp_detection import UNetVgg
import torch
from utils import pt_utils, plot_utils
from dataset import BacteriaDataset
from loss import DiceLoss
from torch.utils.data import DataLoader
import cv2
import os
from apex import amp

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 500
PATIENCE = 50
BEST_EPOCH = 0
BEST_VAL_ACC = -9999.99
NUM_WORKERS = 2
IMAGE_HEIGHT = 128  # 1280 originally
IMAGE_WIDTH = 128  # 1918 originally
MAX_SIDE = 128
RESOLUTION = [IMAGE_HEIGHT, IMAGE_WIDTH]
PIN_MEMORY = True
LOAD_MODEL = False
PLOT_DATA = True
WORKING_PATH = "/home/diego.jardim/Projects/ufrgs/tf"
SAVE_DIR = "/".join([WORKING_PATH, "save_images"])
WRITER_PATH = "/".join([WORKING_PATH, "board"])
TRAIN_DIR = "/media/HD/datasets/bacteria_segmentation/train"
VAL_DIR = "/media/HD/datasets/bacteria_segmentation/val"
BATCHES_PER_UPDATE = 8
CLASS_WEIGHTS = [0.1, 0.4, 1]
CHECKPOINT_PATH = "/".join([WORKING_PATH, "checkpoints", "unet_vgg.pth"])


def get_loaders(train_dir, val_dir, batch_size, seed, num_workers=4, pin_memory=True):
    train_ds = BacteriaDataset(base_path=train_dir, is_train=True, resolution=RESOLUTION)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed)
    val_ds = BacteriaDataset(base_path=val_dir, is_train=False, resolution=RESOLUTION)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
        worker_init_fn=seed)
    return train_loader, val_loader


# Dataloaders
def _init_loader_seed():
    np.random.seed(torch.initial_seed() % 2 ** 32)


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


if __name__ == "__main__":
    train_loader, val_loader = get_loaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        seed=_init_loader_seed()
    )

    ## Network
    nClasses = train_loader.dataset.num_classes
    model = UNetVgg(nClasses)
    model.init_params()
    model.to(DEVICE)

    # Optimization hyperparameters
    core_lr = 0.0075
    base_lr = 0.0075
    base_vgg_weight, base_vgg_bias, core_weight, core_bias = UNetVgg.get_params_by_kind(model, 2)

    optimizer = torch.optim.SGD([{'params': base_vgg_bias, 'lr': base_lr},
                                 {'params': base_vgg_weight, 'lr': base_lr, 'weight_decay': 0.00005},
                                 {'params': core_bias, 'lr': core_lr},
                                 {'params': core_weight, 'lr': core_lr, 'weight_decay': 0.00005}], momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=60, verbose=True)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O0')

    criterion = DiceLoss()

    # Start training...
    best_val_acc = -1
    best_epoch = 0
    g_i_train = 0

    if PLOT_DATA:
        plot_utils.plot_data(train_loader, 1)

    for epoch in range(NUM_EPOCHS):
        print('Epoch %d starting...' % (epoch + 1))

        model.train()
        model.apply(set_bn_eval)

        mean_loss = 0.0
        n_correct = 0
        n_false = 0

        train_metrics = {'pixelwise_acc': 0, 'mean_iou': 0}
        for key, val in train_loader.dataset.class_to_id.items():
            train_metrics[key + '_iou'] = 0
            train_metrics[key + '_samples'] = 0

        optimizer.zero_grad()

        for i_batch, sample_batched in enumerate(train_loader):
            image = sample_batched['image'].to(DEVICE)
            image_np = sample_batched['image_original'].cpu().numpy()
            gt = sample_batched['gt'].to(DEVICE)

            output, total_loss = model.eval_net_with_loss(model, image, gt, CLASS_WEIGHTS, DEVICE)
            total_loss = total_loss / BATCHES_PER_UPDATE

            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (g_i_train + 1) % BATCHES_PER_UPDATE == 0:
                optimizer.step()
                optimizer.zero_grad()

            mean_loss += total_loss.cpu().detach().numpy()

            label_out = torch.nn.functional.softmax(output, dim=1)
            label_out = label_out.cpu().detach().numpy()
            labels = np.argmax(label_out, axis=1)

            gt = gt.cpu().numpy()
            limit_img = 0
            for sample_i in range(labels.shape[0]):
                current_labels = labels[sample_i, ...]
                current_gt = gt[sample_i, ...]
                current_img = image_np[sample_i, ...]

                if mean_loss < 0.009 and limit_img < 5:
                    plot_utils.plot_val_data(current_img, current_labels,
                                             train_loader.dataset.id_to_class,
                                             train_loader.dataset.class_to_color)
                    limit_img += 1

                valid_mask = current_gt != -1
                curr_correct = np.sum(current_gt[valid_mask] == current_labels[valid_mask])

                for key, val in train_loader.dataset.class_to_id.items():
                    if np.sum(valid_mask & (current_gt == val)) > 1:
                        intersection = np.logical_and(valid_mask & (current_gt == val),
                                                      valid_mask & (current_labels == val))
                        union = np.logical_or(valid_mask & (current_gt == val), valid_mask & (current_labels == val))
                        train_metrics[key + '_iou'] += np.sum(intersection) / np.sum(union)
                        train_metrics[key + '_samples'] += 1

                curr_false = np.sum(valid_mask) - curr_correct
                n_correct += curr_correct
                n_false += curr_false
            g_i_train += 1

        mean_loss /= len(train_loader)
        train_acc = n_correct / (n_correct + n_false)

        for key, val in train_loader.dataset.class_to_id.items():
            train_metrics[key + '_iou'] /= train_metrics[key + '_samples']
            train_metrics['mean_iou'] += train_metrics[key + '_iou']
        train_metrics['mean_iou'] /= len(train_loader.dataset.class_to_id) - 2

        print(train_metrics)
        print('Train loss: %f, train acc: %f' % (mean_loss, train_acc))

        # Evaluate network on the validation dataset
        n_correct = 0
        n_false = 0

        val_metrics = {'pixelwise_acc': 0, 'mean_iou': 0}
        for key, val in train_loader.dataset.class_to_id.items():
            val_metrics[key + '_iou'] = 0
            val_metrics[key + '_samples'] = 0

        model.eval()
        for i_batch, sample_batched in enumerate(val_loader):
            image = sample_batched['image'].to(DEVICE)
            image_np = sample_batched['image_original'].cpu().numpy()
            gt = sample_batched['gt'].cpu().numpy()

            label_out, _ = model(image)
            label_out = torch.nn.functional.softmax(label_out, dim=1)
            label_out = label_out.cpu().detach().numpy()
            labels = np.argmax(label_out, axis=1)

            for sample_i in range(labels.shape[0]):
                current_labels = labels[sample_i, ...]
                current_gt = gt[sample_i, ...]
                current_img = image_np[sample_i, ...]

                if PLOT_DATA and epoch >= 50:
                    if mean_loss < 0.009 and limit_img < 5:
                        plot_utils.plot_val_data(current_img, current_labels, val_loader.dataset.id_to_class,
                                                 val_loader.dataset.class_to_color)
                        limit_img += 1

            valid_mask = gt != -1
            for key, val in val_loader.dataset.class_to_id.items():
                if np.sum(valid_mask & (gt == val)) > 1:
                    intersection = np.logical_and(valid_mask & (gt == val), valid_mask & (labels == val))
                    union = np.logical_or(valid_mask & (gt == val), valid_mask & (labels == val))
                    iou = np.sum(intersection) / np.sum(union)
                    val_metrics[key + '_iou'] += iou
                    val_metrics[key + '_samples'] += 1

            curr_correct = np.sum(gt[valid_mask] == labels[valid_mask])
            curr_false = np.sum(valid_mask) - curr_correct
            n_correct += curr_correct
            n_false += curr_false

        for key, val in val_loader.dataset.class_to_id.items():
            val_metrics[key + '_iou'] /= val_metrics[key + '_samples']
            val_metrics['mean_iou'] += val_metrics[key + '_iou']

        val_metrics['mean_iou'] /= len(train_loader.dataset.class_to_id) - 2
        print(val_metrics)

        val_acc = n_correct / (n_correct + n_false)
        lr_scheduler.step(val_acc)

        if BEST_VAL_ACC < val_acc:
            BEST_VAL_ACC = val_acc
            if epoch > 20:
                pt_utils.save_model_with_meta(CHECKPOINT_PATH, model, optimizer, {'acc_train': train_acc, 'val_acc': val_acc})
                print('New best validation acc. Saving...')
            best_epoch = epoch

        if (epoch - BEST_EPOCH) > PATIENCE:
            print("Fnishing training, best validation acc %f", BEST_VAL_ACC)
            break

        print('Val acc: %f -- Best val acc: %f -- epoch %d.' % (val_acc, BEST_VAL_ACC, BEST_EPOCH))
