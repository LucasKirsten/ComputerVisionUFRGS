import numpy as np
from vgg_unet_aspp_detection import UNetVgg
import torch
from utils import pt_utils, plot_utils
from dataset import BacteriaDataset
from utils.train_utils import Trainer
from torch.utils.data import DataLoader
# from apex import amp


# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 1000
PATIENCE = 200
BEST_EPOCH = 0
BEST_VAL_IOU = -9999.99
NUM_WORKERS = 0
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
RESOLUTION = [IMAGE_HEIGHT, IMAGE_WIDTH]
PIN_MEMORY = True
LOAD_MODEL = False
PLOT_DATA = True
AUGMENT_IMG = False
WORKING_PATH = "/home/diego.jardim/Projects/ufrgs/tf"
SAVE_DIR = "/".join([WORKING_PATH, "save_images"])
WRITER_PATH = "/".join([WORKING_PATH, "board_test"])
TRAIN_DIR = "/media/HD/datasets/bacteria_segmentation/train"
VAL_DIR = "/media/HD/datasets/bacteria_segmentation/val"
TEST_DIR = "/media/HD/datasets/bacteria_segmentation/test"
BATCHES_PER_UPDATE = 8
CLASS_WEIGHTS = [0.1, 0.55, 1] # [0.1, 0.4, 1]
CHECKPOINT_PATH = "/".join([WORKING_PATH, "checkpoints", "unet_vgg_test.pth"])


def get_loaders(train_dir, val_dir, test_dir, batch_size, seed, num_workers=4, pin_memory=True):
    train_ds = BacteriaDataset(base_path=train_dir, is_train=True, augmentation=AUGMENT_IMG, resolution=RESOLUTION)
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
    test_ds = BacteriaDataset(base_path=test_dir, is_train=False, resolution=RESOLUTION)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
        worker_init_fn=seed)
    return train_loader, val_loader, test_loader


# Dataloaders
def _init_loader_seed():
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_optimizer(model, core_lr=0.0075, base_lr=0.0075):
    base_vgg_weight, base_vgg_bias, core_weight, core_bias = UNetVgg.get_params_by_kind(model, 2)
    optimizer = torch.optim.SGD([{'params': base_vgg_bias, 'lr': base_lr},
                                 {'params': base_vgg_weight, 'lr': base_lr, 'weight_decay': 0.0001},
                                 {'params': core_bias, 'lr': core_lr},
                                 {'params': core_weight, 'lr': core_lr, 'weight_decay': 0.0001}],
                                momentum=0.9)
    return optimizer


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        seed=_init_loader_seed()
    )

    # Network
    nClasses = len(CLASS_WEIGHTS)
    model = UNetVgg(nClasses)
    model.init_params()
    model.to(DEVICE)

    # Optimization hyperparameters
    scaler = torch.cuda.amp.GradScaler()
    optimizer = get_optimizer(model)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, verbose=True)
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O0')

    if PLOT_DATA:
        plot_utils.plot_data(train_loader, 1, "TRAIN SAMPLE")
        plot_utils.plot_data(val_loader, 1, "VAL SAMPLE")
        plot_utils.plot_data(test_loader, 1, "TEST SAMPLE")

    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      device=DEVICE,
                      writer_path=WRITER_PATH,
                      scaler=scaler)

    for epoch in range(NUM_EPOCHS):
        print('\nEpoch %d starting...' % epoch)

        train_metrics = trainer.train(class_weights=CLASS_WEIGHTS, epoch=epoch)
        val_metrics = trainer.val(class_weights=CLASS_WEIGHTS, epoch=epoch)
        test_metrics = trainer.test(class_weights=CLASS_WEIGHTS, epoch=epoch)

        if BEST_VAL_IOU < val_metrics['mean_iou']:
            BEST_EPOCH = epoch
            BEST_VAL_IOU = val_metrics['mean_iou']
            additional_info = {
                'best_train_acc': train_metrics['train_acc'],
                'best_train_loss': train_metrics['train_loss'],
                'best_train_iou': train_metrics['mean_iou'],
                'best_val_acc': val_metrics['val_acc'],
                'best_val_iou': val_metrics['mean_iou'],
                'best_epoch': BEST_EPOCH,
                'weights': CLASS_WEIGHTS,
                'resolution': RESOLUTION,
                'augmentation': AUGMENT_IMG
            }
            if epoch > 20:
                pt_utils.save_model_with_meta(CHECKPOINT_PATH,
                                              model,
                                              optimizer,
                                              additional_info)
                print('New best validation IOU. Saving...')

        if (epoch - BEST_EPOCH) > PATIENCE:
            trainer.writer.close()
            print(f"Finishing training, best validation IOU {BEST_VAL_IOU:.2f} at epoch {BEST_EPOCH}")
            break
    trainer.writer.close()
