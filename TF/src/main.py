import pathlib as pl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from models.unet import UNet
from utils.data_loader import BacteriaDataset
from utils.plot_utils import plot_data
from utils.train_utils import train, validation, test, save_model_with_meta


BATCH_SIZE = 16
RESOLUTION = (512, 512)
NUM_WORKERS = 0
WORKING_PATH = "./Projects/ufrgs/tf"
DATA_BASE_PATH = "/media/HD/datasets/bacteria_segmentation"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATIENCE = 60
START_EPOCH = 0
MAX_EPOCHS = 1000
BEST_EPOCH = 0
BEST_VAL_ACC = -9999
EXTRA_INFO = None
WRITER_PATH = "/".join([WORKING_PATH, "/board"])

if __name__ == "__main__":
    print(f"Training on {DEVICE}")

    save_model_path = "/".join([WORKING_PATH, "checkpoints"])
    pl.Path(save_model_path).mkdir(parents=True, exist_ok=True)
    save_model_path = "/".join([save_model_path, "unet.pth"])

    pl.Path(WRITER_PATH).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=WRITER_PATH)

    train_dataset = BacteriaDataset(base_path="/".join([DATA_BASE_PATH, "train"]), resolution=RESOLUTION)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=NUM_WORKERS)
    val_dataset = BacteriaDataset(base_path="/".join([DATA_BASE_PATH, "val"]), resolution=RESOLUTION)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=NUM_WORKERS)
    test_dataset = BacteriaDataset(base_path="/".join([DATA_BASE_PATH, "test"]), resolution=RESOLUTION)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=NUM_WORKERS)

    # Model
    print('==> Building model..')
    num_classes = train_dataset.num_classes
    model = UNet(in_channels=3,
                 out_channels=num_classes,
                 n_blocks=4,
                 start_filters=1,
                 activation='relu',
                 normalization='batch',
                 conv_mode='same',
                 dim=2)
    model = model.to(DEVICE)
    print(model)
    print('==> Done!')

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # criterion = nn.functional.nll_loss
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='max',
                                                        patience=20,
                                                        verbose=True)

    # Checking shapes
    sample_batch = next(iter(train_loader))
    plot_data(sample_batch, train_dataset)
    images = sample_batch['image'].to(DEVICE)
    masks = sample_batch['mask'].to(DEVICE)
    with torch.no_grad():
        outputs = model(images)
    print(images.shape)
    print(masks.shape)
    print(outputs.shape)
    targets = torch.argmax(masks, dim=1)
    preds = torch.nn.functional.log_softmax(outputs, dim=1)
    print(preds.shape)
    print(targets.shape)
    loss = criterion(preds, targets)
    print(loss)

    # torch summary
    summary = summary(model, (3, 256, 256))

    string = "# ================================================================== # \n" \
             "#                         Starting Training!                         # \n" \
             "# ================================================================== #"
    print(string)

    for epoch in range(START_EPOCH, MAX_EPOCHS):
        train_acc, train_loss = train(epoch=epoch,
                                      model=model,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      data_loader=train_loader,
                                      device=DEVICE,
                                      writer=writer)

        val_acc, val_loss = validation(epoch=epoch,
                                       model=model,
                                       criterion=criterion,
                                       data_loader=val_loader,
                                       device=DEVICE,
                                       lr_scheduler=lr_scheduler,
                                       writer=writer)

        test_acc, test_loss = test(epoch=epoch,
                                   model=model,
                                   criterion=criterion,
                                   data_loader=test_loader,
                                   device=DEVICE,
                                   writer=writer)

        if epoch >= 10 and val_acc > BEST_VAL_ACC:
            BEST_VAL_ACC = val_acc
            BEST_EPOCH = epoch
            save_model_with_meta(save_model_path,
                                 model,
                                 optimizer,
                                 {'train_acc': train_acc,
                                  'val_acc': val_acc,
                                  'train_loss': train_loss,
                                  'val_loss': val_loss,
                                  'best_epoch': BEST_EPOCH,
                                  'additional_info': EXTRA_INFO})
            print('----- New best validation acc. Saving... -----')

        if (epoch - BEST_EPOCH) > PATIENCE:
            print(f"Finishing training, best validation acc: {BEST_VAL_ACC:.2f} at epoch: {BEST_EPOCH}")
            writer.close()
            break
