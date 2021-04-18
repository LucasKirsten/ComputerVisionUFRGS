import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import plot_utils
from utils.converters import Converters

converters = Converters()
__CLASS_TO_ID = converters.get_class_to_id()


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def calc_partial_iou(labels, batch, metrics_dict, n_correct, n_false, epoch, writer, to_board=False, is_val=False, i_batch=None):
    images_np = batch['image_original'].cpu().numpy()
    gt_np = batch['gt'].cpu().numpy()

    image_idx = 0
    for sample_i in range(labels.shape[0]):
        current_img = images_np[sample_i, ...]
        current_labels = labels[sample_i, ...]
        current_gt = gt_np[sample_i, ...]
        valid_mask = current_gt != -1

        curr_correct = np.sum(current_gt[valid_mask] == current_labels[valid_mask])
        for key, val in __CLASS_TO_ID.items():
            if np.sum(valid_mask & (current_gt == val)) > 1:
                intersection = np.logical_and(valid_mask & (current_gt == val),
                                              valid_mask & (current_labels == val))
                union = np.logical_or(valid_mask & (current_gt == val), valid_mask & (current_labels == val))
                metrics_dict[key + '_iou'] += np.sum(intersection) / np.sum(union)
                metrics_dict[key + '_samples'] += 1

        curr_false = np.sum(valid_mask) - curr_correct
        n_correct += curr_correct
        n_false += curr_false

        if to_board and epoch % 10 == 0:
            image_name = f"Validation/{i_batch}_{image_idx}" if is_val else f"Test/{i_batch}_{image_idx}"
            image_board = plot_utils.get_board_image(current_img, current_labels)
            writer.add_image(image_name, image_board, epoch, dataformats="HWC")
            image_idx += 1

    return n_correct, n_false, metrics_dict


def get_iou(loader, metrics_dict, mean_loss, n_correct, n_false, is_train=True, is_val=False):
    mean_loss /= len(loader)
    train_acc = n_correct / (n_correct + n_false)

    for key, val in __CLASS_TO_ID.items():
        metrics_dict[key + '_iou'] /= metrics_dict[key + '_samples']
        metrics_dict['mean_iou'] += metrics_dict[key + '_iou']
    metrics_dict['mean_iou'] /= len(__CLASS_TO_ID)

    if is_train:
        metrics_dict['train_acc'] = train_acc
        metrics_dict['train_loss'] = mean_loss
    elif is_val:
        metrics_dict['val_acc'] = train_acc
        metrics_dict['val_loss'] = mean_loss
    else:
        metrics_dict['test_acc'] = train_acc
        metrics_dict['test_loss'] = mean_loss

    return metrics_dict


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, lr_scheduler, device, writer_path, scaler):
        self.test_loader = test_loader
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.writer = SummaryWriter(log_dir=writer_path)
        self.batches_per_update = 8
        self.scaler = scaler

    def train(self, class_weights, epoch):
        self.model.train()
        self.model.apply(set_bn_eval)
        mean_loss = 0.0
        n_correct = 0
        n_false = 0

        train_metrics = {'mean_iou': 0}
        for key, val in converters.get_class_to_id().items():
            train_metrics[key + '_iou'] = 0
            train_metrics[key + '_samples'] = 0

        for i_batch, sample_batched in enumerate(self.train_loader):
            output, total_loss = self.model.eval_net_with_loss(model=self.model,
                                                               batch=sample_batched,
                                                               class_weights=class_weights,
                                                               device=self.device)

            self.optimizer.zero_grad()

            if self.scaler is None:
                from apex import amp
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    self.optimizer.step()
            else:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            mean_loss += total_loss.cpu().detach().numpy()
            label_out = torch.nn.functional.softmax(output, dim=1)
            label_out = label_out.cpu().detach().numpy()
            labels = np.argmax(label_out, axis=1)

            n_correct, n_false, train_metrics = calc_partial_iou(labels=labels,
                                                                 batch=sample_batched,
                                                                 metrics_dict=train_metrics,
                                                                 n_correct=n_correct,
                                                                 n_false=n_false,
                                                                 epoch=epoch,
                                                                 writer=self.writer)
        train_metrics = get_iou(loader=self.train_loader,
                                metrics_dict=train_metrics,
                                mean_loss=mean_loss,
                                n_correct=n_correct,
                                n_false=n_false,
                                is_train=True)

        # ...log the accuracy and loss of training
        self.writer.add_scalar("Train/Acc", train_metrics['train_acc'], epoch)
        self.writer.add_scalar("Train/Loss", train_metrics['train_loss'], epoch)
        self.writer.add_scalar("Train/IOU", train_metrics['mean_iou'], epoch)

        print("\nTRAIN METRICS:")
        print(train_metrics)
        print('Train loss: %f | Train acc: %f' % (train_metrics['train_loss'], train_metrics['train_acc']))
        return train_metrics

    def val(self, epoch, class_weights):
        self.model.eval()
        self.model.apply(set_bn_eval)
        n_correct = 0
        n_false = 0
        mean_loss = 0.0

        val_metrics = {'mean_iou': 0}
        for key, val in converters.get_class_to_id().items():
            val_metrics[key + '_iou'] = 0
            val_metrics[key + '_samples'] = 0

        for i_batch, sample_batched in enumerate(self.val_loader):
            output, total_loss = self.model.eval_net_with_loss(model=self.model,
                                                               batch=sample_batched,
                                                               class_weights=class_weights,
                                                               device=self.device)
            mean_loss += total_loss.cpu().detach().numpy()
            label_out = torch.nn.functional.softmax(output, dim=1)
            label_out = label_out.cpu().detach().numpy()
            labels = np.argmax(label_out, axis=1)

            n_correct, n_false, val_metrics = calc_partial_iou(labels=labels,
                                                               batch=sample_batched,
                                                               metrics_dict=val_metrics,
                                                               n_correct=n_correct,
                                                               n_false=n_false,
                                                               epoch=epoch,
                                                               to_board=True,
                                                               is_val=True,
                                                               i_batch=i_batch,
                                                               writer=self.writer)
        val_metrics = get_iou(loader=self.val_loader,
                              metrics_dict=val_metrics,
                              mean_loss=mean_loss,
                              n_correct=n_correct,
                              n_false=n_false,
                              is_train=False,
                              is_val=True)
        self.lr_scheduler.step(val_metrics['mean_iou'])

        # ...log the accuracy and loss of training
        self.writer.add_scalar("Val/Acc", val_metrics['val_acc'], epoch)
        self.writer.add_scalar("Val/Loss", val_metrics['val_loss'], epoch)
        self.writer.add_scalar("Val/IOU", val_metrics['mean_iou'], epoch)

        print("\nVAL METRICS:")
        print(val_metrics)
        print('Val loss: %f | Val acc: %f' % (val_metrics['val_loss'], val_metrics['val_acc']))
        return val_metrics

    def test(self, epoch, class_weights):
        self.model.eval()
        self.model.apply(set_bn_eval)
        n_correct = 0
        n_false = 0
        mean_loss = 0.0

        test_metrics = {'mean_iou': 0}
        for key, val in converters.get_class_to_id().items():
            test_metrics[key + '_iou'] = 0
            test_metrics[key + '_samples'] = 0

        for i_batch, sample_batched in enumerate(self.test_loader):
            output, total_loss = self.model.eval_net_with_loss(model=self.model,
                                                               batch=sample_batched,
                                                               class_weights=class_weights,
                                                               device=self.device)
            mean_loss += total_loss.cpu().detach().numpy()
            label_out = torch.nn.functional.softmax(output, dim=1)
            label_out = label_out.cpu().detach().numpy()
            labels = np.argmax(label_out, axis=1)

            n_correct, n_false, test_metrics = calc_partial_iou(labels=labels,
                                                                batch=sample_batched,
                                                                metrics_dict=test_metrics,
                                                                n_correct=n_correct,
                                                                n_false=n_false,
                                                                epoch=epoch,
                                                                to_board=True,
                                                                is_val=False,
                                                                i_batch=i_batch,
                                                                writer=self.writer)
        test_metrics = get_iou(loader=self.test_loader,
                               metrics_dict=test_metrics,
                               mean_loss=mean_loss,
                               n_correct=n_correct,
                               n_false=n_false,
                               is_train=False,
                               is_val=False)
        self.lr_scheduler.step(test_metrics['mean_iou'])

        # ...log the accuracy and loss of training
        self.writer.add_scalar("Test/Acc", test_metrics['test_acc'], epoch)
        self.writer.add_scalar("Test/Loss", test_metrics['test_loss'], epoch)
        self.writer.add_scalar("Test/IOU", test_metrics['mean_iou'], epoch)

        print("\nTEST METRICS:")
        print(test_metrics)
        print('Test loss: %f | Test acc: %f' % (test_metrics['test_loss'], test_metrics['test_acc']))
        return test_metrics
