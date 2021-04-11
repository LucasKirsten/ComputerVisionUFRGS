import time
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
import cv2


def save_model_with_meta(file, model, optimizer, additional_info):
    dict_to_save = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'date': str(datetime.datetime.now()), 'additional_info': {}}
    dict_to_save['additional_info'].update(additional_info)
    torch.save(dict_to_save, file)


def get_images_with_labels(image, pred, id_to_class, class_to_color):
    # image_np = image.transpose(1, 2, 0)
    color_label = np.zeros((image.shape[0], image.shape[1], 3))
    for key, val in id_to_class.items():
        color_label[np.where((pred == key).all(axis=2))] = class_to_color[val]
    return (image / 255) * 0.5 + (color_label / 255) * 0.5


def train(epoch, model, criterion, optimizer, data_loader, device, writer):
    model.train()

    n_correct = 0
    n_false = 0
    epoch_loss = []
    time_train = []
    batch_size = data_loader.batch_size

    for step, sample_batched in enumerate(data_loader):
        start_time = time.time()

        original_img = sample_batched['original_img'].to(device)
        original_mask = sample_batched['original_mask'].to(device)
        inputs = sample_batched['image'].to(device)

        outputs = model(inputs)
        targets = torch.argmax(original_mask, dim=3)
        preds = torch.nn.functional.log_softmax(outputs, dim=1)

        optimizer.zero_grad()
        loss = criterion(preds, targets)

        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

        original_img_np = original_img.cpu().numpy()
        original_mask_np = original_mask.cpu().numpy()

        inputs_np = inputs.cpu().numpy()
        predicted_labels = preds.cpu().detach().numpy()
        # predicted_labels = np.argmax(predicted_labels, axis=1)

        for sample_i in range(predicted_labels.shape[0]):
            current_target = original_mask_np[sample_i, ...]
            current_inputs = inputs_np[sample_i, ...].transpose(1, 2, 0)
            current_prediction = predicted_labels[sample_i, ...].transpose(1, 2, 0)         
            valid_mask = current_target != -1

            curr_correct = np.sum(current_target[valid_mask] == current_prediction[valid_mask])
            curr_false = np.sum(valid_mask) - curr_correct
            n_correct += curr_correct
            n_false += curr_false

        time_train.append(time.time() - start_time)

    train_acc = n_correct / (n_correct + n_false)
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    writer.add_scalar("Train/Acc", train_acc, epoch)
    writer.add_scalar("Train/Loss", average_epoch_loss_train, epoch)

    # print statistics
    print(f"Epoch: {epoch} => "
          f"Train Acc: {train_acc:.2f} | "
          f"Train Loss: {average_epoch_loss_train:.2f} | "
          f"Avg time/img: {(sum(time_train) / len(time_train)) / batch_size:.2f} s")

    return train_acc, average_epoch_loss_train


def validation(epoch, model, criterion, data_loader, device, writer, lr_scheduler=None):
    model.eval()

    n_correct = 0
    n_false = 0
    epoch_loss_val = []
    time_val = []
    batch_size = data_loader.batch_size

    for step, sample_batched in enumerate(data_loader):
        start_time = time.time()

        original_img = sample_batched['original_img'].to(device)
        original_mask = sample_batched['original_mask'].to(device)
        inputs = sample_batched['image'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            targets = torch.argmax(original_mask, dim=3)
            preds = torch.nn.functional.log_softmax(outputs, dim=1)
            loss = criterion(preds, targets)

        original_img_np = original_img.cpu().numpy()
        original_mask_np = original_mask.cpu().numpy()
        inputs_np = inputs.cpu().numpy()
        predicted_labels = preds.cpu().detach().numpy()

        image_idx = 0
        for sample_i in range(predicted_labels.shape[0]):
            current_target = original_mask_np[sample_i, ...]
            current_inputs = inputs_np[sample_i, ...].transpose(1, 2, 0)
            current_prediction = predicted_labels[sample_i, ...].transpose(1, 2, 0)         
            valid_mask = current_target != -1

            curr_correct = np.sum(current_target[valid_mask] == current_prediction[valid_mask])
            curr_false = np.sum(valid_mask) - curr_correct
            n_correct += curr_correct
            n_false += curr_false

            if step < 2:
                image_board = get_images_with_labels(current_inputs,
                                                     current_prediction,
                                                     data_loader.dataset.id_to_class,
                                                     data_loader.dataset.class_to_color)
                image_name = f"Validation/doc_{step}_{image_idx}"
                writer.add_image(image_name, image_board, epoch, dataformats="HWC")
                image_idx += 1

        epoch_loss_val.append(loss.item())
        time_val.append(time.time() - start_time)

    val_acc = n_correct / (n_correct + n_false)
    average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
    lr_scheduler.step(val_acc)

    writer.add_scalar("Val/Acc", val_acc, epoch)
    writer.add_scalar("Val/Loss", average_epoch_loss_val, epoch)

    # print statistics
    print(f"Epoch: {epoch} => "
          f"Val   Acc: {val_acc:.2f} | "
          f"Val   Loss: {average_epoch_loss_val:.2f} | "
          f"Avg time/img: {(sum(time_val) / len(time_val)) / batch_size:.2f} s")

    return val_acc, average_epoch_loss_val


def test(epoch, model, criterion, data_loader, device, writer):
    model.eval()

    n_correct = 0
    n_false = 0
    epoch_loss_test = []
    time_test = []
    batch_size = data_loader.batch_size

    for step, sample_batched in enumerate(data_loader):
        start_time = time.time()
        
        original_img = sample_batched['original_img'].to(device)
        original_mask = sample_batched['original_mask'].to(device)
        inputs = sample_batched['image'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            targets = torch.argmax(original_mask, dim=3)
            preds = torch.nn.functional.log_softmax(outputs, dim=1)
            loss = criterion(preds, targets)

        original_img_np = original_img.cpu().numpy()
        original_mask_np = original_mask.cpu().numpy()
        inputs_np = inputs.cpu().numpy()
        predicted_labels = preds.cpu().detach().numpy()

        image_idx = 0
        for sample_i in range(predicted_labels.shape[0]):
            current_target = original_mask_np[sample_i, ...]
            current_inputs = inputs_np[sample_i, ...].transpose(1, 2, 0)
            current_prediction = predicted_labels[sample_i, ...].transpose(1, 2, 0)         
            valid_mask = current_target != -1

            curr_correct = np.sum(current_target[valid_mask] == current_prediction[valid_mask])
            curr_false = np.sum(valid_mask) - curr_correct
            n_correct += curr_correct
            n_false += curr_false

            if step < 2:
                image_board = get_images_with_labels(current_inputs,
                                                     current_prediction,
                                                     data_loader.dataset.id_to_class,
                                                     data_loader.dataset.class_to_color)
                image_name = f"Testing/doc_{step}_{image_idx}"
                writer.add_image(image_name, image_board, epoch, dataformats="HWC")
                image_idx += 1

        epoch_loss_test.append(loss.item())
        time_test.append(time.time() - start_time)

    test_acc = n_correct / (n_correct + n_false)
    average_epoch_loss_test = sum(epoch_loss_test) / len(epoch_loss_test)

    writer.add_scalar("Test/Acc", test_acc, epoch)
    writer.add_scalar("Test/Loss", average_epoch_loss_test, epoch)

    # print statistics
    print(f"Epoch: {epoch} => "
          f"Test  Acc: {test_acc:.2f} | "
          f"Test  Loss: {average_epoch_loss_test:.2f} | "
          f"Avg time/img: {(sum(time_test) / len(time_test)) / batch_size:.2f} s")

    return test_acc, average_epoch_loss_test
