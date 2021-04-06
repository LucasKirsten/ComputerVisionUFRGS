import glob
import pathlib as pl
import random
import shutil

if __name__ == "__main__":
    ori_masks_path = "../data/masks"
    train_base_path = "../data/train"
    val_base_path = "../data/val"

    train_mask_path = "/".join([train_base_path, "masks"])
    train_img_path = "/".join([train_base_path, "images"])
    val_mask_path = "/".join([val_base_path, "masks"])
    val_img_path = "/".join([val_base_path, "images"])

    pl.Path(train_mask_path).mkdir(parents=True, exist_ok=True)
    pl.Path(train_img_path).mkdir(parents=True, exist_ok=True)
    pl.Path(val_mask_path).mkdir(parents=True, exist_ok=True)
    pl.Path(val_img_path).mkdir(parents=True, exist_ok=True)

    images_list = glob.glob("../data/images/*.png")
    random.shuffle(images_list)

    test_size = 0.3
    total_samples = len(images_list)
    val_samples = int(total_samples * test_size)
    train_samples = total_samples - val_samples

    for idx, img_path in enumerate(images_list):
        file_name = img_path.split("/")[-1]
        mask_path = "/".join([ori_masks_path, file_name])

        if idx < train_samples:
            shutil.copyfile(img_path, "/".join([train_img_path, file_name]))
            shutil.copyfile(mask_path, "/".join([train_mask_path, file_name]))
        else:
            shutil.copyfile(img_path, "/".join([val_img_path, file_name]))
            shutil.copyfile(img_path, "/".join([val_mask_path, file_name]))
