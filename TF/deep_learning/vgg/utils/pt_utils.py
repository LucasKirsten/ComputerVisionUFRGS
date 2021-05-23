import torch
import numpy as np
import datetime
from utils.converters import Converters


converters = Converters()
__MEAN = converters.get_mean()
__STD = converters.get_std()


def get_labels(output):
    output = torch.nn.functional.softmax(output, dim=1)
    output = torch.squeeze(output)
    _, indices = torch.max(output, 0)
    return indices.detach().cpu().numpy()


def prepare_s_image_for_pt(img, device):
    img_pt = img.astype(np.float32) / 255.0
    for i in range(3):
        img_pt[..., i] -= __MEAN[i]
        img_pt[..., i] /= __STD[i]
    img_pt = img_pt.transpose(2, 0, 1)
    return torch.from_numpy(img_pt[None, ...]).to(device)


def load_model(modelClass, path, nClasses):
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model = modelClass(nClasses, device)
    model.load_state_dict(torch.load(path, map_location=device_str))
    model.eval()
    model.to(device)
    return model, device


def save_model_with_meta(file, model, optimizer, additional_info):
    dict_to_save = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'date': str(datetime.datetime.now()), 'additional_info': {}}
    dict_to_save['additional_info'].update(additional_info)
    torch.save(dict_to_save, file)


def load_model_with_meta(modelClass, path, nClasses, device_name=None):
    if device_name is not None:
        loaded_torch = torch.load(path, map_location=device_name)
        device = torch.device(device_name)
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        loaded_torch = torch.load(path, map_location=device_str)
        device = torch.device(device_str)

    model = modelClass(nClasses)

    if 'model_state_dict' in loaded_torch:
        model.load_state_dict(loaded_torch['model_state_dict'])
    else:
        model.load_state_dict(loaded_torch)

    res = loaded_torch['additional_info']['resolution'] if loaded_torch['additional_info']['resolution'] is not None else None

    model.eval()
    model.to(device)

    if 'model_state_dict' in loaded_torch:
        print('Model Info:')
        for key, item in loaded_torch.items():
            if key == 'model_state_dict' or key == 'optimizer_state_dict':
                continue
            print('%s: %s' % (key, item))

    return model, device, res
