import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import transform
from skimage import img_as_ubyte
# from utils import img_utils

def elastic_distortion_image(image, label, label_cval = 2, sigma = 100, alpha = 2000, pad_size = 30):
    
    original_shape = image.shape
    image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="constant")
    
    if label is not None:
        label = np.pad(label, pad_size, mode="constant", constant_values = label_cval)
        
    padded_shape = image.shape[:2]
    
    dx = gaussian_filter((np.random.rand(*padded_shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*padded_shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    x, y = np.meshgrid(np.arange(padded_shape[1]), np.arange(padded_shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    
    for channel in range(3):
        image[..., channel] = map_coordinates(image[..., channel], indices, order=1).reshape(padded_shape)
        
    if label is not None:
        label = map_coordinates(label, indices, order=0).reshape(padded_shape)
        label = label[pad_size:original_shape[0]+pad_size, pad_size:original_shape[1]+pad_size]
        
        
    image = image[pad_size:original_shape[0]+pad_size, pad_size:original_shape[1]+pad_size, :]
    
    return image, label


def random_rotation_image(image, label, angle_variation = 5, preserve_size = True, label_cval = 2):
    '''
    Perform a big rotation (multiple of 90 degrees) and then perform a small one based on 
    the angle_variation parameter.
    
    If preserve_size is True, the rotated image will be resize to the original size (scikit-image 
    rotation modifies the size).
    '''
    
    direction = np.random.randint(0, 4)
    angle_variation = np.random.normal(0, angle_variation)
    
    angle = direction * 90 + angle_variation
    angle = np.clip(int(angle), 0, 360)
    
    original_image_shape = image.shape[:2]
    
    image = transform.rotate(image, angle, resize=True)
    label = transform.rotate(label, angle, resize=True, order=0, mode='constant', cval = label_cval)
    
    if preserve_size:
        image = transform.resize(image, original_image_shape)
        label = transform.resize(label, original_image_shape, order=0, mode='constant', cval = label_cval, preserve_range=True)
    
    return img_as_ubyte(image), label

def random_rotation_image_max_size(image, label, angle_variation = 5, max_side = None, label_cval = 2):
    '''
    Perform a big rotation (multiple of 90 degrees) and then perform a small one based on 
    the angle_variation parameter.
    
    If preserve_size is True, the rotated image will be resize to the original size (scikit-image 
    rotation modifies the size).
    '''
    
    direction = np.random.randint(0, 4)
    angle_variation = np.random.normal(0, angle_variation)
    
    angle = direction * 90 + angle_variation
    angle = np.clip(angle, -360, 360)
    
    image = transform.rotate(image, angle, resize=True)
    label = transform.rotate(label, angle, resize=True, order=0, mode='constant', cval = label_cval)
    
    
#    new_shape = img_utils.calculate_shape_max_side(image.shape[:2], max_side)
#    
#    if new_shape[0] != image.shape[0] or new_shape[1] != image.shape[1]:#   
#        image = transform.resize(image, new_shape)
#        label = transform.resize(label, new_shape, order=0, mode='constant', cval = label_cval, preserve_range=True)
    
    return img_as_ubyte(image), label
   
def uniform_noise_image(image, minmax = 10):
    
    uni_noise = np.random.uniform(-minmax, minmax, image.shape)
    image = image.astype("int16")
    noise_img = image + uni_noise
    return  ceil_floor_image(noise_img)


def gaussian_noise_image(image, mean=0, std=1):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    gaus_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    noise_img = image + gaus_noise
    return ceil_floor_image(noise_img)


def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image


def random_crop_image(image, label, ver_margin = 20, hon_margin = 20):
    
    sw = np.random.randint(1, hon_margin)
    ew = np.random.randint(1, hon_margin)
    sh = np.random.randint(1, ver_margin)
    eh = np.random.randint(1, ver_margin)
    
    image = image[sh:-eh, sw:-ew, :]
    
    if label is not None:
        label = label[sh:-eh, sw:-ew]
    
    return image, label