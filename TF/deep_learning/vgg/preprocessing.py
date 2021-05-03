import numpy as np
from skimage import transform


def random_rotation_image_max_size(image, label, angle_variation=5, label_cval=2):
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

    image = transform.rotate(image, angle, resize=False)
    label = transform.rotate(label, angle, resize=False, order=0, mode='constant', cval = label_cval)

    return image, label


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
