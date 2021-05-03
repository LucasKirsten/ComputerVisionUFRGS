''' All main function are here '''

import cv2
import numpy as np
from skimage.feature import hog
from skimage.exposure import equalize_adapthist

def hog_extractor(image):
    """
    Return defined hog feature extractor.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    hog : numpy.ndarray
        HOG features.

    """
    return hog(image, orientations=8, pixels_per_cell=(8,8),
               cells_per_block=(1,1), visualize=False, multichannel=True)

def get_mask_proposal(image):
    """
    Return proposed masks.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    sure_ery : numpy.ndarray
        Possible ery class masks.
    spi : numpy.ndarray
        Possible spi class mask.

    """
    # define kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    # augment contrast of objects in scene
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[...,0]
    blur = equalize_adapthist(lab, clip_limit=0.01)
    blur = np.uint8(blur*255)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # remove noises
    contour = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    segmented = np.copy(th)
    for cnt in contour:
        perimeter = cv2.arcLength(cnt,True)
        if perimeter>8:
            continue
        
        # remove noise
        draw = np.ones_like(segmented)
        cv2.drawContours(draw, [cnt], 0, (0,0,0), -1)
        
        segmented = segmented * draw
    mask = np.uint8(segmented)
        
    # split possible ery x spi classes
    kernel = np.ones((3,3),np.uint8)
    unknown = cv2.erode(mask, kernel, iterations=3)
    sure_ery = cv2.dilate(unknown, kernel, iterations=3)
    
    # unknown labels
    unknown = cv2.subtract(mask, sure_ery)
    
    # probably spi
    contour = cv2.findContours(unknown, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[1]
    spi = np.zeros_like(sure_ery)
    for cnt in contour:
            
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        if area<1 or perimeter<1:
            continue
        
        _ = cv2.drawContours(spi, [cnt], 0, 255, -1)
    
    return sure_ery, spi

def get_feats_labels(image, mask, label, return_contours=False):
    
    """
    Extract features from mask and assing label.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    mask : numpy.ndarray
        Input mask to guide features and labels extraction from segments.
    label : int
        Label value to assign to all mask segments.
    return_contours : bool (optional, default False)
        Either to return the contours from the input mask.

    Returns
    -------
    features : numpy.ndarray
        Features extracted from all segments using HOG.
    labels : numpy.ndarray
        Labels to use in some classifier.
    cnts_return : numpy.ndarray
        If return_contours=True, returns the contours from the mask segments.

    """
    
    # find contours in mask to determine segments
    contour = cv2.findContours(mask.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]
    
    # iterate over contours
    features, labels, cnts_return = [],[],[]
    for cnt in contour:
        if len(cnt)<2:
            continue
        
        # remove very small regions
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area<1 or perimeter<1:
            continue
        
        # get min and max X and Y values from contour
        xmin = int(min(cnt[:,:,0]))
        xmax = int(max(cnt[:,:,0]))
        ymin = int(min(cnt[:,:,1]))
        ymax = int(max(cnt[:,:,1]))
        
        # define a window of at least 32x32 pixels
        if xmax-xmin<32:
            aug = (xmax-xmin)//2
            xmin = max(xmin-aug, 0)
            xmax = min(xmax+aug, image.shape[1])
        if ymax-ymin<32:
            aug = (ymax-ymin)//2
            xmin = max(ymin-aug, 0)
            xmax = min(ymax+aug, image.shape[0])
        
        # draw contours for mask and apply this on the input image
        mask = cv2.drawContours(np.zeros_like(image), [cnt], 0, (1,1,1), -1)
        crop = image * mask
        # crop the region and resize it to 32x32 (same number of features)
        crop = crop[ymin:ymax, xmin:xmax]
        crop = cv2.resize(crop, (32,32))
        
        # extract HOG features
        feats = hog_extractor(crop)
        
        features.append(feats)
        labels.append(label)
        cnts_return.append(cnt)
    
    if return_contours:
        return features, labels, cnts_return
    else:
        return features, labels
    
def post_process(ery_mask, spi_mask):

    """
    Post process predicted masks.

    Parameters
    ----------
    ery_mask : numpy.ndarray
        Input predicted ery masks.
    spi_mask : numpy.ndarray
        Input predicted spi masks.

    Returns
    -------
    ery_mask : numpy.ndarray
        Post process ery masks.
    spi_mask : numpy.ndarray
        Post process spi masks.

    """
    
    # dilate ery_masks to fill gaps
    kernel = np.ones((5,5),np.uint8)
    ery_mask = cv2.dilate(ery_mask, kernel, iterations=2)
    
    # fill ery masks
    contour = cv2.findContours(ery_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contour:
        _ = cv2.drawContours(ery_mask, [cnt], -1, (255,255,255), -1)
    
    # erode mask to not overfill the regions
    ery_mask = cv2.erode(ery_mask, kernel, iterations=2)
    
    # remove regions of ery that may be included in the spi mask
    spi_mask[ery_mask==255] = 0
    
    return ery_mask, spi_mask

def get_background_descriptor(y_true, y_pred):
    """
    Return background segments for training a classifier.
    The background refers to possible False Positives when appling vanilla mask proposal.

    Parameters
    ----------
    y_true : list<numpy.ndarray>
        Ground truth masks.
    y_pred : list<numpy.ndarray>
        Input predicted spi masks.

    Returns
    -------
    background : numpy.ndarray
        Background mask.

    """
    
    # define FP examples from proposed masks compared to ground truth ones
    background = (y_pred[0]|y_pred[1]).astype('float') - (y_true[1]|y_true[0]).astype('float')
    background[background<0] = 0

    # iterate over contours and remove large regions
    contour = cv2.findContours(background.astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contour:
        perimeter = cv2.arcLength(cnt,True)
        if perimeter<30:
            continue

        # remove noise
        draw = np.ones_like(background)
        cv2.drawContours(draw, [cnt], 0, (0,0,0), -1)

        background = background * draw
        
    return background.astype('uint8')

def predict_masks(image, clf):
    """
    Returns ery and spi predicted masks based on an input image and sklearn classifier.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    clf : sklearn classifier
        Some sklearn classifier.

    Returns
    -------
    ery_pred : numpy.ndarray
        Predicted ery mask.
    spi_pred : numpy.ndarray
        Predicted spi mask.

    """
    
    # get mask proposal
    y_pred  = list(get_mask_proposal(image))
    
    # get features and contours (label can be ignored)
    feats,_,contours = get_feats_labels(image, y_pred[0], -1, return_contours=True)
    
    # fill the predicted masks
    ery_pred = np.zeros_like(y_pred[0])
    spi_pred = np.zeros_like(y_pred[1])
    
    if len(feats)>0:
        class_pred = clf.predict(feats)
        
        for cnt, cp in zip(contours, class_pred):
            if cp==1:
                _ = cv2.drawContours(ery_pred, [cnt], 0, (255,255,255), -1)
            elif cp==2:
                _ = cv2.drawContours(spi_pred, [cnt], 0, (255,255,255), -1)
    
    feats,_,contours = get_feats_labels(image, y_pred[1], -1, return_contours=True)
    if len(feats)>0:
        class_pred = clf.predict(feats)
        for cnt, cp in zip(contours, class_pred):
            if cp==1:
                _ = cv2.drawContours(ery_pred, [cnt], 0, (255,255,255), -1)
            elif cp==2:
                _ = cv2.drawContours(spi_pred, [cnt], 0, (255,255,255), -1)
    
    return ery_pred, spi_pred