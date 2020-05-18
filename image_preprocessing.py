import cv2
import numpy as np
'''
CRZ:
For image preprocessing, image registration.'''

def gamma_normalization(img):
    '''TODO: gamma normalization for retina images.'''
    return img

def crop_mask_image(img,padding=25):
    mask_path = 'mask.jpg'
    #mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    diameter = 1650
    radius = int(1650/2)
    height,width  =img.shape[:2]
    center = (int(width/2),int(height/2))
    pad_size=radius + padding
    crop_img = img[center[1]-pad_size:center[1]+pad_size, center[0]-pad_size:center[0]+pad_size]
    mask = np.zeros(crop_img.shape, np.uint8)

    center = (radius+padding,radius+padding)
    cv2.circle(mask, center, radius, color=(255,255,255), thickness=-1)
    r_img=cv2.add(crop_img, np.zeros(np.shape(crop_img), dtype=np.uint8), mask=mask)
    return r_img

def image_preprocess(img):
    #split channels, grab green channel only.
    b, g, r = cv2.split(img)
    img0 = gamma_normalization(g)
    img1 = crop_mask_image(img0)
    return img1

def read_image_and_preprocess(image_path):
    image = cv2.imread(image_path)
    image = image_preprocess(image)
    return image

def retina_registration(imgs):
    '''TODO: registration for retina images.'''
    pass