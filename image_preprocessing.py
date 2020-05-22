import cv2
import numpy as np
from Hilditch import hilditch
from vesselExtract import vessel_extract_api
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
    img0 = crop_mask_image(g)
    img0 = cv2.resize(img0,(600,600),interpolation=cv2.INTER_AREA)
    img1 = vessel_extract_api(img0)
    img2 = hilditch(img1)
    return img2

def image_preprocess_display(img):
    #split channels, grab green channel only.
    b, g, r = cv2.split(img)
    img0 = crop_mask_image(g)
    img0 = cv2.resize(img0,(600,600),interpolation=cv2.INTER_AREA)
    return img0

def read_image_and_preprocess(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image = image_preprocess(image)
    else:
        print('Error reading image :', image_path)
    return image

def retina_registration(imgs):
    '''TODO: registration for retina images.'''
    pass

# if __name__ == '__main__':
#     res = read_image_and_preprocess('regular-fundus-training/1/1_l1.jpg')
#     cv2.namedWindow('Result',cv2.WINDOW_NORMAL)