import cv2
import numpy as np
from image_preprocessing import read_image_and_preprocess, get_minutiae_values,image_preprocess,image_preprocess_display,image_thinning

cells = [(-1, -1),
         (-1, 0),
         (-1, 1),
         (0, 1),
         (1, 1),
         (1, 0),
         (1, -1),
         (0, -1),
         (-1, -1)]

def get_minutiae_values(img):
    '''
    Get minutiae map for image
    :param img: a bi-valued image
    :return: map of crossing numbers
    '''
    img = np.where(img > 0, 1, 0)  # Binarization
    pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    values = np.zeros(shape=(9, img.shape[0], img.shape[1]))
    for k in range(0, 9):
        values[k] = pad_img[1 + cells[k][0]:img.shape[0] + 1 + cells[k][0],
                    1 + cells[k][1]:img.shape[1] + 1 + cells[k][1]]
    crossings = np.zeros(shape=img.shape)
    for k in range(0, 8):
        crossings += np.abs(values[k] - values[k + 1])
    crossings /= 2
    return crossings

from dishPosition import dishPosition
def disk_mask(oriPic,bifur_map,image_radius=int(825*6/17),verbose = True):
    """
    :param n: size ot return list
    :param stepSize:  r distance between two circle
    :param oriPic:  picture with original color
    :param biPic:  bi-picture with only 1-pixel width vessels
    :return:
    """
    centerX, centerY = dishPosition(oriPic)
    img_centerY = int(oriPic.shape[0]/2)
    img_centerX = int(oriPic.shape[1]/2)
    if verbose:
        dis_img = oriPic.copy()
        dis_img = cv2.cvtColor(dis_img,cv2.COLOR_GRAY2BGR)
        cv2.namedWindow('Extracting', cv2.WINDOW_NORMAL)
        cv2.circle(dis_img,center = (centerX, centerY),radius = 2,color = (0,255,255))
        cv2.circle(dis_img, center=(img_centerX, img_centerY), radius=image_radius, color=(0, 0, 255))


    max_radius = 150
    center_distance = np.sqrt((centerX-img_centerX)**2+(centerY-img_centerY)**2)
    over_radius = max_radius + center_distance
    if over_radius>image_radius:
        print('Failed to extract: Disk is too far away from center.')
        return np.zeros_like(bifur_map)
    if verbose:
        print(over_radius,image_radius)

    for j in range(bifur_map.shape[0]):
        for k in range(bifur_map.shape[1]):
            if bifur_map[j,k]>0:
                dis = (j - centerY) * (j - centerY) + (k - centerX) * (k - centerX)
                dis = np.sqrt(dis)
                if dis > max_radius:
                    bifur_map[j, k]=0
                    if verbose:
                        cv2.circle(dis_img, center=(k, j), radius=2, color=(0, 255, 255))
    if verbose:
        cv2.imshow('Extracting',dis_img)
        cv2.waitKey(1)
        if cv2.waitKey(0) & 0xff == ord('c'):
            cv2.waitKey(1)
    return bifur_map

def extract_bifurcation(img):
    '''
    extract bifurcation and ending points.
    :param img: bi-valued image
    :return: a list of bifurcation points
    '''
    minutiae_map = get_minutiae_values(img)
    bifurcation_points = np.where(np.logical_and(minutiae_map == 3, img > 0), 1, 0)
    return bifurcation_points

def extract_bifur_feature(image_name):
    image = cv2.imread(image_name)
    if image is not None:
        pro_image = image_preprocess_display(image)
        thin_image = image_thinning(pro_image.copy())
        bifur_map = extract_bifurcation(thin_image)
        bifur_map = disk_mask(pro_image,bifur_map)
    return bifur_map

def extract_bifur_features(image_names):
    descriptor_for_all = []
    for name in image_names:
        descriptor_for_all.append(extract_bifur_feature(name))
    return np.array(descriptor_for_all)

extract_bifur_feature('Sidra_custom/01/1.jpg')