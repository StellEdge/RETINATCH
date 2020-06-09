import cv2
import numpy as np
from image_preprocessing import read_image_and_preprocess, get_minutiae_values

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
    image = read_image_and_preprocess(image_name)
    bifur_map = extract_bifurcation(image)
    return bifur_map

def extract_bifur_features(image_names):
    descriptor_for_all = []
    for name in image_names:
        descriptor_for_all.append(extract_bifur_feature(name))
    return np.array(descriptor_for_all)