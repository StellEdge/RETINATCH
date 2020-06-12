import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
from dishPosition import dishPosition
from image_preprocessing import image_preprocess_display, image_thinning

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


def feature_extracting_circle(n, stepSize, oriPic, biPic,image_radius=int(825*6/17),verbose = True):
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
        dis_img = (dis_img * 0.6 + biPic*0.4).astype(np.uint8)
        dis_img = cv2.cvtColor(dis_img,cv2.COLOR_GRAY2BGR)
        cv2.namedWindow('Extracting', cv2.WINDOW_NORMAL)
        cv2.circle(dis_img,center = (centerX, centerY),radius = 2,color = (0,255,255))
        cv2.circle(dis_img, center=(img_centerX, img_centerY), radius=image_radius, color=(0, 0, 255))


    des = []
    acc = [0]
    # extract_bifurcation(biPic)
    bifur_map = extract_bifurcation(biPic)
    # radius = []
    if verbose:
        for i in range(1, n + 1):
            cv2.circle(dis_img, center=(centerX, centerY), radius=i * stepSize, color=(0, 255, 255))
    #     radius.append((i * stepSize))
    max_radius = n*stepSize
    center_distance = np.sqrt((centerX-img_centerX)**2+(centerY-img_centerY)**2)
    over_radius = max_radius + center_distance
    if over_radius>image_radius:
        print('Failed to extract: Disk is too far away from center.')
        return np.array([])
    if verbose:
        print(over_radius,image_radius)

    count = np.zeros(shape= (n,))
    for j in range(bifur_map.shape[0]):
        for k in range(bifur_map.shape[1]):
            if bifur_map[j,k]>0:
                dis = (j - centerY) * (j - centerY) + (k - centerX) * (k - centerX)
                dis = np.sqrt(dis)
                if dis < max_radius:
                    count[int(dis/stepSize)]+=1
                    if verbose:
                        cv2.circle(dis_img, center=(k, j), radius=2, color=(0, 255, 25*int(dis/stepSize)))
                else:
                    if verbose:
                        cv2.circle(dis_img, center=(k, j), radius=2, color=(255, 255, 0))

    if verbose:
        print(count)
        cv2.imshow('Extracting',dis_img)
        cv2.waitKey(1)
        if cv2.waitKey(0) & 0xff == ord('c'):
            cv2.waitKey(1)

    return np.array(count)


def extract_circle_feature(n, stepSize, oriPic, biPic):
    '''
    alias
    :return:
    '''
    return feature_extracting_circle(n, stepSize, oriPic, biPic)


N_STEP = 10
STEP_LEN = 15


def extract_circle_feature_single(image_name):
    image = cv2.imread(image_name)
    ori_img = image_preprocess_display(image)
    bi_img = image_thinning(ori_img)
    return extract_circle_feature(N_STEP, STEP_LEN, ori_img, bi_img)


def extract_circle_features(image_names):

    descriptor_for_all = []

    #cv2.namedWindow('Extracting', cv2.WINDOW_NORMAL)
    for image_name in image_names:
        print('extracting', image_name)
        image = cv2.imread(image_name)
        if image is None:
            continue
        ori_img = image_preprocess_display(image)
        # cv2.imshow('Extracting', ori_img)
        # cv2.waitKey(1)
        bi_img = image_thinning(ori_img)
        # cv2.imshow('Extracting', bi_img)
        # cv2.waitKey(1)
        # if cv2.waitKey(0) & 0xff == ord('c'):
        #     continue

        descriptor = extract_circle_feature(N_STEP, STEP_LEN, ori_img, bi_img)
        descriptor_for_all.append(descriptor)
    return np.array(descriptor_for_all)


if __name__ == '__main__':
    import time
    t0 = time.process_time()
    print(extract_circle_feature_single('Sidra_custom/03/1.jpg'))
    t1 = time.process_time()
    print(t1-t0)
    print(extract_circle_feature_single('Sidra_custom/03/2.jpg'))
