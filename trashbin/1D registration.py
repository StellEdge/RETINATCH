import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
from image_preprocessing import image_preprocess_display,vessel_extract_api

def get_yellow_spot(img):
    binaryThreshold = 0.2
    imgGray = img.copy()
    imgx = imgGray.shape[0]
    imgy = imgGray.shape[1]
    if imgx > imgy:
        imgr = imgy / 2
    else:
        imgr = imgx / 2

    for i in range(imgx):
        for j in range(imgy):
            if imgGray[i][j] < 5:
                imgGray[i][j] = 255
    imgGray = imgGray.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(8, 8))
    imgCLAHE = clahe.apply(imgGray)

    #cv2.imwrite('imgCLAHE.jpg', imgCLAHE)

    minValue = np.min(imgCLAHE)
    maxValue = np.max(imgCLAHE)
    imgBinary = imgCLAHE > (binaryThreshold * (maxValue - minValue) + minValue)
    imgBinary = imgBinary * maxValue
    imgBinary = imgBinary.astype(np.uint8)

    #cv2.imwrite('imgBinary.jpg', imgBinary)

    kernel = np.ones((8, 8), np.uint8)
    imgDilate = cv2.dilate(imgBinary, kernel)
    imgErode = cv2.erode(imgDilate, kernel)

    #print(imgBinary)
    #print(imgErode)
    # x_min = 100000
    # x_max = 0
    # y_min = 100000
    # y_max = 0
    #
    # for i in range(imgx):
    #     for j in range(imgy):
    #         if imgErode[i][j] == 0:
    #             if i<x_min:
    #                 x_min = i
    #             if i>x_max:
    #                 x_max = i
    #             if j < y_min:
    #                 y_min = j
    #             if j > y_max:
    #                 y_max = j
    #
    # centerX = (x_max+x_min)/2
    # centerY = (y_max+y_min)/2

    allNum = 0
    allX = 0
    allY = 0
    for i in range(imgx):
        for j in range(imgy):
            if imgErode[i][j] == 0:
                allX = allX + i
                allY = allY + j
                allNum = allNum + 1

    centerX = allX / allNum
    centerY = allY / allNum
    return np.array([centerX, centerY])

def get_optic_disk(img):
    binaryThreshold = 0.8

    imgGray = img.copy()
    imgx = imgGray.shape[0]
    imgy = imgGray.shape[1]
    if imgx > imgy:
        imgr = imgy / 2
    else:
        imgr = imgx / 2

    # for i in range(imgx):
    # 	for j in range(imgy):
    # 		if imgGray[i][j] < 5:
    # 			imgGray[i][j] = 255
    imgGray = imgGray.astype(np.uint8)

    # cv2.imwrite('imgGray.jpg', imgGray)

    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(8, 8))
    imgCLAHE = clahe.apply(imgGray)

    # cv2.imwrite('imgCLAHE.jpg', imgCLAHE)

    minValue = np.min(imgCLAHE)
    maxValue = np.max(imgCLAHE)
    imgBinary = imgCLAHE < (binaryThreshold * (maxValue - minValue) + minValue)
    imgBinary = imgBinary * maxValue
    imgBinary = imgBinary.astype(np.uint8)

    # cv2.imwrite('imgBinary.jpg', imgBinary)

    kernel = np.ones((8, 8), np.uint8)
    imgDilate = cv2.dilate(imgBinary, kernel)
    imgErode = cv2.erode(imgDilate, kernel)

    # print(imgBinary)
    # print(imgErode)

    allNum = 0
    allX = 0
    allY = 0
    # x_min = 100000
    # x_max = 0
    # y_min = 100000
    # y_max = 0
    #
    # for i in range(imgx):
    #     for j in range(imgy):
    #         if imgErode[i][j] == 0:
    #             if i<x_min:
    #                 x_min = i
    #             if i>x_max:
    #                 x_max = i
    #             if j < y_min:
    #                 y_min = j
    #             if j > y_max:
    #                 y_max = j
    # centerX = (x_max+x_min)/2
    # centerY = (y_max+y_min)/2

    for i in range(imgx):
        for j in range(imgy):
            if imgErode[i][j] == 0:
                allX = allX + i
                allY = allY + j
                allNum = allNum + 1

    centerX = allX / allNum
    centerY = allY / allNum

    #cv2.imwrite('imgErode.jpg', imgErode)
    return np.array([centerX, centerY])



def makeborder(img,border_width = 100):
    return cv2.copyMakeBorder(img, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT,
                              value=0)

def get_image_series(img):
    img_series = {}
    # no padding
    img_series['original'] = image_preprocess_display(img)

    # with padding
    img_series['original_padding'] = makeborder(img_series['original']).astype(np.uint8)
    img_series['mask'] = np.where(img_series['original_padding'] > 0, 1, 0).astype(np.uint8)
    img_series['importance'] = np.where(img_series['original_padding'] > 0, 1, 0).astype(np.uint8)
    img_series['vessel_mask'] = makeborder(vessel_extract_api(img_series['original'])).astype(np.uint8)
    # img_series['sk_vessel_mask'] = opencv2skimage(img_series['vessel_mask'])
    # blur_base = cv2.GaussianBlur(img_series['vessel_mask'], (15, 15), 0).astype(np.uint8)
    # blur_base[img_series['vessel_mask']>0]=255
    # img_series['vessel_mask_blur'] = blur_base
    return img_series

def get_aligned_image(img):
    yellow_spot = get_yellow_spot(img)
    optic_disk = get_optic_disk(img)

    img_display = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.circle(img_display,center=(int(yellow_spot[1]),int(yellow_spot[0])), radius=5,color=(0,255,255))
    cv2.circle(img_display, center=(int(optic_disk[1]),int(optic_disk[0])), radius=5,color=(255,0,0))

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Result', img_display)
    cv2.waitKey(1)
    if cv2.waitKey(0) & 0xff == ord('c'):
        cv2.waitKey(1)

    new_center = (yellow_spot+optic_disk)/2
    old_center = np.array([img.shape[0]/2,img.shape[1]/2])

    transfrom_move =  old_center - new_center

    M = np.float32([[1, 0, transfrom_move[0]], [0, 1, transfrom_move[1]]])  # 10
    shifted = cv2.warpAffine(img, M,img.shape[:2])  # 11
    cv2.imshow('Result',shifted)
    cv2.waitKey(1)
    if cv2.waitKey(0) & 0xff == ord('c'):
        cv2.waitKey(1)

    vec = optic_disk - yellow_spot
    distance = np.linalg.norm(vec)
    standard_dist=150
    ratio = standard_dist/distance
    angle = -np.arctan(vec[0]/vec[1])*180/np.pi

    M1 = cv2.getRotationMatrix2D( (img.shape[1]/2,img.shape[0]/2), angle, ratio)
    # 第三个参数：变换后的图像大小
    res1 = cv2.warpAffine(img, M1,img.shape[:2])

    cv2.imshow('Result',res1)
    cv2.waitKey(1)
    if cv2.waitKey(0) & 0xff == ord('c'):
        cv2.waitKey(1)
    return res1


img1 = cv2.imread('../Sidra_SHJTU/01/{6FD932E0-E140-4464-AFDE-4D0F0C2E4D3D}.jpg')
img1_series = get_image_series(img1)

img2 = cv2.imread('../Sidra_SHJTU/01/{A8A04672-8251-4A5E-B504-5DAEA352708F}.jpg')
img2_series = get_image_series(img2)

img1_aligned = get_aligned_image(img1_series['original_padding'])
img2_aligned = get_aligned_image(img2_series['original_padding'])

blend = (0.5* img1_aligned+0.5 *img2_aligned).astype(np.uint8)
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.imshow('Result', blend)
cv2.waitKey(1)
if cv2.waitKey(0) & 0xff == ord('c'):
    cv2.waitKey(1)