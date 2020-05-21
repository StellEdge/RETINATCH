from __future__ import print_function

from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


def showImg(imgName, img, wsize=(400, 400)):
    cv2.namedWindow(imgName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(imgName, wsize[0], wsize[1])
    cv2.imshow(imgName, img)


def homofilter(I):
    I = np.double(I)
    m, n = I.shape
    rL = 0.5
    rH = 2
    c = 2
    d0 = 20
    I1 = np.log(I + 1)
    FI = np.fft.fft2(I1)
    n1 = np.floor(m / 2)
    n2 = np.floor(n / 2)
    D = np.zeros((m, n))
    H = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = ((i - n1) ** 2 + (j - n2) ** 2)
            H[i, j] = (rH - rL) * (np.exp(c * (-D[i, j] / (d0 ** 2)))) + rL
    I2 = np.fft.ifft2(H * FI)
    I3 = np.real(np.exp(I2) - 1)
    I4 = I3 - np.min(I3)
    I4 = I4 / np.max(I4) * 255
    dstImg = np.uint8(I4)
    return dstImg


def gaborfilter(srcImg):
    dstImg = np.zeros(srcImg.shape[0:2])
    filters = []
    ksize = [5, 7, 9, 11, 13]
    j = 0
    for K in range(len(ksize)):
        for i in range(12):
            theta = i * np.pi / 12 + np.pi / 24
            gaborkernel = cv2.getGaborKernel((ksize[K], ksize[K]), sigma=2 * np.pi, theta=theta, lambd=np.pi / 2,
                                             gamma=0.5)
            gaborkernel /= 1.5 * gaborkernel.sum()
            filters.append(gaborkernel)
    for kernel in filters:
        gaborImg = cv2.filter2D(srcImg, cv2.CV_8U, kernel)
        np.maximum(dstImg, gaborImg, dstImg)
    return np.uint8(dstImg)


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8U, kern, borderType=cv2.BORDER_REPLICATE)
        np.maximum(accum, fimg, accum)
    return accum


def process_threaded(img, filters, threadn=8):
    accum = np.zeros_like(img)

    def f(kern):
        return cv2.filter2D(img, cv2.CV_8U, kern)

    pool = ThreadPool(processes=threadn)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)
    return accum


###    Gabor特征提取
def getGabor(img, filters):
    res = []  # 滤波结果
    for i in range(len(filters)):
        res1 = process(img, filters[i])
        res.append(np.asarray(res1))

    pl.figure(2)
    for temp in range(len(res)):
        pl.subplot(4, 6, temp + 1)
        pl.imshow(res[temp], cmap='gray')
    pl.show()
    return res  # 返回滤波结果,结果为24幅图，按照gabor角度排列


def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        # kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern = cv2.getGaborKernel((ksize, ksize), 2 * np.pi, theta, 17.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def print_gabor(filters):
    for i in range(len(filters)):
        showImg(str(i), filters[i])


def reverse_image(img):
    antiImg = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            antiImg[i][j] = 255 - img[i][j]
    return antiImg


def pass_mask(mask, img):
    # qwe = reverse_image(img)
    qwe = img.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                qwe[i][j] = 0
    # asd = cv2.filter2D(qwe, cv2.CV_8U, mask)
    return qwe


def showKern(filters):
    for i in list(range(16)):
        kern = filters[i]
        kern = kern - np.min(kern)
        kern = kern / np.max(kern) * 255
        kern = np.clip(kern, 0, 255)
        kern = np.uint8(kern)
        plt.suptitle('Gabor matched filter kernel')
        plt.subplot(4, 4, i + 1), plt.imshow(kern, 'gray'), plt.axis('off'), plt.title('theta=' + str(i) + r'/pi')
    plt.show()


def calcDice(predict_img, groundtruth_img):
    predict = predict_img.copy()
    groundtruth = groundtruth_img.copy()
    predict[predict < 128] = 0
    predict[predict >= 128] = 1
    groundtruth[groundtruth < 128] = 0
    groundtruth[groundtruth >= 128] = 1
    predict_n = 1 - predict
    groundtruth_n = 1 - groundtruth
    TP = np.sum(predict * groundtruth)
    FP = np.sum(predict * groundtruth_n)
    TN = np.sum(predict_n * groundtruth_n)
    FN = np.sum(predict_n * groundtruth)
    # print(TP, FP, TN, FN)
    dice = 2 * np.sum(predict * groundtruth) / (np.sum(predict) + np.sum(groundtruth))
    return dice


def adjust_gamma(imgs, gamma=1.0):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            new_imgs[i, j] = cv2.LUT(np.array(imgs[i, j], dtype=np.uint8), table)
    return new_imgs


def build_filters2(sigma=1, YLength=10):
    filters = []
    widthOfTheKernel = np.ceil(np.sqrt((6 * np.ceil(sigma) + 1) ** 2 + YLength ** 2))
    if np.mod(widthOfTheKernel, 2) == 0:
        widthOfTheKernel = widthOfTheKernel + 1
    widthOfTheKernel = int(widthOfTheKernel)
    # print(widthOfTheKernel)
    for theta in np.arange(0, np.pi, np.pi / 16):
        # theta = np.pi/4
        matchFilterKernel = np.zeros((widthOfTheKernel, widthOfTheKernel), dtype=np.float)
        for x in range(widthOfTheKernel):
            for y in range(widthOfTheKernel):
                halfLength = (widthOfTheKernel - 1) / 2
                x_ = (x - halfLength) * np.cos(theta) + (y - halfLength) * np.sin(theta)
                y_ = -(x - halfLength) * np.sin(theta) + (y - halfLength) * np.cos(theta)
                if abs(x_) > 3 * np.ceil(sigma):
                    matchFilterKernel[x][y] = 0
                elif abs(y_) > (YLength - 1) / 2:
                    matchFilterKernel[x][y] = 0
                else:
                    matchFilterKernel[x][y] = -np.exp(-.5 * (x_ / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        m = 0.0
        for i in range(matchFilterKernel.shape[0]):
            for j in range(matchFilterKernel.shape[1]):
                if matchFilterKernel[i][j] < 0:
                    m = m + 1
        mean = np.sum(matchFilterKernel) / m
        for i in range(matchFilterKernel.shape[0]):
            for j in range(matchFilterKernel.shape[1]):
                if matchFilterKernel[i][j] < 0:
                    matchFilterKernel[i][j] = matchFilterKernel[i][j] - mean
        filters.append(matchFilterKernel)

    return filters


def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu) / sigma
    return x


def sigmoid(X):
    return 1.0 / (1 + np.exp(-float(X)))


def Normalize(data):
    k = np.zeros(data.shape, np.float)
    # k = np.zeros_like(data)
    # m = np.average(data)
    mx = np.max(data)
    mn = np.min(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            k[i][j] = (float(data[i][j]) - mn) / (mx - mn) * 255
    qwe = np.array(k, np.uint8)
    return qwe


def grayStretch(img, m=60.0 / 255, e=8.0):
    k = np.zeros(img.shape, np.float)
    ans = np.zeros(img.shape, np.float)
    mx = np.max(img)
    mn = np.min(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k[i][j] = (float(img[i][j]) - mn) / (mx - mn)
    eps = 0.01
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ans[i][j] = 1 / (1 + (m / (k[i][j] + eps)) ** e) * 255
    ans = np.array(ans, np.uint8)
    return ans


def vesselsExtract(picturePath, targetWidth):
    # 原图
    # srcImg = cv2.imread(path + ('%02d' % num) + '_test.tif', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    # print(srcImg.shape)
    oriSrcImg = cv2.imread(picturePath, cv2.IMREAD_COLOR)
    oriHeight, oriWidth, oriChannel = oriSrcImg.shape
    print(oriHeight, oriWidth, oriChannel)

    # 放缩
    scaleRate = float(targetWidth) / oriWidth
    targetHeight = int(scaleRate * oriHeight)
    srcImg = cv2.resize(oriSrcImg, (targetWidth, targetHeight), interpolation=cv2.INTER_AREA)
    print(srcImg.shape)

    # 标定图
    grayImg = cv2.split(srcImg)[1]

    # 提取掩膜
    ret0, th0 = cv2.threshold(grayImg, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(th0, np.ones((7, 7), np.uint8))
    # showImg("mask", mask)

    # 高斯滤波
    blurImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
    # cv2.imwrite("blurImg.png", blurImg)

    # HE
    heImg = cv2.equalizeHist(blurImg)
    # cv2.imwrite("heImg.png", heImg)

    # CLAHE 光均衡化+对比度增强
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    claheImg = clahe.apply(blurImg)
    # cv2.imwrite("claheImg.png", claheImg)

    # 同态滤波 光均衡化
    homoImg = homofilter(blurImg)

    preMFImg = adjust_gamma(claheImg, gamma=1.5)
    filters = build_filters2(sigma=1, YLength=10)
    # showKern(filters)
    gaussMFImg = process(preMFImg, filters)
    gaussMFImg_mask = pass_mask(mask, gaussMFImg)
    grayStretchImg = grayStretch(gaussMFImg_mask, m=30.0 / 255, e=8)

    # 二值化
    ret1, th1 = cv2.threshold(grayStretchImg, 30, 255, cv2.THRESH_OTSU)
    # th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, np.ones((3,3)))
    predictImg = th1.copy()
    cv2.imwrite("predictImg.png", predictImg)
    return predictImg

def vessel_extract_api(origrayImg):
    # 提取掩膜
    ret0, th0 = cv2.threshold(origrayImg, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(th0, np.ones((7, 7), np.uint8))

    # 高斯滤波
    blurImg = cv2.GaussianBlur(origrayImg, (5, 5), 0)

    # HE
    heImg = cv2.equalizeHist(blurImg)

    # CLAHE 光均衡化+对比度增强
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    claheImg = clahe.apply(blurImg)

    # 同态滤波 光均衡化
    homoImg = homofilter(blurImg)

    preMFImg = adjust_gamma(claheImg, gamma=1.5)
    filters = build_filters2(sigma=1, YLength=10)
    # showKern(filters)
    gaussMFImg = process(preMFImg, filters)
    gaussMFImg_mask = pass_mask(mask, gaussMFImg)
    grayStretchImg = grayStretch(gaussMFImg_mask, m=30.0 / 255, e=8)

    # 二值化
    ret1, th1 = cv2.threshold(grayStretchImg, 30, 255, cv2.THRESH_OTSU)
    # th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, np.ones((3,3)))
    predictImg = th1.copy()
    return predictImg

if __name__ == '__main__':
    vesselsExtract("test.jpg", 600)
