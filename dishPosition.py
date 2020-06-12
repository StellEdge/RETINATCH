import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2 

def dishPosition(imgOrigin,verbose = False):
	imgx = imgOrigin.shape[0]
	imgy = imgOrigin.shape[1]
	if imgx > imgy:
		imgr = imgy / 2
	else:
		imgr = imgx / 2

	if len(imgOrigin.shape) == 3:
		imgGray = imgOrigin[:, :, 1]
	else:
		imgGray = np.copy(imgOrigin)
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
	imgBinary = (imgCLAHE == maxValue)
	imgBinary = imgBinary * 255
	imgBinary = imgBinary.astype(np.uint8)

	# cv2.imwrite('imgBinary.jpg', imgBinary)

	kernel = np.ones((8,8), np.uint8)
	imgDilate = cv2.dilate(imgBinary, kernel)
	imgErode = cv2.erode(imgDilate, kernel)

	if verbose:
		print(imgBinary)
		print(imgErode)

	allNum = 0
	allX = 0
	allY = 0
	for i in range(imgx):
		for j in range(imgy):
			if imgErode[i][j] == 255:
				allY = allY + i
				allX = allX + j
				allNum = allNum + 1

	centerX = int(np.round(allX / allNum))
	centerY = int(np.round(allY / allNum))
	if verbose:
		print(centerX, centerY)

	# cv2.circle(imgOrigin, (int(centerX), int(centerY)), 10, (0, 255, 0), 4)
	# cv2.imwrite('imgPoint.jpg', imgOrigin)
	return centerX, centerY

