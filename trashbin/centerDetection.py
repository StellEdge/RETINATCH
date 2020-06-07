import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2 

binaryThreshold = 0.2

imgOrigin = cv2.imread("../../../Programs/Tencentdata/1030110910/FileRecv/eye2.png", cv2.IMREAD_COLOR)
imgx = imgOrigin.shape[0]
imgy = imgOrigin.shape[1]
if imgx > imgy:
	imgr = imgy / 2
else:
	imgr = imgx / 2
	
imgGray = imgOrigin[:, :, 1]
for i in range(imgx):
	for j in range(imgy):
		if imgGray[i][j] < 5:
			imgGray[i][j] = 255
imgGray = imgGray.astype(np.uint8)

cv2.imwrite('imgGray.jpg', imgGray)

clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(8, 8))
imgCLAHE = clahe.apply(imgGray)

cv2.imwrite('imgCLAHE.jpg', imgCLAHE)

minValue = np.min(imgCLAHE)
maxValue = np.max(imgCLAHE)
imgBinary = imgCLAHE > (binaryThreshold * (maxValue - minValue) + minValue)
imgBinary = imgBinary * maxValue
imgBinary = imgBinary.astype(np.uint8)

cv2.imwrite('imgBinary.jpg', imgBinary)

kernel = np.ones((8,8), np.uint8)
imgDilate = cv2.dilate(imgBinary, kernel)
imgErode = cv2.erode(imgDilate, kernel)

print(imgBinary)
print(imgErode)


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
print(centerX, centerY)

imgVessel = imgErode - imgBinary

cv2.imwrite('imgDilate.jpg', imgDilate)
cv2.imwrite('imgErode.jpg', imgErode)
cv2.imwrite('imgVessel.jpg', imgVessel)

# 黄斑