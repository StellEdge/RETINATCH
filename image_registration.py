from image_preprocessing import image_preprocess_display,vessel_extract_api
from skimage.transform import PolynomialTransform
from skimage import transform
from skimage.measure import ransac
import cv2
import numpy as np

def skimage2opencv(src):
    src = src*255
    src = src.astype(np.uint8)
    #cv2.cvtColor(src,cv2.COLOR_RGB2BGR)
    return src

def opencv2skimage(src):
    #cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    src = src.astype(np.float32)
    src = src / 255
    return src

class PolyTF(PolynomialTransform):
    def estimate(*data):
        return PolynomialTransform.estimate(*data, order=2)

img1 = cv2.imread('Sidra_SHJTU/01/{6FD932E0-E140-4464-AFDE-4D0F0C2E4D3D}.jpg')
img1 = image_preprocess_display(img1)
img1 = vessel_extract_api(img1)

img2 = cv2.imread('Sidra_SHJTU/01/{A8A04672-8251-4A5E-B504-5DAEA352708F}.jpg')
img2 = image_preprocess_display(img2)
img2 = vessel_extract_api(img2)

sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des1, des2)
matches.sort(key=lambda m:m.distance)



src = []
dst = []
center = np.array([img1.shape[0]/2,img1.shape[1]/2])
#new_match = []
for match in matches:
    #if np.linalg.norm(np.array(kp1[match.queryIdx].pt)-center)<250:
        #new_match.append(match)
    src.append(kp1[match.queryIdx].pt)
    dst.append(kp2[match.trainIdx].pt)
src = np.array(src)
dst = np.array(dst)

# matching_result = cv2.drawMatches(img1, kp1, img2, kp2, new_match, None, flags=2)
# cv2.namedWindow('MatchResult', cv2.WINDOW_NORMAL)
# cv2.imshow('MatchResult', matching_result)
# cv2.waitKey(1)
# if cv2.waitKey(0) & 0xff == ord('c'):
#     cv2.waitKey(1)

model, inliers = ransac((src, dst), PolyTF, min_samples=5,
                               residual_threshold=2, max_trials=1000)
#sk_img1=opencv2skimage(img1)
sk_img2=opencv2skimage(img2)

sk_img2_warped = transform.warp(sk_img2, model)

img2_warped = skimage2opencv(sk_img2_warped)

cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
blend = (img2_warped*0.5 + img1*0.5).astype(np.uint8)
cv2.imshow('Result',blend )
if cv2.waitKey(0) & 0xff == ord('c'):
    cv2.waitKey(1)

print('fit complete.')


'''
Now let's talk about 
'''