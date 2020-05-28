
'''
Now let's talk about neural networks
'''
# import torch
# import torch.nn as nn
#
# cuda = torch.cuda.is_available()
#
# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
#
# #init
# # 0 0 0 cos ˆ α −sin ˆ α θ
# # 0 0 0 sin ˆ α cos ˆ α θ
# alpha = 1
# theta = 0.5
# t = Tensor([
#     [0,0,0,np.cos(alpha),-np.sin(alpha),theta],
#     [0,0,0,np.sin(alpha),np.cos(alpha),theta],
# ])
#
# origin_img = Tensor(sk_img1)
# target_img = Tensor(sk_img2)
#
# start_epoch = 0
# end_epoch = 100
# #for epoch in range(start_epoch, end_epoch):

import nevergrad as ng
import cv2
import numpy as np
from image_preprocessing import image_preprocess_display,vessel_extract_api
from skimage.transform import PolynomialTransform
from skimage import transform

class PolyTF(PolynomialTransform):
    def estimate(*data):
        return PolynomialTransform.estimate(*data, order=2)

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

border_width=200
img1 = cv2.imread('Sidra_SHJTU/01/{6FD932E0-E140-4464-AFDE-4D0F0C2E4D3D}.jpg')
img1 = image_preprocess_display(img1)
img1 = vessel_extract_api(img1)
img1 = cv2.copyMakeBorder(img1, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=0)

img2 = cv2.imread('Sidra_SHJTU/01/{A8A04672-8251-4A5E-B504-5DAEA352708F}.jpg')
img2 = image_preprocess_display(img2)
img2 = vessel_extract_api(img2)
img2 = cv2.copyMakeBorder(img2, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=0)
sk_img1 = opencv2skimage(img1)
sk_img2 = opencv2skimage(img2)

def square(x):
    return sum((x - .5) ** 2)

def registeration(w):
    model = PolyTF()
    model.params = w
    sk_img2_warped = transform.warp(sk_img2, model)
    res_map = np.where(sk_img2_warped+sk_img1 == 2)
    return -np.sum(res_map)

instrum = ng.p.Instrumentation(ng.p.Array(shape=(2,6),init=)) #, y=ng.p.Scalar()
optimizer = ng.optimizers.OnePlusOne(parametrization=instrum, budget=100)
recommendation = optimizer.minimize(registeration)  # best value
print(recommendation)
# for _ in range(optimizer.budget):
#     x = optimizer.ask()
#     value = square(x)
#     optimizer.tell(x, value)
# recommendation = optimizer.provide_recommendation()