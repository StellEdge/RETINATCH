
'''
Corrupted
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
img1 = cv2.imread('../Sidra_SHJTU/01/{6FD932E0-E140-4464-AFDE-4D0F0C2E4D3D}.jpg')
img1 = image_preprocess_display(img1)
img1 = vessel_extract_api(img1)
img1 = cv2.copyMakeBorder(img1, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=0)

img2 = cv2.imread('../Sidra_SHJTU/01/{A8A04672-8251-4A5E-B504-5DAEA352708F}.jpg')
img2 = image_preprocess_display(img2)
img2 = vessel_extract_api(img2)
img2 = cv2.copyMakeBorder(img2, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=0)
sk_img1 = opencv2skimage(img1)
sk_img2 = opencv2skimage(img2)

import torch
import torch.nn as nn
import torchvision

model = torchvision.models.vgg19(pretrained=True)


class discriminator_VGG(nn.Module):
    def __init__(self, channel_in, channel_gain, input_size):
        super(discriminator_VGG, self).__init__()

        # first conv
        self.in_conv = [nn.Conv2d(channel_in, channel_gain, 3, 1, 1, bias=True),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(channel_gain, channel_gain, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(channel_gain, affine=True),
                        nn.LeakyReLU(0.2, True),
                        ]
        self.in_conv = nn.Sequential(*self.in_conv)
        cur_dim = channel_gain
        cur_size = input_size / 2
        self.conv_layers = []
        for i in range(3):
            self.conv_layers += self.build_conv_block(cur_dim)
            cur_dim *= 2
            cur_size /= 2
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.out_conv = [
            nn.Conv2d(cur_dim, cur_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cur_dim, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(cur_dim, cur_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(cur_dim, affine=True),
            nn.LeakyReLU(0.2, True),
        ]
        '''
        self.out_conv=[
            nn.Conv2d(cur_dim,1,3,1,1,bias=True),
            ]
        '''
        self.out_conv = nn.Sequential(*self.out_conv)

        cur_size /= 2
        cur_size = int(cur_size)
        self.linear = [nn.Linear(cur_dim * cur_size * cur_size, 100),
                       nn.LeakyReLU(0.2, True),
                       nn.Linear(100, 1)
                       ]
        self.linear = nn.Sequential(*self.linear)

    def build_conv_block(self, channel_gain):
        model = [
            nn.Conv2d(channel_gain, channel_gain * 2, 3, 1, 1, bias=True),
            nn.BatchNorm2d(channel_gain * 2, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channel_gain * 2, channel_gain * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(channel_gain * 2, affine=True),
            nn.LeakyReLU(0.2, True),
        ]
        return model

    def forward(self, x):
        x = self.in_conv(x)
        x = self.conv_layers(x)
        x = self.out_conv(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out