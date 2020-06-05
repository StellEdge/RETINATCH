from image_preprocessing import image_preprocess_display,vessel_extract_api,get_mask
from skimage.transform import PolynomialTransform
from skimage import transform
from skimage.measure import ransac
import cv2
import numpy as np
from concurrent import futures

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

border_width=100
def makeborder(img):
    return cv2.copyMakeBorder(img, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=0)

def get_image_series(img):
    img_series= {}
    #no padding
    img_series['original'] = image_preprocess_display(img)

    # with padding
    img_series['original_padding'] = makeborder(img_series['original']).astype(np.uint8)
    img_series['mask'] = np.where(img_series['original_padding'] > 0,1,0).astype(np.uint8)
    img_series['importance'] = np.where(img_series['original_padding'] > 0, 1, 0).astype(np.uint8)
    img_series['vessel_mask'] = makeborder(vessel_extract_api(img_series['original'])).astype(np.uint8)
    img_series['sk_vessel_mask'] = opencv2skimage(img_series['vessel_mask'])
    # blur_base = cv2.GaussianBlur(img_series['vessel_mask'], (15, 15), 0).astype(np.uint8)
    # blur_base[img_series['vessel_mask']>0]=255
    # img_series['vessel_mask_blur'] = blur_base
    return img_series

def get_image_downsampling_series(img_series,downscale,sigma):
    img_series_new = {}
    for k,v in img_series.items():
        img_series_new[k] = cv2.resize(v, (int(v.shape[0]/downscale) , int(v.shape[1]/downscale)), interpolation=cv2.INTER_AREA)
    for t in ['original_padding','vessel_mask','sk_vessel_mask']:
        img_series_new[t]=cv2.GaussianBlur(img_series_new[t], (sigma*2+1, sigma*2+1), 0).astype(np.uint8)
    return img_series_new


class PolyTF(PolynomialTransform):
    def estimate(*data):
        return PolynomialTransform.estimate(*data, order=2)


img1 = cv2.imread('../Sidra_SHJTU/01/{6FD932E0-E140-4464-AFDE-4D0F0C2E4D3D}.jpg')
img1_series = get_image_series(img1)
# series=[]
# for k in img1_series.values():
#     if k.shape[0]!=800:
#         continue
#     series.append(k)
# series = np.hstack(series)
# cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
# cv2.imshow('Result',series )
# cv2.waitKey(1)
# if cv2.waitKey(0) & 0xff == ord('c'):
#     cv2.waitKey(1)


img2 = cv2.imread('../Sidra_SHJTU/01/{A8A04672-8251-4A5E-B504-5DAEA352708F}.jpg')
img2_series = get_image_series(img2)

sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
kp1, des1 = sift.detectAndCompute(img1_series['original_padding'], None)
kp2, des2 = sift.detectAndCompute(img2_series['original_padding'], None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des1, des2)
matches.sort(key=lambda m:m.distance)

# matching_result = cv2.drawMatches(img1_series['original'], kp1,img2_series['original'], kp2, matches[:10], None, flags=2)

src = []
dst = []
new_match = []
#center = (img1_series['original'].shape[0]/2,img1_series['original'].shape[1]/2)
for match in matches[10:]:
    # if np.linalg.norm(np.array(kp1[match.queryIdx].pt)-center)<250:
    #     new_match.append(match)
    src.append(kp1[match.queryIdx].pt)
    dst.append(kp2[match.trainIdx].pt)
src = np.array(src)
dst = np.array(dst)

model, inliers = ransac((src, dst), PolyTF, min_samples=6,
                               residual_threshold=0.8, max_trials=4000)
sk_img1=img1_series['sk_vessel_mask']
sk_img2=img2_series['sk_vessel_mask']

sk_img2_warped = transform.warp(sk_img2, model)

img2_warped = skimage2opencv(sk_img2_warped)

print('fit complete.')


blend = (img2_warped*0.5 + img1_series['vessel_mask']*0.5).astype(np.uint8)
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
# cv2.imshow('Result',blend )
# cv2.waitKey(1)
# if cv2.waitKey(0) & 0xff == ord('c'):
#     cv2.waitKey(1)


'''
precisely 
'''

old_tran_param = model.params
import nevergrad as ng

fine_tuning_stage_masks = [
    np.array([
        [0,0,0,1,1,1],
        [0,0,0,1,1,1],
    ]),
    np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ]),
]


alpha = 0.5
'''
###
#try to get a better init, stage 1
stage_parameters = [
    (1,8,1),
    (1,8,1),
    (1,4,1),
    (1,2,0.1),
    (1,1,0.01),
]

img1_series_stage=[]
img2_series_stage=[]
for param in stage_parameters:
    img1_series_stage.append(get_image_downsampling_series(img1_series,param[0],param[1]))
    img2_series_stage.append(get_image_downsampling_series(img2_series,param[0],param[1]))

cur_stage=0
def registeration_stage1(w,a):
    m = PolyTF()
    m.params = np.array(
        [
            [0, 0, 0, np.cos(a), np.sin(a), w[0]],
            [0, 0, 0, np.sin(a), np.cos(a), w[1]],
        ]
    )
    sk_img2_warped = transform.warp(img2_series_stage[cur_stage]['sk_vessel_mask'], m)
    #sk_img2_mask_warped = transform.warp(img2_series_stage[cur_stage]['mask'], m)

    #cross = np.where(np.logical_and(img1_series['mask'] >0,sk_img2_mask_warped >0 ),1,0)
    #cross_area = max(np.sum(cross),1)

    res_map = (img1_series_stage[cur_stage]['sk_vessel_mask']) * ((sk_img2_warped-img1_series_stage[cur_stage]['sk_vessel_mask'])**2)

    img2_warped = skimage2opencv(sk_img2_warped)
    blend_2 = (img2_warped * 0.5 + img1_series_stage[cur_stage]['vessel_mask'] * 0.5).astype(np.uint8)
    cv2.imshow('Result', blend_2)
    cv2.waitKey(1)

    result = np.sum(res_map) #/ np.sum(cross_area)
    return result

# for epoch in range(4):
#     for stage_mask in fine_tuning_stage_masks:
#ng.families.DifferentialEvolution()
instrum = ng.p.Instrumentation(ng.p.Array(init=np.zeros(shape=(2,))).set_bounds(lower=-200,upper=200),a=ng.p.Scalar(init=0,lower=0,upper=2*np.math.pi) )
optimizer = ng.optimizers.PSO(parametrization=instrum, budget=200,num_workers= 8)
with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
    recommendation = optimizer.minimize(registeration_stage1)  # best value
print(recommendation)
'''

weight_edge = np.zeros_like(img2_series['mask']).astype(np.float32)

center = [weight_edge.shape[0]/2,weight_edge.shape[1]/2]
for i in range(weight_edge.shape[0]):
    for j in range(weight_edge.shape[1]):
        weight_edge[i,j]=20*(((i-center[0])**2+(j-center[1])**2)/(int(1650/2))**2)+1
#print(weight_edge)
####
def registeration(w):
    m = PolyTF()
    m.params = alpha * w + old_tran_param
    sk_img2_warped = transform.warp(sk_img2, m)
    sk_img2_mask_warped = transform.warp(img2_series['mask'], m)

    #cross = np.where(np.logical_and(img1_series['mask'] >0,sk_img2_mask_warped >0 ),1,0)
    #cross_area = max(np.sum(cross),1)

    res_map = weight_edge * (sk_img1) * ((sk_img2_warped-sk_img1)**2)

    img2_warped = skimage2opencv(sk_img2_warped)
    blend_2 = (img2_warped * 0.5 + img1_series['vessel_mask'] * 0.5).astype(np.uint8)
    cv2.imshow('Result', blend_2)
    cv2.waitKey(1)

    result = np.sum(res_map) #/ np.sum(cross_area)
    return result

# for epoch in range(4):
#     for stage_mask in fine_tuning_stage_masks:
#ng.families.DifferentialEvolution()
instrum = ng.p.Instrumentation(ng.p.Array(init=np.zeros_like(old_tran_param)))
optimizer = ng.optimizers.OnePlusOne(parametrization=instrum, budget=200,num_workers= 8)
with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
    recommendation = optimizer.minimize(registeration)  # best value
print(recommendation)

'''
####staged fitting
def registeration_stage_1(w):
    m = PolyTF()
    m.params = np.hstack([np.zeros_like(w),w])+old_tran_param
    sk_img2_warped = transform.warp(sk_img2, m)
    res_map = sk_img1 * ((sk_img2_warped-sk_img1)**2)

    img2_warped = skimage2opencv(sk_img2_warped)
    blend_2 = (img2_warped * 0.5 + img1 * 0.5).astype(np.uint8)
    cv2.imshow('Result', blend_2)
    cv2.waitKey(1)
    return np.sum(res_map)

instrum = ng.p.Instrumentation(ng.p.Array(init=np.zeros(shape=(2,3)))) #, y=ng.p.Scalar()
optimizer = ng.optimizers.OnePlusOne(parametrization=instrum, budget=200,num_workers= 8)
with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
    recommendation = optimizer.minimize(registeration_stage_1)  # best value

result_mid = recommendation.args[0]
result_mid = np.hstack([np.zeros_like(result_mid),result_mid])

def registeration_stage_2(w):
    m = PolyTF()
    m.params = w + old_tran_param
    sk_img2_warped = transform.warp(sk_img2, m)
    res_map = sk_img1 * ((sk_img2_warped-sk_img1)**2)

    img2_warped = skimage2opencv(sk_img2_warped)
    blend_2 = (img2_warped * 0.5 + img1 * 0.5).astype(np.uint8)
    cv2.imshow('Result', blend_2)
    cv2.waitKey(1)
    return np.sum(res_map)

instrum = ng.p.Instrumentation(ng.p.Array(init=result_mid)) #, y=ng.p.Scalar()
optimizer = ng.optimizers.OnePlusOne(parametrization=instrum, budget=200,num_workers= 8)
with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
    recommendation = optimizer.minimize(registeration_stage_2)  # best value
'''

result = alpha * recommendation.args[0] + old_tran_param
model_new = PolyTF()
model_new.params = result

sk_img2_warped = transform.warp(sk_img2, model_new)

img2_warped = skimage2opencv(sk_img2_warped)



blend_2 = (img2_warped*0.5 + img1_series['vessel_mask']*0.5).astype(np.uint8)
result = np.hstack([blend,blend_2,img2_warped]).astype(np.uint8)
cv2.imshow('Result',result )

print('fit complete.')
cv2.waitKey(1)
if cv2.waitKey(0) & 0xff == ord('c'):
    cv2.waitKey(1)
