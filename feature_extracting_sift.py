import cv2
import numpy as np
from image_preprocessing import read_image_and_preprocess


def keypoint_to_dict(k):
    '''Since pickle cannot save <Keypoint>'''
    temp = {'pt': k.pt, 'size': k.size, 'angle': k.angle, 'octave': k.octave,
            'class_id': k.class_id}
    return temp

def dict_to_keypoint(k):
    '''Since pickle cannot save <Keypoint>'''
    r = cv2.KeyPoint()
    r.pt= k['pt']
    r.size= k['size']
    r.angle= k['angle']
    r.octave = k['octave']
    r.class_id= k['class_id']
    return r

def extract_single_image_features(image):
    sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
    #print('extracting', image_name)
    #image = read_image_and_preprocess(image_name)

    if image is None:
        return [],[]
    #b, g, r = cv2.split(image)  # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key_points, descriptor = sift.detectAndCompute(image, None)
    if len(key_points) <= 10:
        print('Key points are not enough. Skipped:')
        return [],[]

    key_points_dict = [keypoint_to_dict(i) for i in key_points]
    return key_points_dict,descriptor

def extract_features(image_names):
    sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
    key_points_for_all = []
    descriptor_for_all = []
    # colors_for_all = []
    cv2.namedWindow('Extracting', cv2.WINDOW_NORMAL)
    for image_name in image_names:
        print('extracting',image_name)
        image = read_image_and_preprocess(image_name[1:])
        cv2.imshow('Extracting', image)
        cv2.waitKey(1)
        # if cv2.waitKey(0) & 0xff == ord('c'):
        #     continue
        if image is None:
            continue
        key_points, descriptor = sift.detectAndCompute(image, None)

        if len(key_points) <= 10:
            print('Key points are not enough. Skipped:',image_name)
            key_points_for_all.append([])
            descriptor_for_all.append([])
            continue

        key_points_dict = [keypoint_to_dict(i) for i in key_points]
        key_points_for_all.append(key_points_dict)
        descriptor_for_all.append(descriptor)
        # colors = np.zeros((len(key_points), 3))
        # for i, key_point in enumerate(key_points):
        #     p = key_point.pt
        #     colors[i] = image[int(p[1])][int(p[0])]
        # colors_for_all.append(colors)
    return np.array(key_points_for_all), np.array(descriptor_for_all) #, np.array(colors_for_all)

img = cv2.imread('regular-fundus-training/1/1_l2.jpg')
b,g,r = cv2.split(img)
kp, res = extract_single_image_features(g)
kp = [dict_to_keypoint(i) for i in kp]
img = cv2.drawKeypoints(img, kp, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
cv2.imshow('Test',img)
if cv2.waitKey(0) & 0xff == ord('c'):
    cv2.destroyAllWindows()
    cv2.waitKey(1)
print('sd')