import cv2
import numpy as np
from dishPosition import dishPosition
import os

process_dir = 'Sidra_custom'
subdir_list = os.listdir(process_dir)

for subdir in subdir_list:
    cur_dir = os.path.join(process_dir,subdir)
    imgname_list = os.listdir(cur_dir)
    imgdata_list = []
    for imgname in imgname_list:
        img_path = os.path.join(cur_dir,imgname)
        cur_img = cv2.imread(img_path)
        centerX, centerY = dishPosition(cur_img)
        img_centerY = int(cur_img.shape[0] / 2)
        img_centerX = int(cur_img.shape[1] / 2)
        center_distance = np.sqrt((centerX - img_centerX) ** 2 + (centerY - img_centerY) ** 2)
        imgdata_list.append([center_distance,cur_img])
    if len(imgdata_list)<2:
        print('Image is too few.')
        continue
    imgdata_list.sort(key = lambda x:x[0])
    cv2.imwrite(os.path.join(cur_dir,'1.jpg'),imgdata_list[0][1])
    cv2.imwrite(os.path.join(cur_dir, '2.jpg'),imgdata_list[1][1])
    print(cur_dir,'Complete.')
