import cv2
import numpy as np
from Hilditch import hilditch
from vesselExtract import vessel_extract_api
import queue
'''
CRZ:
For image preprocessing, image registration.'''

def get_minutiae_values(img):
    '''
    Get minutiae map for image
    :param img: a bi-valued image
    :return: map of crossing numbers
    '''
    cells = [(-1, -1),
             (-1, 0),
             (-1, 1),
             (0, 1),
             (1, 1),
             (1, 0),
             (1, -1),
             (0, -1),
             (-1, -1)]
    img = np.where(img > 0, 1, 0)  # Binarization
    pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    values = np.zeros(shape=(9, img.shape[0], img.shape[1]))
    for k in range(0, 9):
        values[k] = pad_img[1 + cells[k][0]:img.shape[0] + 1 + cells[k][0],
                    1 + cells[k][1]:img.shape[1] + 1 + cells[k][1]]
    crossings = np.zeros(shape=img.shape)
    for k in range(0, 8):
        crossings += np.abs(values[k] - values[k + 1])
    crossings /= 2
    return crossings

def smooth_image(img):
    img = img.astype(np.uint8)
    # cv2.namedWindow('SMOOTH', cv2.WINDOW_NORMAL)
    bi_img = np.where(img > 0, 1, 0)  # Binarization
    pad_img = cv2.copyMakeBorder(bi_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    for i in range(1, pad_img.shape[0]-1):
        for j in range(1, pad_img.shape[1]-1):
            if np.sum(pad_img[i-1:i+2,j-1:j+2])==1:
                #a noise
                pad_img[i,j]=0
    res_img = pad_img[1:bi_img.shape[0]+1,1:bi_img.shape[1]+1]*255

    minutiae_map = get_minutiae_values(res_img)
    bifur_map = np.where(np.logical_and(minutiae_map == 3, img > 0), 1, 0)
    ending_map = np.where(np.logical_and(minutiae_map == 1, img > 0), 1, 0)
    bifur_ending_map = 3 * bifur_map + ending_map


    for loop in range(2):
        for i in range(0, bifur_ending_map.shape[0]):
            for j in range(0, bifur_ending_map.shape[1]):
                if bifur_ending_map[i, j] == 1:
                    search_map = np.zeros_like(bifur_ending_map)
                    find_result = None
                    delete_list=[]
                    q = queue.Queue()
                    point=(i,j)
                    q.put((point, 0))
                    cells = [ (-1, 0),(1, 0), (0, -1), (0, 1),(-1, -1), (1, 1),(-1, 1),  (1, -1)]
                    target_point = None
                    while not q.empty():
                        p, dist = q.get()
                        search_map[p[0], p[1]] = 1
                        if bifur_ending_map[p[0], p[1]] > 0 and p != point:
                            #get a point
                            target_point = p
                            if bifur_ending_map[p[0], p[1]] == 3:
                                '''another bifurcation point is the nearest'''
                                if dist>4:
                                    #do not delete
                                    delete_list=[]
                                break
                            elif bifur_ending_map[p[0], p[1]] == 1:
                                '''another ending point is the nearest'''
                                delete_list.append(p)
                                break
                        else:
                            delete_list.append(p)
                            for c in cells:
                                s = (p[0] + c[0], p[1] + c[1])
                                if res_img[s[0], s[1]] > 0 and search_map[s[0], s[1]] == 0:
                                    search_map[s[0], s[1]] == 1
                                    q.put((s, dist + 1))
                    if len(delete_list)!=0:

                        # width = np.max([target_point[0]-point[0]+1,target_point[1]-point[1]+1,32])
                        # display_img =res_img[point[0]-width:point[0]+width,point[1]-width:point[1]+width].astype(np.uint8)
                        # display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                        # display_img[target_point[0]-point[0]+width ,target_point[1]-point[1]+width] = (255, 0, 0)
                        # display_img[width, width] = (0, 255, 0)
                        # cv2.imshow('SMOOTH', display_img.astype(np.uint8))
                        # cv2.waitKey(150)
                        # if cv2.waitKey(0) & 0xff == ord('c'):
                        #     cv2.waitKey(1)
                        for p in delete_list:
                            bifur_ending_map[p[0],p[1]]=0
                            res_img[p[0],p[1]]=0

                        # width = np.max([target_point[0]-point[0]+1,target_point[1]-point[1]+1,32])
                        # display_img =res_img[point[0]-width:point[0]+width,point[1]-width:point[1]+width].astype(np.uint8)
                        # display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                        # display_img[target_point[0]-point[0]+width ,target_point[1]-point[1]+width] = (255, 0, 0)
                        # display_img[width, width] = (0, 255, 0)
                        # cv2.imshow('SMOOTH', display_img.astype(np.uint8))
                        # cv2.waitKey(150)
                        # if cv2.waitKey(0) & 0xff == ord('c'):
                        #     cv2.waitKey(1)

                    # cv2.imshow('SMOOTH', res_img.astype(np.uint8))
                    # cv2.waitKey(1)
        '''
        minutiae_map = get_minutiae_values(res_img)
        bifur_map = np.where(np.logical_and(minutiae_map == 3, img > 0), 1, 0)
        ending_map = np.where(np.logical_and(minutiae_map == 1, img > 0), 1, 0)
        bifur_ending_map = 3 * bifur_map + ending_map
        '''
        # display_img = res_img.astype(np.uint8)
        # display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        # for i in range(0, display_img.shape[0]):
        #     for j in range(0, display_img.shape[1]):
        #         if bifur_ending_map[i, j] == 3:
        #             display_img[i,j]=(0, 255, 0)
        #             cv2.circle(display_img, (j, i), 3, (0, 255, 0), 1)
        #         elif bifur_ending_map[i, j] == 1:
        #             display_img[i, j] = (255,0, 0)
        #             cv2.circle(display_img, (j, i), 3, (255, 0, 0), 1)
        # cv2.imshow('SMOOTH',display_img.astype(np.uint8))
        # cv2.waitKey(1)
        # if cv2.waitKey(0) & 0xff == ord('c'):
        #     cv2.waitKey(1)

    return res_img.astype(np.uint8)

def get_mask(shape,radius = int(1650/2),padding=25):
    #pad_size=radius + padding
    height,width = shape[:2]
    center = (int(width/2),int(height/2))
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, center, radius, color=(255, 255, 255), thickness=-1)
    return mask

def crop_mask_image(img,padding=25):
    mask_path = 'trashbin/mask.jpg'
    #mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    diameter = 1650
    radius = int(1650/2)
    height,width  =img.shape[:2]
    center = (int(width/2),int(height/2))
    pad_size=radius + padding
    crop_img = img[center[1]-pad_size:center[1]+pad_size, center[0]-pad_size:center[0]+pad_size]
    mask = np.zeros(crop_img.shape, np.uint8)

    center = (radius+padding,radius+padding)
    cv2.circle(mask, center, radius, color=(255,255,255), thickness=-1)
    r_img=cv2.add(crop_img, np.zeros(np.shape(crop_img), dtype=np.uint8), mask=mask)

    return r_img

def image_preprocess(img):
    #split channels, grab green channel only.
    b, g, r = cv2.split(img)
    ratio = g.shape[0]/g.shape[1]
    img0 = cv2.resize(g, (int(1736*ratio),1736), interpolation=cv2.INTER_AREA)
    img0 = crop_mask_image(img0)
    img0 = cv2.resize(img0,(600,600),interpolation=cv2.INTER_AREA)
    img1 = vessel_extract_api(img0)
    img2 = hilditch(img1)
    img2 = smooth_image(img2)

    return img2

def image_preprocess_display(img):
    #split channels, grab green channel only.
    b, g, r = cv2.split(img)
    ratio = g.shape[0]/g.shape[1]
    img0 = cv2.resize(g, (int(1736*ratio),1736 ), interpolation=cv2.INTER_AREA)
    img0 = crop_mask_image(img0)
    img0 = cv2.resize(img0,(600,600),interpolation=cv2.INTER_AREA)
    return img0

def image_thinning(img):
    img1 = vessel_extract_api(img)
    img2 = hilditch(img1)
    img2 = smooth_image(img2)
    return img2

def read_image_and_preprocess(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image = image_preprocess(image)
    else:
        print('Error reading image :', image_path)
    return image

def retina_registration(imgs):
    '''TODO: registration for retina images.'''
    pass

if __name__ == '__main__':
    res = read_image_and_preprocess('regular-fundus-training/1/1_l1.jpg')
    cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
    cv2.imshow('Result',res)