import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from image_preprocessing import *
'''
CRZ:Main dish for retinatch
'''
from dataloader import get_fundus_images_data
data_saving_path = 'saving'
feature_saving_paths = {
    'kp':data_saving_path + '/extracted_features_kp.npy',
    'des':data_saving_path + '/extracted_features_des.npy'
}

data_folder = 'regular-fundus-training'

def init_saving_folders():
    if not (os.path.exists(data_saving_path)):
        os.mkdir(data_saving_path)

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





def extract_single_image_features(image_name):
    sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
    print('extracting', image_name)
    image = read_image_and_preprocess(image_name[1:])

    if image is None:
        return [],[]
    #b, g, r = cv2.split(image)  # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key_points, descriptor = sift.detectAndCompute(image, None)
    if len(key_points) <= 10:
        print('Key points are not enough. Skipped:', image_name)
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

def find_best_match_index(image_name,data,model_points,models,threshold = 30,verbose = False):
    kp1, des1 = extract_single_image_features(image_name)
    kp1 = [dict_to_keypoint(i) for i in kp1]
    if (len(kp1)==0 or len(des1)==0):
        print('feature extraction failed.')
        return -1

    #choose matcher:
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    best_index = -1
    best_match_distance = 1e10
    for index,model_des in enumerate(models):
        if len(model_des)==0:
            continue
        matches = flann.match(des1, model_des)
        # matches = bf.match(des1, model_des)
        score = 0
        matches.sort(key=lambda m:m.distance)
        for match in matches:
            score += match.distance
        avg_score = score / len(matches)
        if avg_score < threshold:
            if verbose:
                print("Fingerprint matches. Model index",index+1,'dist:',avg_score,'Name:',data[index][2])
            if avg_score <best_match_distance:
                best_index = index
                best_match_distance = avg_score

                kp2 = [dict_to_keypoint(i) for i in model_points[index]]
                img1 = read_image_and_preprocess(image_name[1:])
                img2 = read_image_and_preprocess(data[index][2][1:])
                matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
                cv2.putText(matching_result, image_name[1:]+'               '+data[index][2][1:],
                            (40, 50),
                            cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 255, 255), 2)
                cv2.putText(matching_result,'Best dist: '+str(best_match_distance), (40, 120),
                            cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 255, 255), 2)
                cv2.imshow('MatchResult', matching_result)
                cv2.waitKey(1)
                # if cv2.waitKey(0) & 0xff == ord('c'):
                #     continue
        else:
            if verbose:
                print("Fingerprint does not match. Model index",index+1,'dist:',avg_score,'Name:',data[index][2])

    return best_index, best_match_distance

def main():
    init_saving_folders()
    FORCE_RELOAD = True
    labels, data = get_fundus_images_data(data_folder)

    #find all l1 images
    new_data=[]
    for d in data:
        if d[1][-2:]=='l1':
            new_data.append(d)
    data = new_data

    if not (os.path.exists(feature_saving_paths['kp'])
            and os.path.exists(feature_saving_paths['des'] )) or FORCE_RELOAD:
        print('Save file not found, extracting features.')
        kp, des = extract_features([i[2] for i in data])
        np.save(feature_saving_paths['kp'], kp)
        np.save(feature_saving_paths['des'], des)
    else:
        kp = np.load(feature_saving_paths['kp'],allow_pickle=True)
        des = np.load(feature_saving_paths['des'],allow_pickle=True)
    print('features loaded.')

    print('total model number:',kp.shape[0])

    all_acc=[]
    all_fal=[]
    trs=range(100,251,25)
    cv2.namedWindow('MatchResult', cv2.WINDOW_NORMAL)
    for threshold in trs:
        matched = 0
        false_match = 0
        total = 4
        for t in range(1,1+total):
            match_res,best_dist = find_best_match_index('/'+data_folder+ '/'+str(t)+'/'+str(t)+'_l2.jpg',data,kp,des,threshold)
            if match_res ==-1:
                print('Match failed.')
            else:
                print('Matched',match_res+1,'name:',data[match_res][2])
                if data[match_res][0]==t:
                    matched+=1
                else:
                    false_match+=1
        acc = matched*1.0 / total
        wrong_match = false_match * 1.0 / total
        all_acc.append(acc)
        all_fal.append(wrong_match)
        print('Accuracy',acc)
        print('False match',wrong_match)
    plt.figure()
    plt.plot(trs,all_fal)
    plt.plot(trs,all_acc)
    plt.legend(['false match','Accuracy'],loc='best')
    plt.savefig('resultbf.png',format='png')
    plt.show()
if __name__ == '__main__':
    main()