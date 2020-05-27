import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from image_preprocessing import *
'''
CRZ:Main dish for retinatch
'''
from dataloader import get_fundus_images_data
from feature_extracting import extract_y_features,extract_y_feature
data_saving_path = 'saving'
feature_saving_paths = {
    'kp':data_saving_path + '/extracted_features_kp.npy',
    'des':data_saving_path + '/extracted_features_des.npy'
}

data_folder = 'regular-fundus-training'

def init_saving_folders():
    if not (os.path.exists(data_saving_path)):
        os.mkdir(data_saving_path)

def dict_to_keypoint(k):
    '''Since pickle cannot save <Keypoint>'''
    r = cv2.KeyPoint()
    r.pt= k['pt']
    r.size= k['size']
    r.angle= k['angle']
    r.octave = k['octave']
    r.class_id= k['class_id']
    return r

def find_best_match_index(image_name,data,model_points,models,threshold = 30,verbose = False):
    image = read_image_and_preprocess(image_name)
    kp1, des1 = extract_y_feature(image)
    kp1 = [dict_to_keypoint(i) for i in kp1]   #for KeyPoint
    if (len(kp1)==0 or len(des1)==0):
        print('feature extraction failed.')
        return -1

    #choose matcher:
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    # flann = cv2.FlannBasedMatcher(index_params, search_params)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    best_index = -1
    best_match_distance = 1e10
    for index,model_des in enumerate(models):
        if len(model_des)==0:
            continue
        # matches = flann.match(des1, model_des)
        matches = bf.match(des1, model_des)
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
                img1 = image
                # img2 = cv2.imread(data[index][2])
                # img2 = image_preprocess_display(img2)
                img2 = read_image_and_preprocess(data[index][2])
                matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
                cv2.putText(matching_result, image_name+'               '+data[index][2],
                            (10, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
                cv2.putText(matching_result,'Best dist: '+str(best_match_distance), (10, 50),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
                cv2.imshow('MatchResult', matching_result)
                cv2.waitKey(1)
                if cv2.waitKey(0) & 0xff == ord('c'):
                    continue
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
        kp, des = extract_y_features([i[2] for i in data])
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
            match_res,best_dist = find_best_match_index(data_folder+ '/'+str(t)+'/'+str(t)+'_l2.jpg',data,kp,des,threshold)
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