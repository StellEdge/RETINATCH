import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from image_preprocessing import *

'''
CRZ:Main dish for retinatch
'''
from dataloader import get_fundus_images_data,get_fundus_images_data_Sidra
#from feature_extracting import extract_y_features, extract_y_feature
#from cntBifurcation import extract_circle_feature_single, extract_circle_features
from feature_extract_triangle import  extract_bifur_feature,extract_bifur_features
from triangle_match import triange_match

data_saving_path = 'saving'
feature_saving_paths = {
    'kp': data_saving_path + '/extracted_features_kp.npy',
    'des': data_saving_path + '/extracted_features_des.npy',
    'test': data_saving_path + '/extracted_features_test.npy'
}

data_folder = 'regular-fundus-training'


def init_saving_folders():
    if not (os.path.exists(data_saving_path)):
        os.mkdir(data_saving_path)


def dict_to_keypoint(k):
    '''Since pickle cannot save <Keypoint>'''
    r = cv2.KeyPoint()
    r.pt = k['pt']
    r.size = k['size']
    r.angle = k['angle']
    r.octave = k['octave']
    r.class_id = k['class_id']
    return r


'''
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
'''


def vote_for_match_result(image_des, data, model_points, models, threshold=5, vote_threshold=0.6, verbose=False):
    des1 = image_des
    if len(image_des) == 0:
        return -1, 0
    # kp1 = [dict_to_keypoint(i) for i in kp1]   #for KeyPoint ignored now
    best_index = -1
    best_vote_rate = 0
    for index, model_des in enumerate(models):
        if len(model_des) == 0:
            # no descriptor for this image
            continue

        vote_weights = np.ones_like(des1)
        vote_weights[-1] = 2
        vote_weights[-2] = 1.5
        vote_weights[-3] = 1.5
        vote_res = np.zeros_like(des1)
        for pos, (a, b) in enumerate(zip(des1, model_des)):
            if np.abs(a - b) < threshold:
                vote_res[pos] = vote_weights[pos]
            else:
                vote_res[pos] = 0

        vote_rate = np.sum(vote_res) / np.sum(vote_weights)
        if vote_rate > vote_threshold:
            #if verbose:
            print("Fingerprint matches. Model index", index + 1, 'dist:', vote_rate, 'Name:', data[index])
            print('model:', model_des,'test:', des1)

            if vote_rate > best_vote_rate:
                best_index = index
                best_vote_rate = vote_rate

                # kp2 = [dict_to_keypoint(i) for i in model_points[index]]
                # img1 = image
                # # img2 = cv2.imread(data[index][2])
                # # img2 = image_preprocess_display(img2)
                # img2 = read_image_and_preprocess(data[index][2])
                # matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
                # cv2.putText(matching_result, image_name+'               '+data[index][2],
                #             (10, 20),
                #             cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
                # cv2.putText(matching_result,'Best dist: '+str(best_match_distance), (10, 50),
                #             cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
                # cv2.imshow('MatchResult', matching_result)
                # cv2.waitKey(1)
                # if cv2.waitKey(0) & 0xff == ord('c'):
                #     continue
        else:
            if verbose:
                print("Fingerprint does not match. Model index", index + 1, 'vote_rate:', vote_rate, 'Name:',
                      data[index][2])

    return best_index, best_vote_rate


def find_single_match(image_des, models, max_baseline_failure=6,min_param_support=0.5,triangle_ignore_len=20,triangle_ignore_angle_cos=-0.5,verbose=False):
    for index, model_des in enumerate(models):
        result = triange_match(model_des,image_des,max_baseline_failure,min_param_support,triangle_ignore_len,triangle_ignore_angle_cos)
        if result:
            return index,0
    return -1,0

def main():
    init_saving_folders()
    FORCE_RELOAD = False
    #labels, data = get_fundus_images_data(data_folder)
    data = get_fundus_images_data_Sidra()
    limiter = 10
    # find all l1 images
    # new_data = []
    # for d in data:
    #     if d[1][-2:] == 'l1':
    #         new_data.append(d)
    # data = new_data

    if not (os.path.exists(feature_saving_paths['kp'])
            and os.path.exists(feature_saving_paths['des'])) or FORCE_RELOAD:
        print('Save file not found, extracting features.')
        kp = []
        #des = extract_circle_features(data)
        des = extract_bifur_features(data)
        np.save(feature_saving_paths['kp'], kp)
        np.save(feature_saving_paths['des'], des)
    else:
        kp = np.load(feature_saving_paths['kp'], allow_pickle=True)
        des = np.load(feature_saving_paths['des'], allow_pickle=True)
    print('features loaded.')

    des = des[:limiter]
    print('total model number:', des.shape[0])


    total = 20
    # cv2.namedWindow('MatchResult', cv2.WINDOW_NORMAL)
    if not os.path.exists(feature_saving_paths['test']) or FORCE_RELOAD:
        test_des = []
        for t,p in zip(range(1, 1 + total),os.listdir('Sidra_custom')):
            # match_res,best_dist = find_best_match_index(data_folder+ '/'+str(t)+'/'+str(t)+'_l2.jpg',data,kp,des,threshold)
            d = extract_bifur_feature(os.path.join('Sidra_custom',p,'2.jpg'))
            test_des.append(d)

        test_des = np.array(test_des)
        test_des.dump(feature_saving_paths['test'])
    else:
        test_des = np.load(feature_saving_paths['test'], allow_pickle=True)

    test_des=test_des[:limiter]
    print('load complete')
    plt.figure()

    vote_trs = [0.5]
    trs = [0.1]
    for vote_threshold in vote_trs:
        all_acc = []
        all_fal = []
        for threshold in trs:
            matched = 0
            false_match = 0
            for t in range(1, 1 + total):
                # match_res,best_dist = find_best_match_index(data_folder+ '/'+str(t)+'/'+str(t)+'_l2.jpg',data,kp,des,threshold)
                #match_res, best_dist = vote_for_match_result(test_des[t - 1], data,
                                                             # kp, des, threshold=threshold,
                                                             # vote_threshold=vote_threshold)
                match_res ,best_dist = find_single_match(test_des[t - 1],des,max_baseline_failure=threshold,min_param_support=vote_threshold)
                if match_res == -1:
                    print('Match failed.')
                else:
                    print('Matched', match_res + 1, 'name:', data[match_res])
                    if data[match_res] == t:
                        matched += 1
                    else:
                        false_match += 1
            acc = matched * 1.0 / total
            wrong_match = false_match * 1.0 / total
            all_acc.append(acc)
            all_fal.append(wrong_match)
            print('vote_trs', vote_threshold, 'trs', threshold, 'Accuracy', acc, 'False match', wrong_match)

        # plt.plot(trs,all_fal,label='False acc '+str(vote_threshold))
        plt.plot(trs, all_acc, label='Acc ' + str(vote_threshold))
    plt.legend(loc='best')
    plt.savefig('result_triangle.png', format='png')
    plt.show()


if __name__ == '__main__':
    main()
