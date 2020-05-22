import cv2
import numpy as np
from sklearn.neighbors import kneighbors_graph
import queue

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

cells = [(-1, -1),
         (-1, 0),
         (-1, 1),
         (0, 1),
         (1, 1),
         (1, 0),
         (1, -1),
         (0, -1),
         (-1, -1)]


def get_minutiae_values(img):
    '''
    Get minutiae map for image
    :param img: a bi-valued image
    :return: map of crossing numbers
    '''
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


# test = np.array(
#     [
#         [1,0,0,1],
#         [0,1,0,1],
#         [0,0,1,0],
#         [0,1,1,0]
#     ]
# )
# res = get_minutiae_values(test)
# print(res)
def y_feature_conversion():
    '''TODO: get y_feature with rotation stablity.'''
    # connectivity = kneighbors_graph(
    #     X, n_neighbors=params['n_neighbors'], include_self=False)
    pass


def search_and_mark(img, point, bifur_ending_map, delete_map):
    search_map = np.zeros_like(bifur_ending_map)
    # search_map[point[0], point[1]] = 1
    find_result = []
    q = queue.Queue()
    q.put((point, 0))
    cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    while not q.empty():
        p, dist = q.get()
        search_map[p[0], p[1]] = 1
        '''TODO: fix this 8-neighber algorithm'''
        # flags = [True for c in cells]
        # for i,c in enumerate(cells):
        #     s = (p[0] + c[0], p[1] + c[1])
        #     if bifur_ending_map[s[0], s[1]] > 0:
        #         find_result.append([s, bifur_ending_map[s[0], s[1]]])
        #         #search_map[s[0], s[1]] = 1
        #         #if search_map[s[0], s[1]] == 0:
        #         for j in range(i-1,i+2):
        #             actual_index = j % 8
        #             flags[actual_index] = False
        # if len(find_result)>=3:
        #     break
        # for i,c in enumerate(cells):
        #     s = (p[0] + c[0], p[1] + c[1])
        #
        #     if flags[i] and img[s[0],s[1]]>0 and search_map[s[0], s[1]] ==0:
        #         q.put(s)
        #         search_map[s[0], s[1]] = 1

        if bifur_ending_map[p[0], p[1]] > 0 and p != point:
            find_result.append([p, bifur_ending_map[p[0], p[1]], dist])
            for c in cells:
                s = (p[0] + c[0], p[1] + c[1])
                search_map[s[0], s[1]] = 1
        else:
            for c in cells:
                s = (p[0] + c[0], p[1] + c[1])
                if img[s[0], s[1]] > 0 and search_map[s[0], s[1]] == 0:
                    q.put((s, dist + 1))
        if len(find_result) >= 3:
            break

    '''如果该分叉点有两个“相同”的连接分叉点（这两个连接分叉点实际上是同一个分叉点），
    则将它和与它连接的这两个“相同”的分叉点打上删除标记.'''
    dict = {}
    for k, v, d in find_result:
        if not k in dict:
            dict[k] = [1, [d]]
        else:
            dict[k][0] = dict[k][0] + 1
            dict[k][1].append(d)
    # flag = False
    if len(dict) != len(find_result):
        for k, v in dict.items():
            # if v[0] >1 and :
            '''TODO: remove error '''
            delete_map[k[0], k[1]] = 1
        delete_map[point[0], point[1]] = 1

    '''如果该分叉点有一个与之连接的末梢点且二者之间满足毛刺的结构特征，
    则将分叉点和末梢点打上删除标记'''
    flag = False
    for k, v, d in find_result:
        if v == 1 and np.sum((np.array(k) - np.array(point)) ** 2) < 8:
            flag = True
    if flag:
        delete_map[point[0], point[1]] = 1

    return


def bifurcation_points_filter(img, bifur_map, ending_map):
    '''
    如果该分叉点有两个“相同”的连接分叉点（这两个连接分叉点实际上是同一个分叉点），
    则将它和与它连接的这两个“相同”的分叉点打上删除标记.
    如果该分叉点有一个与之连接的末梢点且二者之间满足毛刺的结构特征，则将分叉点和末梢点打上删除标记

    如果该末梢点有一个与之直接连接的末梢点，则将这两个末梢点打上删除标记
    如果在半径R(R<2λ)内有另一个末梢点与之满足断纹的结构特征，则将这两个末梢点打上删除标记
    :param bifur_map:
    :param ending_map:
    :return:
    '''
    bifur_ending_map = 3 * bifur_map + ending_map
    delete_map = np.zeros_like(bifur_map)
    for i in range(0, bifur_map.shape[0]):
        for j in range(0, bifur_map.shape[1]):
            if bifur_map[i, j] != 0 and delete_map[i, j] == 0:
                #distance filter
                distance = np.linalg.norm(np.array([i-bifur_map.shape[0]/2,j-bifur_map.shape[1]/2]))
                if distance > 550 :
                    delete_map[i, j] = 1
                else:
                    search_and_mark(img, [i, j], bifur_ending_map, delete_map)
    bifur_map_new = np.where(bifur_map - delete_map > 0, 1, 0)
    return bifur_map_new


def extract_vectors( map, target_neighbor):
    '''
    search and convert?
    wanted vecter:
    (V[1],V[2],V[3],V[4])
    V[i]=(dist,angle) of i-th nearest point. Angle is based on the 1st nearest point.
    :param img:
    :param map:
    :param start_point:
    :param target_neighbor:
    :param dist_limit:
    :return:
    '''
    points = []
    for i in range(0, map.shape[0]):
        for j in range(0, map.shape[1]):
            if map[i, j] > 0:
                points.append([i, j])
    points = np.array(points)
    # find k nearest neighbors
    all_neighbors = kneighbors_graph(points, n_neighbors=target_neighbor, include_self=False).toarray()
    # cell_4 = [(1,0),(-1,0),(0,1),(0,-1)]
    extracted_vector = []
    kps = []
    for i in range(all_neighbors.shape[0]):
        neighbors_index = []
        for j in range(all_neighbors.shape[1]):
            if all_neighbors[i, j] != 0:
                neighbors_index.append(j)
        distances = [(np.sqrt(np.sum((points[j] - points[i]) ** 2)), j) for j in neighbors_index]
        distances.sort(key=lambda x: x[0])
        neighbors_index = [j[1] for j in distances]
        distances = [j[0] for j in distances]

        base_direction = points[neighbors_index[0]] - points[i]
        #base_model = np.sqrt(np.sum((base_direction) ** 2))
        base_model = np.linalg.norm(base_direction)
        #Cosine: a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cosine_angle = [points[j].dot(base_direction ) / (np.linalg.norm(points[j])*base_model) for j in neighbors_index[1:]]
        # cosine_angle = [np.sum(np.multiply((points[j] - points[i]), base_direction)) / (
        #             base_model * np.sqrt(np.sum((points[j] - points[i]) ** 2)))
        #                 for j in neighbors_index[1:]]
        cosine_angle = np.arccos(cosine_angle).tolist()
        # for k in range(3):
        #     print(neighbors_index[k + 1], distances[k + 1], points[neighbors_index[k + 1]] - points[i], cosine_angle[k])
        r = cv2.KeyPoint()
        r.pt = tuple(points[i])
        r.size = base_model
        r.angle = 0
        r.octave = 0
        r.class_id = 0
        kps.append(r)
        extracted_vector.append(distances + cosine_angle)
    return np.array(kps), np.array(extracted_vector,dtype=np.float32)


def extract_y_feature(img):
    '''
    extract bifurcation and ending points.
    :param img: Grayscale image
    :return:
    '''
    minutiae_map = get_minutiae_values(img)
    bifurcation_points = np.where(np.logical_and(minutiae_map == 3, img > 0), 1, 0)
    ending_points = np.where(np.logical_and(minutiae_map == 1, img > 0), 1, 0)

    bifurcation_points_new = bifurcation_points_filter(img, bifurcation_points, ending_points)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if bifurcation_points_new[i, j] > 0:
                cv2.circle(img, (j, i), 3, (0, 255, 0), 1)
            elif bifurcation_points[i, j] > 0:
                cv2.circle(img, (j, i), 3, (0, 0, 255), 1)
            elif ending_points[i, j] > 0:
                cv2.circle(img, (j, i), 3, (255, 0, 0), 1)
            else:
                pass
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Result', img)
    cv2.waitKey(1)
    if cv2.waitKey(0) & 0xff == ord('q'):
        # cv2.destroyAllWindows()
        cv2.waitKey(1)
    '''y_feature extraction plan 1:'''
    kp, res = extract_vectors(bifurcation_points_new, 4)
    kp = [keypoint_to_dict(i) for i in kp]
    return kp, res

def extract_y_features(image_names):
    key_points_for_all = []
    descriptor_for_all = []
    # colors_for_all = []
    cv2.namedWindow('Extracting', cv2.WINDOW_NORMAL)
    for image_name in image_names:
        print('extracting',image_name)
        image = read_image_and_preprocess(image_name)
        cv2.imshow('Extracting', image)
        cv2.waitKey(1)
        # if cv2.waitKey(0) & 0xff == ord('c'):
        #     continue
        if image is None:
            continue
        key_points, descriptor = extract_y_feature(image)

        if len(key_points) <= 10:
            print('Key points are not enough. Skipped:',image_name)
            key_points_for_all.append([])
            descriptor_for_all.append([])
            continue

        key_points_for_all.append(key_points)
        descriptor_for_all.append(descriptor)
    return np.array(key_points_for_all), np.array(descriptor_for_all) #, np.array(colors_for_all)


# img = cv2.imread('refine_image/250/250_l1.png', cv2.IMREAD_GRAYSCALE)
# kp, res = extract_y_feature(img)
