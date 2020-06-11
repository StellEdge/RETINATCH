import random
import numpy as np
import math
import multiprocessing
import cv2


def build_baseline_vectors_model(model_bifur_map):
    '''

    :param model_bifur_map: the bifurcation point map of model, include n points.
    :return: all possible baseline vectors. len = n/2
    '''
    points = []
    h, w = model_bifur_map.shape
    for i in range(h):
        for j in range(w):
            if model_bifur_map[i][j] != 0:
                points.append([i, j])

    index = np.array(range(len(points)))
    random.shuffle(index)

    shuffle_points = []
    for i in range(0, len(index) - (len(index) % 2), 2):
        shuffle_points.append([points[index[i]], points[index[i + 1]]])

    return np.array(shuffle_points)


def get_template_triangle(baseline_vector, c):
    '''

    :param baseline_vector: vector AB
    :param c: point C chosen, should be optional parameter.
    :return:rotation factor z[r, theta] r is ratio, theta is radian
    '''
    A = baseline_vector[0]
    B = baseline_vector[1]
    C = c
    lenAB = math.sqrt((A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]))
    lenAC = math.sqrt((A[0] - C[0]) * (A[0] - C[0]) + (A[1] - C[1]) * (A[1] - C[1]))
    r = lenAB / lenAC

    cos = ((A[0] - B[0]) * (A[0] - C[0]) + (A[1] - B[1]) * (A[1] - C[1])) / (lenAB * lenAC)

    theta = math.acos(cos)

    z = [r, theta]
    return z


def get_similar_triangles(test_bifur_map, record, z, epsilon=20,
                          triangle_ignore_len=20, triangle_ignore_angle_cos=-0.5,verbose = False):
    '''

    :param test_bifur_map: the bifurcation point map of test, include n points.
    :param record: a array storing the bifurcation points of test, include n points.
    :param z: rotation factor z of template triangle. z=[r,theta]
    :param epsilon: the deviation of point C (int)
    :return: array of similar triangles. Cn_2 * [[A'B'],C]
    '''
    triangles = []
    tmp = record.copy()

    if verbose:
        cv2.namedWindow('similiar', cv2.WINDOW_AUTOSIZE)

        right_img = np.zeros(shape=(test_bifur_map.shape[0], test_bifur_map.shape[1], 3)).astype(np.uint8)
        for j in range(test_bifur_map.shape[0]):
            for k in range(test_bifur_map.shape[1]):
                if test_bifur_map[j, k] > 0:
                    cv2.circle(right_img, center=(j, k), radius=4, color=(255, 255, 0))

    for i in range(record.shape[0]):
        A = record[i]
        for j in range(i + 1, record.shape[0]):
            B = record[j]
            C = rotation(A, B, z)

            if verbose:
                new_right_img = right_img.copy()
                #for c in C:
                #    cv2.polylines(new_right_img, [np.array([A, B, c]).astype(np.int32)], True,
                #                  (0, 255, 255), 1)
                point_flag = False

            for c in C:
                x_min = int(max(0, c[1] - epsilon))
                x_max = int(min(test_bifur_map.shape[1], c[1] + epsilon + 1))
                y_min = int(max(0, c[0] - epsilon))
                y_max = int(min(test_bifur_map.shape[0], c[0] + epsilon + 1))
                if x_min >= test_bifur_map.shape[1] or y_min >= test_bifur_map.shape[0] or\
                        x_max < 0 or y_max < 0 or y_min > y_max or x_min > x_max:
                    continue
                tmp = test_bifur_map[y_min:y_max,x_min:x_max]
                s = np.sum(tmp)
                if s == 0:
                    continue
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        #add euclidian dist
                        if test_bifur_map[y][x] == 1 and (x-c[1])**2+(y-c[0])**2<=epsilon**2:
                            good_point = np.array([y, x])

                            if verbose:
                                point_flag = True
                                cv2.polylines(new_right_img, [np.array([A, B, good_point]).astype(np.int32)], True,
                                              (255, 255, 0), 1)

                            if not is_bad_triangle(A, B, good_point, triangle_ignore_len, triangle_ignore_angle_cos):
                                triangles.append([[A, B], good_point])
                            # don't limit them here?
                            # if not is_bad_triangle(A, B, C, triangle_ignore_len, triangle_ignore_angle_cos):
                            #     triangles.append([[A, B], C])
            if verbose:
                cv2.imshow('similiar', new_right_img)
                cv2.waitKey(1)
                if cv2.waitKey(0) & 0xff == ord('c'):   #point_flag and
                    cv2.waitKey(1)

    return np.array(triangles)


def rotation(A, B, z):
    '''

    A, B: two points (np.array)
    z: rotation factor z of template triangle (np.array)
    return: two points C1, C2 (np.array)
    '''
    C = []
    cos = np.cos(z[1])
    sin = np.sin(z[1])
    x = B[1] - A[1]
    y = B[0] - A[0]
    delta_x = z[0] * (x * cos - y * sin)
    delta_y = z[0] * (y * cos + x * sin)
    C1 = [A[0] + delta_y, A[1] + delta_x]
    C.append(C1)
    C2 = [B[0] - delta_y, B[1] - delta_x]
    C.append(C2)
    C = np.array(C)
    return C


def cal_transform_param(vector_a, vector_b):
    '''
    Calculate the vector transformation parameters of a to b.
    Implented with complex number.
    :param vector_a: baseline vector of model
    :param vector_b: baseline vector of test
    :return: transformation params
    '''
    vec_a = vector_a[1] - vector_a[0]
    vec_b = vector_b[1] - vector_b[0]
    complex_a = complex(vec_a[0], vec_a[1])
    complex_b = complex(vec_b[0], vec_b[1])
    result = complex_b / complex_a
    return [result.real, result.imag]


# def cluster_and_cal_max_support(data, min_support_rate, radius=50):
#     '''
#
#     :param data: all transformation parameters
#     :return: max support rate
#     '''
#
#     total_num = len(data)
#     max_support_count = total_num * min_support_rate + 1
#     multip = 10e2
#     size_a = size_b = int(2 * multip)
#     support_matrix = np.zeros(shape=(size_a, size_b)).astype(np.uint32)
#     for p in data:
#         new_p = [p[0] * multip, p[1] * multip]
#         if new_p[0] >= size_a or new_p[1] >= size_b:
#             continue
#         support_matrix[new_p[0], new_p[1]] += 1
#
#         # bfs here:
#         nearest_p = []
#         search_matrix = np.zeros_like(support_matrix)
#         cells = [[1, 0], [-1, 0], [0, 1], [0, -1]]
#         q = []
#         q.append([new_p, 0])
#         search_matrix[new_p[0], new_p[1]] = 1
#         while len(q) != 0:
#             cur_p, cur_dist = q.pop(0)
#             if cur_dist > 0 and support_matrix[cur_p[0], cur_p[1]] > 0:
#                 nearest_p = cur_p
#                 break
#
#             if cur_dist + 1 > radius:
#                 continue
#             for c in cells:
#                 next_p = [cur_p[0] + c[0], cur_p[1] + c[1]]
#                 if search_matrix[next_p[0], next_p[1]] == 0:
#                     # haven't searched p
#                     q.append([next_p, cur_dist + 1])
#                     search_matrix[next_p[0], next_p[1]] = 1
#
#         if len(nearest_p) != 0:
#             # we found a nearest point
#             d_i = support_matrix[new_p[0], new_p[1]]
#             support_matrix[new_p[0], new_p[1]] = 0
#             d_j = support_matrix[nearest_p[0], nearest_p[1]]
#             support_matrix[nearest_p[0], nearest_p[1]] = 0
#
#             d_k = d_i + d_j
#             if d_k >= max_support_count:
#                 # print('found max support')
#                 return min_support_rate + 0.01
#             else:
#                 new_a = (new_p[0] * d_i + nearest_p[0] * d_j) / d_k
#                 new_b = (new_p[1] * d_i + nearest_p[1] * d_j) / d_k
#                 support_matrix[new_a, new_b] = d_k
#
#     # cluster failed to found a parameter which has enough large support rate.
#     return 0.0


cur_total = 0
size_a = size_b = int(10)
support_matrix = np.zeros(shape=(size_a, size_b)).astype(np.uint32)


# import cv2
# cv2.namedWindow('CLUSTER', cv2.WINDOW_AUTOSIZE)
def dynamic_clustering(is_init, p=[0, 0], multip=100, radius=30, min_param_support=0.6, windowname='xxx'):
    global size_a, size_b, support_matrix, cur_total
    if is_init:
        cur_total = 0
        size_a = size_b = int(3 * multip)
        support_matrix = np.zeros(shape=(size_a, size_b)).astype(np.uint32)
        return False
    else:
        new_p = [int(p[0] * multip + 1.5 * multip), int(p[1] * multip + 1.5 * multip)]
        if new_p[0] >= size_a or new_p[1] >= size_b or new_p[0] < 0 or new_p[1] < 0:
            return False

        cur_total += 1
        support_matrix[new_p[0], new_p[1]] += 1

        cv2.imshow(windowname, (255 * support_matrix.copy() / support_matrix.max()).astype(np.uint8))
        cv2.waitKey(1)

        while True:
            # bfs here:
            nearest_p = []
            search_matrix = np.zeros_like(support_matrix)
            cells = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            q = []
            q.append([new_p, 0])
            search_matrix[new_p[0], new_p[1]] = 1
            while len(q) != 0:
                cur_p, cur_dist = q.pop(0)
                if cur_dist > 0 and support_matrix[cur_p[0], cur_p[1]] > 0:
                    nearest_p = cur_p
                    break

                if cur_dist + 1 > radius:
                    continue
                for c in cells:
                    next_p = [cur_p[0] + c[0], cur_p[1] + c[1]]
                    if 0 <= next_p[0] < 3 * multip and 0 <= next_p[1] < 3 * multip and \
                            search_matrix[next_p[0], next_p[1]] == 0:
                        # haven't searched p
                        q.append([next_p, cur_dist + 1])
                        search_matrix[next_p[0], next_p[1]] = 1

            if len(nearest_p) != 0:
                # we found a nearest point
                d_i = support_matrix[new_p[0], new_p[1]]
                support_matrix[new_p[0], new_p[1]] = 0
                d_j = support_matrix[nearest_p[0], nearest_p[1]]
                support_matrix[nearest_p[0], nearest_p[1]] = 0

                d_k = d_i + d_j
                if d_k / cur_total >= min_param_support and d_k > 40:
                    # print('found max support')
                    return True
                else:
                    new_a = int((new_p[0] * d_i + nearest_p[0] * d_j) / d_k)
                    new_b = int((new_p[1] * d_i + nearest_p[1] * d_j) / d_k)
                    support_matrix[new_a, new_b] = d_k
                    new_p = [new_a, new_b]
            else:
                return False
        return False


def map_to_points(map):
    '''
    convert map representation into list of points.
    :param map:
    :return:
    '''
    points = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j] > 0:
                points.append([i, j])
    return np.array(points)


def is_bad_triangle(A, B, C, triangle_ignore_len=20, triangle_ignore_angle_cos=-0.5):
    len_ab = np.linalg.norm(B - A)
    if len_ab < triangle_ignore_len:
        return True
    len_ac = np.linalg.norm(C - A)
    if len_ac < triangle_ignore_len:
        return True
    len_bc = np.linalg.norm(C - B)
    if len_bc < triangle_ignore_len:
        return True
    edge_list = [len_ab, len_ac, len_bc]
    edge_list.sort(reverse=True)
    max_angle_cos = (edge_list[1] ** 2 + edge_list[2] ** 2 - edge_list[0] ** 2) / (2 * edge_list[2] * edge_list[1])
    if max_angle_cos < triangle_ignore_angle_cos:
        return True
    return False


def verify_baseline(baseline_vector, model_bifur_points, test_bifur_map, test_bifur_points,
                    min_param_support, triangle_ignore_len, triangle_ignore_angle_cos):
    cv2.namedWindow('CLUSTER' + str(baseline_vector), cv2.WINDOW_AUTOSIZE)

    dynamic_clustering(True)

    np.random.shuffle(model_bifur_points)
    c_index = 0
    all_transfrom_params = []
    # traverse all possible C's
    while c_index < model_bifur_points.shape[0]:
        C = model_bifur_points[c_index]
        c_index += 1
        # do not choose A and B
        if (C == baseline_vector[0]).all() or (C == baseline_vector[1]).all():
            continue

        # filter bad triangle here
        if is_bad_triangle(baseline_vector[0], baseline_vector[1], C, triangle_ignore_len, triangle_ignore_angle_cos):
            continue

        template_z = get_template_triangle(baseline_vector, C)
        similar_triangles = get_similar_triangles(test_bifur_map, test_bifur_points, template_z, epsilon=20,
                                                  triangle_ignore_len=triangle_ignore_len - 2,
                                                  triangle_ignore_angle_cos=triangle_ignore_angle_cos - 0.05)
        # print('similar_triangles Extracted ', C)
        for s in similar_triangles:
            # if s[0] represents vector A'B'
            a_b = cal_transform_param(baseline_vector, s[0])
            if dynamic_clustering(False, a_b, min_param_support=min_param_support,
                                  windowname='CLUSTER' + str(baseline_vector)):
                print(baseline_vector, 'finish', True)
                return True

    print(baseline_vector, 'finish', False)
    return False


def triange_match(model_bifur_map, test_bifur_map, max_baseline_failure, min_param_support,
                  triangle_ignore_len, triangle_ignore_angle_cos,verbose = True):
    model_baseline_vectors = build_baseline_vectors_model(model_bifur_map)
    model_bifur_points = map_to_points(model_bifur_map)

    baseline_match_failure = 0
    print('ready to match')

    if verbose:
        cv2.namedWindow('triangles',cv2.WINDOW_AUTOSIZE)
        left_img = np.zeros(shape=(model_bifur_map.shape[0],model_bifur_map.shape[1],3)).astype(np.uint8)
        for j in range(model_bifur_map.shape[0]):
            for k in range(model_bifur_map.shape[1]):
                if model_bifur_map[j, k] > 0:
                    cv2.circle(left_img, center=(j, k), radius=4, color=(255, 255, 0))
        right_img = np.zeros(shape=(test_bifur_map.shape[0],test_bifur_map.shape[1],3)).astype(np.uint8)
        for j in range(test_bifur_map.shape[0]):
            for k in range(test_bifur_map.shape[1]):
                if test_bifur_map[j, k] > 0:
                    cv2.circle(right_img, center=(j, k), radius=4, color=(255, 255, 0))


    test_bifur_points = []
    for y in range(test_bifur_map.shape[0]):
        for x in range(test_bifur_map.shape[1]):
            if test_bifur_map[y, x] == 1:
                test_bifur_points.append([y, x])
    test_bifur_points = np.array(test_bifur_points)

    '''
    results=[]
    pool = multiprocessing.Pool(processes=4)
    for i in range(int(max_baseline_failure*model_baseline_vectors.shape[0])):
        results.append(pool.apply_async(verify_baseline,
                                    args=(model_baseline_vectors[i],model_bifur_points,test_bifur_map,test_bifur_points,
                    min_param_support,triangle_ignore_len, triangle_ignore_angle_cos)))

    pool.close()
    pool.join()

    res = np.array([i.get() for i in results]).any()
    return res
    '''

    for baseline_vector in model_baseline_vectors:
        if baseline_match_failure >= max_baseline_failure * model_baseline_vectors.shape[0]:
            return False

        # init clustering
        dynamic_clustering(True)

        np.random.shuffle(model_bifur_points)
        c_index = 0
        # traverse all possible C's
        while c_index < model_bifur_points.shape[0]:
            C = model_bifur_points[c_index]
            c_index += 1
            # do not choose A and B
            if (C == baseline_vector[0]).all() or (C == baseline_vector[1]).all():
                continue

            # filter bad triangle here
            if is_bad_triangle(baseline_vector[0], baseline_vector[1], C, triangle_ignore_len,
                               triangle_ignore_angle_cos):
                continue

            if verbose:
                new_left_img = left_img.copy()
                new_right_img = right_img.copy()
                cv2.polylines(new_left_img, [np.array([baseline_vector[0], baseline_vector[1], C])], True, (122, 255, 0), 1)
                combine_img = np.hstack((new_left_img, new_right_img))
                cv2.imshow('triangles',combine_img)
                cv2.waitKey(1)

            template_z = get_template_triangle(baseline_vector, C)
            similar_triangles = get_similar_triangles(test_bifur_map, test_bifur_points, template_z, epsilon=5,
                                                      triangle_ignore_len=triangle_ignore_len - 2,
                                                      triangle_ignore_angle_cos=triangle_ignore_angle_cos - 0.05)
            if verbose:
                print(similar_triangles.shape[0],'similar_triangles Extracted ', C)

                for s in similar_triangles:
                    cv2.polylines(new_right_img, [np.array([s[0][0], s[0][1], s[1].astype(np.int32)])], True, (0, 255, 255), 1)
                combine_img = np.hstack((new_left_img,new_right_img))
                cv2.imshow('triangles',combine_img)
                cv2.waitKey(1)
                if cv2.waitKey(0) & 0xff == ord('c'):
                    cv2.waitKey(1)


            for s in similar_triangles:
                # if s[0] represents vector A'B'
                a_b = cal_transform_param(baseline_vector, s[0])
                if dynamic_clustering(False, a_b, min_param_support=min_param_support):
                    return True

        baseline_match_failure += 1
        #print(baseline_match_failure, 'failed baseline')

    return False


if __name__ == '__main__':
    import time
    import cv2
    from image_preprocessing import image_preprocess_display, image_thinning, read_image_and_preprocess, \
        get_minutiae_values


    def extract_bifurcation(img):
        '''
        extract bifurcation and ending points.
        :param img: bi-valued image
        :return: a list of bifurcation points
        '''
        minutiae_map = get_minutiae_values(img)
        bifurcation_points = np.where(np.logical_and(minutiae_map == 3, img > 0), 1, 0)
        return bifurcation_points


    for i in range(1,5):
        model_image = read_image_and_preprocess('Sidra_custom/0'+str(i)+'/1.jpg')
        for j in range(1,2):
            test_image = read_image_and_preprocess('Sidra_custom/0'+str(j)+'/2.jpg')
            start=time.time()
            res = triange_match(extract_bifurcation(model_image), extract_bifurcation(test_image), 0.1, 0.6,triangle_ignore_len=20, triangle_ignore_angle_cos=-0.5,verbose=False)
            end=time.time()
            t=str(i)+' '+str(j)+' result:'+str(res)+',time cost:'+str(end-start)+'s'
            print(t)

    #model_image = read_image_and_preprocess('Sidra_custom/03/1.jpg')
    #test_image = read_image_and_preprocess('Sidra_custom/03/2.jpg')

    #res = triange_match(extract_bifurcation(model_image), extract_bifurcation(test_image), 0.2, 0.6,triangle_ignore_len=20, triangle_ignore_angle_cos=-0.5,verbose=False)
    #print(res)
