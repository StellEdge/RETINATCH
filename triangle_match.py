import random
import numpy as np
import math
import queue
def build_baseline_vectors_model(model_bifur_map):
    '''

    :param model_bifur_map: the bifurcation point map of model, include n points.
    :return: all possible baseline vectors. len = n/2
    '''
    points = []
    h,w = model_bifur_map.shape
    for i in range(h):
        for j in range(w):
            if model_bifur_map[i][j] != 0:
                points.append([i, j])

    index = np.array(range(len(points)))
    random.shuffle(index)

    shuffle_points = []
    for i in range(0, len(index)-(len(index)%2), 2):
        shuffle_points.append([points[index[i]], points[index[i+1]]])

    return np.array(shuffle_points)

def get_template_triangle(baseline_vector,c):
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


def get_similar_triangles(test_bifur_map,record,z,epsilon=2,triangle_ignore_len = 20,triangle_ignore_angle_cos = -0.5):
    '''

    :param test_bifur_map: the bifurcation point map of test, include n points.
    :param record: a array storing the bifurcation points of test, include n points.
    :param z: rotation factor z of template triangle. z=[r,theta]
    :param epsilon: the deviation of point C (int)
    :return: array of similar triangles. Cn_2 * [[A'B'],C]
    '''
    triangles=[]
    tmp=record.copy()
    for i in range(record.shape[0]):
        A=record[i]
        for j in range(i+1,record.shape[0]):
            B=record[j]
            C=rotation(A,B,z)
            for c in C:
                x_min=int(max(0,c[1]-epsilon))
                x_max=int(min(test_bifur_map.shape[1],c[1]+epsilon+1))
                y_min=int(max(0,c[0]-epsilon))
                y_max=int(min(test_bifur_map.shape[0],c[0]+epsilon+1))
                if x_min>=test_bifur_map.shape[1] or y_min>=test_bifur_map.shape[0] or x_max<0 or y_max<0 or y_min>y_max or x_min>x_max:
                    continue
                tmp=test_bifur_map[y_min:y_max]
                tmp=tmp[x_min:x_max]
                s=np.sum(tmp)
                if s==0:
                    continue
                C_real=[]
                for y in range(y_min,y_max):
                    for x in range(x_min,x_max):
                        if test_bifur_map[y][x]==1:
                            C=np.array([y,x])
                            if not is_bad_triangle(A,B,C,triangle_ignore_len,triangle_ignore_angle_cos):
                                triangles.append([[A,B],C])

    return np.array(triangles)


def rotation(A,B,z):
    '''

    A, B: two points (np.array)
    z: rotation factor z of template triangle (np.array)
    return: two points C1, C2 (np.array)
    '''
    C=[]
    cos=np.cos(z[1])
    sin=np.sin(z[1])
    x=B[1]-A[1]
    y=B[0]-A[0]
    delta_x=z[0]*(x*cos-y*sin)
    delta_y=z[0]*(y*cos+x*sin)
    C1=[A[0]+delta_y,A[1]+delta_x]
    C.append(C1)
    C2=[B[0]-delta_y,B[1]-delta_x]
    C.append(C2)
    C=np.array(C)
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
    result = complex_b/complex_a
    return [result.real, result.imag]

#cal_transform_param(np.array([[1,1],[2,2]]),np.array([[0,0],[0,1]]))

def cluster_and_cal_max_support(data,min_support_rate,radius = 50):
    '''

    :param data: all transformation parameters
    :return: max support rate
    '''


    total_num = len(data)
    max_support_count = total_num * min_support_rate+1
    multip = 10e2
    size_a = size_b = 2*multip
    support_matrix = np.zeros(shape=(size_a,size_b)).astype(np.uint32)
    for p in data:
        new_p = [p[0]*multip ,p[1]*multip ]
        if new_p[0]>=size_a or new_p[1]>=size_b:
            continue
        support_matrix[new_p[0],new_p[1]]+=1

        #bfs here:
        nearest_p = []
        search_matrix = np.zeros_like(support_matrix)
        cells = [[1,0],[-1,0],[0,1],[0,-1]]
        q = []
        q.append([new_p,0])
        search_matrix[new_p[0], new_p[1]] = 1
        while len(q)!=0:
            cur_p,cur_dist = q.pop(0)
            if cur_dist>0 and support_matrix[cur_p[0],cur_p[1]]>0:
                nearest_p = cur_p
                break

            if cur_dist+1>radius:
                continue
            for c in cells:
                next_p = [cur_p[0]+c[0],cur_p[1]+c[1]]
                if search_matrix[next_p[0],next_p[1]]==0:
                    #haven't searched p
                    q.append([next_p,cur_dist+1])
                    search_matrix[next_p[0], next_p[1]] = 1

        if len(nearest_p)!=0:
            #we found a nearest point
            d_i = support_matrix[new_p[0],new_p[1]]
            support_matrix[new_p[0], new_p[1]]=0
            d_j = support_matrix[nearest_p[0],nearest_p[1]]
            support_matrix[nearest_p[0], nearest_p[1]]=0

            d_k = d_i+d_j
            if d_k >=max_support_count:
                #print('found max support')
                return min_support_rate+0.01
            else:
                new_a = (new_p[0] *d_i +nearest_p[0]*d_j)/d_k
                new_b = (new_p[1] * d_i + nearest_p[1] * d_j)/d_k
                support_matrix[new_a,new_b] = d_k

    #cluster failed to found a parameter which has enough large support rate.
    return 0.0


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

def is_bad_triangle(A,B,C,triangle_ignore_len = 20,triangle_ignore_angle_cos = -0.5 ):
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

def triange_match(model_bifur_map, test_bifur_map, max_baseline_failure, min_param_support=0.5,triangle_ignore_len = 20,triangle_ignore_angle_cos = -0.5 ):
    model_baseline_vectors = build_baseline_vectors_model(model_bifur_map)
    model_bifur_points = map_to_points(model_bifur_map)


    baseline_match_failure = 0
    print('ready to match')

    record=[]
    for y in range(test_bifur_map.shape[0]):
        for x in range(test_bifur_map.shape[1]):
            if test_bifur_map[y][x]==1:
                record.append([y,x])
    record=np.array(record)

    for baseline_vector in model_baseline_vectors:
        if baseline_match_failure > max_baseline_failure:
            return False


        np.random.shuffle(model_bifur_points)
        c_index = 0
        all_transfrom_params = []
        #traverse all possible C's
        while c_index<model_bifur_points.shape[0]:
            C = model_bifur_points[c_index]
            c_index += 1
            # do not choose A and B
            if (C == baseline_vector[0]).all() or (C == baseline_vector[1]).all():
                continue

            # filter bad triangle here
            if is_bad_triangle(baseline_vector[0],baseline_vector[1],C,triangle_ignore_len,triangle_ignore_angle_cos):
                continue
            # len_ab = np.linalg.norm(baseline_vector[1]-baseline_vector[0])
            # if len_ab<triangle_ignore_len:
            #     continue
            # len_ac = np.linalg.norm(C - baseline_vector[0])
            # if len_ac<triangle_ignore_len:
            #     continue
            # len_bc = np.linalg.norm(C - baseline_vector[1])
            # if len_bc<triangle_ignore_len:
            #     continue
            # edge_list = [len_ab,len_ac,len_bc]
            # edge_list.sort(reverse=True)
            # max_angle_cos = (edge_list[1]**2+edge_list[2]**2-edge_list[0]**2)/(2*edge_list[2]*edge_list[1])
            # if max_angle_cos <triangle_ignore_angle_cos:
            #     continue


            template_z = get_template_triangle(baseline_vector, C)
            similar_triangles = get_similar_triangles(test_bifur_map,record, template_z,triangle_ignore_len=triangle_ignore_len-2,triangle_ignore_angle_cos=triangle_ignore_angle_cos-0.05)

            for s in similar_triangles:
                # if s[0] represents vector A'B'
                all_transfrom_params.append(cal_transform_param(baseline_vector, s[0]))

        #all_transfrom_params = np.array(all_transfrom_params)
        max_support = cluster_and_cal_max_support(all_transfrom_params,min_param_support)
        if max_support > min_param_support:
            return True
    return True



if __name__ == '__main__':
    import cv2
    from image_preprocessing import image_preprocess_display,image_thinning,read_image_and_preprocess,get_minutiae_values


    def extract_bifurcation(img):
        '''
        extract bifurcation and ending points.
        :param img: bi-valued image
        :return: a list of bifurcation points
        '''
        minutiae_map = get_minutiae_values(img)
        bifurcation_points = np.where(np.logical_and(minutiae_map == 3, img > 0), 1, 0)
        return bifurcation_points

    model_image = read_image_and_preprocess('Sidra_SHJTU/01/{A8A04672-8251-4A5E-B504-5DAEA352708F}.jpg')
    test_image = read_image_and_preprocess('Sidra_SHJTU/01/{A8A04672-8251-4A5E-B504-5DAEA352708F}.jpg')

    res = triange_match(extract_bifurcation(model_image),extract_bifurcation(test_image),6,0.6)
    print(res)