import random
import numpy as np
import math
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


def get_similar_triangles(test_bifur_map,z):
    '''

    :param test_bifur_map: the bifurcation point map of test, include n points.
    :param z: rotation factor z of template triangle. z=[r,theta]
    :return: array of similar triangles. Cn_2 * [[A'B'],C]
    '''
    record={}
    epsilon=0.5 #the deviation of point C
    min_length=2.0 #the min length permitted
    min_difference=2.0 #the min difference between a side and the sum of the other two sides
    triangles=[]
    for y in range(test_bifur_map.shape[0]):
        for x in range(test_bifur_map.shape[1]):
            if test_bifur_map[y][x]==1:
                record[(y,x)]=1

    tmp=record.copy()
    for A in record.keys():
        del tmp[A]
        for B in tmp.keys():
            C=rotation(test_bifur_map.shape,A,B,z,epsilon)
            for c in C:
                C_real=find_C(c,record,epsilon)
                for c_real in C_real:
                    if judge_triangle(A,B,c_real,min_length,min_difference):
                        A_array=[A[0],A[1]]
                        B_array=[B[0],B[1]]
                        triangles.append([[A_array,B_array],c_real])

    print(len(triangles))
    return triangles


def rotation(shape,A,B,z,epsilon):
    '''

    shape: the shape of image (np.array)
    A, B: two points (tuple)
    z: rotation factor z of template triangle (np.array)
    epsilon: the deviation of point C (double)
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
    if C1[0]>=-epsilon and C1[0]<shape[0]+epsilon and C1[1]>=-epsilon and C1[1]<shape[1]+epsilon:
        C.append(C1)
    C2=[B[0]-delta_y,B[1]-delta_x]
    if C2[0]>=-epsilon and C2[0]<shape[0]+epsilon and C2[1]>=-epsilon and C2[1]<shape[1]+epsilon:
        C.append(C2)
    return C

def find_C(C,record,epsilon):
    '''

    C: the target point (np.array)
    record: a dict storing bifurcation points (dict)
    epsilon: the deviation of point C (double)
    return: the real point C_real (np.array)
    '''
    C_real=[]
    for key in record.keys():
        distance=np.sqrt(pow(C[0]-key[0],2)+pow(C[1]-key[1],2))
        if distance<epsilon:
            C_real.append([key[0],key[1]])

    return C_real

def judge_triangle(A,B,C,min_length,min_difference):
    '''

    A,B: tuple
    C: np.array
    min_length: the min length permitted
    min_difference: the min difference between a side and the sum of the other two sides
    return: True if it's a triangle
    '''
    dAB=np.sqrt(pow(A[0]-B[0],2)+pow(A[1]-B[1],2))
    dAC=np.sqrt(pow(A[0]-C[0],2)+pow(A[1]-C[1],2))
    dBC=np.sqrt(pow(B[0]-C[0],2)+pow(B[1]-C[1],2))
    dmin=min(dAB,min(dAC,dBC))
    if dmin<min_length:
        return False
    if dAB+dAC-dBC<min_difference:
        return False
    if dAB+dBC-dAC<min_difference:
        return False
    if dBC+dAC-dAB<min_difference:
        return False


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

def cluster_and_cal_max_support(data):
    '''

    :param data: all transformation parameters
    :return: max support
    '''
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

def triange_match(model_bifur_map, test_bifur_map, max_baseline_failure, min_param_support,triangle_ignore_len = 20,triangle_ignore_angle_cos = -0.5 ):
    model_baseline_vectors = build_baseline_vectors_model(model_bifur_map)
    model_bifur_points = map_to_points(model_bifur_map)

    baseline_match_failure = 0
    print('ready to match')
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
            while (C == baseline_vector[0]).all() or (C == baseline_vector[1]).all():
                C = model_bifur_points[c_index]
                c_index += 1

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
            similar_triangles = get_similar_triangles(test_bifur_map, template_z)

            for s in similar_triangles:
                # if s[0] represents vector A'B'
                all_transfrom_params.append(cal_transform_param(baseline_vector, s[0]))

        all_transfrom_params = np.array(all_transfrom_params)
        max_support = cluster_and_cal_max_support(all_transfrom_params)
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