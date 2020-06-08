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
    for i in range(0, len(index), 2):
        shuffle_points.append([points[index[i]], points[index[i+1]]])

    return shuffle_points

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