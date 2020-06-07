import numpy as np

def build_baseline_vectors_model(model_bifur_map):
    '''

    :param model_bifur_map: the bifurcation point map of model, include n points.
    :return: all possible baseline vectors. len = n/2
    '''
    return np.array([])

def get_template_triangle(baseline_vector,c):
    '''

    :param baseline_vector: vector AB
    :param c: point C chosen, should be optional parameter.
    :return:rotation factor z
    '''
    z = 0 #rotation factor
    return z

def get_similar_triangles(test_bifur_map,z):
    '''

    :param test_bifur_map: the bifurcation point map of test, include n points.
    :param z: rotation factor z of template triangle
    :return: array of similar triangles. Cn_2 * [[A'B'],C]
    '''
    return np.array([])

def cal_transform_param(vector_a,vector_b):
    '''

    :param vector_a: baseline vector of model
    :param vector_b: baseline vector of test
    :return: transformation params
    '''
    return np.array([0,0])

def cluster_and_cal_max_support(data):
    '''

    :param data: all transformation parameters
    :return: max support
    '''
    return 0.0

def triange_match(model_bifur_map,test_bifur_map,max_baseline_failure,min_param_support):
    model_baseline_vectors = build_baseline_vectors_model(model_bifur_map)
    baseline_match_failure = 0
    for baseline_vector in model_baseline_vectors:
        if baseline_match_failure>max_baseline_failure:
            return False

        #TODO:Choose a random C point here.
        C = [0,0]
        template_z = get_template_triangle(baseline_vector,C)
        similar_triangles = get_similar_triangles(test_bifur_map,template_z)

        all_transfrom_params = []
        for s in similar_triangles:
            #if s[0] represents vector A'B'
            all_transfrom_params.append(cal_transform_param(baseline_vector,s[0]))

        all_transfrom_params = np.array(all_transfrom_params)
        max_support = cluster_and_cal_max_support(all_transfrom_params)
        if max_support>min_param_support:
            return True
    return True