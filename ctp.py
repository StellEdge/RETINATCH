import numpy as np

def cal_transform_param(vector_a,vector_b):
    '''

    :param vector_a: baseline vector of model
    :param vector_b: baseline vector of test
    :return: transformation params
    '''
    x1=vector_a[1][1]-vector_a[0][1]
    y1=vector_a[1][0]-vector_a[0][0]
    x2=vector_b[1][1]-vector_b[0][1]
    y2=vector_b[1][0]-vector_b[0][0]
    divisor=x1*x1+y1*y1
    a=1.0*(x1*x2+y1*y2)/divisor
    b=1.0*(x1*y2-x2*y1)/divisor
    return np.array([a,b])
