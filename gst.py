import numpy as np


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
