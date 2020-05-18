import cv2
import numpy as np
import matplotlib.pyplot as plt

#输入：灰度图
#输出：细化结果
def hilditch(img):
    h=img.shape[0]
    w=img.shape[1]
    out=np.zeros((h,w),dtype=np.int)

    delta_x=np.array([1,1,0,-1,-1,-1,0,1])
    delta_y=np.array([0,-1,-1,-1,0,1,1,1])

    for y in range(h):
        for x in range(w):
            if img[y,x]>=1:
                out[y,x]=1
    tmp=out.copy()
    tmp=cv2.copyMakeBorder(tmp,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
    count=1
    while count>0:
        count=0

        for y in range(1,h+1):
            for x in range(1,w+1):
                if tmp[y,x]<1:
                    continue
                judge=0
                b=np.zeros(9,dtype=np.int)
                for i in range(1,9):
                    if tmp[y+delta_y[i-1],x+delta_x[i-1]]==0:
                        b[i]=1
                    elif tmp[y+delta_y[i-1],x+delta_x[i-1]]==-1:
                        b[i]=-1

                if 1-abs(b[1])+1-abs(b[3])+1-abs(b[5])+1-abs(b[7])>0:
                    judge+=1

                s=0
                for i in range(1,9):
                    s+=abs(b[i])
                if s>=2:
                    judge+=1

                s=0
                for i in range(1,9):
                    if b[i]==1:
                        s+=1
                if s>=1:
                    judge+=1

                if func_nc8(b)==1:
                    judge+=1

                sum=0
                for i in range(1,9):
                    if b[i]!=-1:
                        sum+=1
                    else:
                        c=b[i]
                        b[i]=0
                        if func_nc8(b)==1:
                            sum+=1
                        b[i]=c
                if sum==8:
                    judge+=1

                if judge==5:
                    tmp[y,x]=-1
                    count+=1
        for y in range(1,h+1):
            for x in range(1,w+1):
                if tmp[y,x]==-1:
                    tmp[y,x]=0
    out=tmp[1:h+1,1:w+1]
    for y in range(h):
        for x in range(w):
            if out[y,x]<=0:
                out[y,x]=0
            if out[y,x]==1:
                out[y,x]=255
    return out            

#计算8-连接数
def func_nc8(b):
    n_odd=np.array([1,3,5,7])
    d=np.zeros(10)
    for i in range(10):
        j=i
        if i==9:
            j=1
        if abs(b[j])==1:
            d[i]=1
        else:
            d[i]=0
    
    sum=0
    for i in range(4):
        j=n_odd[i]
        sum=sum+d[j]-d[j]*d[j+1]*d[j+2]
    return sum


