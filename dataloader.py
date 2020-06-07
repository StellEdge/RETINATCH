import csv
import numpy as np
'''
CRZ:For data loading
'''
def get_fundus_images_data(folder='',filename='regular-fundus-training.csv',indexlimit = 3):
    with open(folder+'/'+filename , newline='') as csvfile:
        datareader = csv.reader(csvfile)    #delimiter=' ', quotechar='|'
        labels = [row for row in datareader]
        data = labels[1:100] #limit for test
        labels = labels[0]
        '''standardlize types:'''
        # patient_id
        # image_id:str type
        # image_path:str type
        # Overall
        # quality
        # Artifact
        # Clarity
        # Field
        # definition
        # left_eye_DR_Level
        # right_eye_DR_Level
        # patient_DR_Level
        for i in range(len(data)):
            new_data = []
            for num,item in enumerate(data[i]):
                if num >= indexlimit:
                    break
                if (num > 2 or num < 1):
                    new_data.append(int(item))
                else:
                    new_data.append(item[1:])
            data[i] = new_data
        return labels[:indexlimit], data

import os
def get_fundus_images_data_Sidra(folder='Sidra_custom'):
    process_dir = folder
    subdir_list = os.listdir(process_dir)
    data = []
    for subdir in subdir_list:
        data.append(os.path.join(process_dir,subdir,'1.jpg'))
    return data