import csv
import numpy as np
'''
CRZ:For data loading
'''
def get_fundus_images_data(folder='',filename='regular-fundus-training.csv',indexlimit = 3):
    with open(folder+'/'+filename , newline='') as csvfile:
        datareader = csv.reader(csvfile)    #delimiter=' ', quotechar='|'
        labels = [row for row in datareader]
        data = labels[1:40] #limit for test
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
                    new_data.append(item)
            data[i] = new_data
        return labels[:indexlimit], data

#get_fundus_images_data()