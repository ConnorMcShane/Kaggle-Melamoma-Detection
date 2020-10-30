import csv
import os
import numpy as np
import math

def data_sort(image_path, csv_path, val_ratio=0.2, training=True):
    
    all_image_paths = []
    all_image_labels = []
    all_image_info = []

    file = open(csv_path)
    csv_reader = csv.reader(file)
    next(csv_reader)
    site_dict = {'head/neck':0, 'upper extremity':1, 'lower extremity':2, 'torso':3, 'palms/soles':4, 'oral/genital':5}
    age_dict = {'0':0, '5':1, '10':2, '15':3, '20':4, '25':5, '30':6, '35':7, '40':8, '45':9, '50':10, '55':11, '60':12, '65':13, '70':14, '75':15, '80':16, '85':17, '90':18}
    mole_num_categories = [[1,3],[4,6],[7,9],[10,12],[13,15],[16,18],[19,21],[22,24],[25,27],[28,30],[31,33],[34,36],[37,39],[40,42],[43,45],[46,50],[51,55],[56,65],[66,75],[76,100],[100,1000]]
    for row in csv_reader:
        all_image_paths.append(os.path.join(image_path, str(row[0]) + '.jpg'))
        
        info_1 = 0
        info_2 = 0
        if row[2] == 'male':
            info_1 = 1
        
        if row[2] == 'female':
            info_2 = 1
        
        site_list = [0,0,0,0,0,0]
        age_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        number_of_moles_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i, rng in enumerate(mole_num_categories):
            if rng[0] <= int(row[5]) <= rng[1]:
                number_of_moles_list[i] = 1
                break
        try:
            site_list[site_dict[row[4]]] = 1
        except:
            exception = 1
        try:
            age_list[age_dict[row[3]]] = 1
        except:
            exception = 1

        all_image_info.append([info_1, info_2] + age_list + site_list + number_of_moles_list)

        if training == True:
            all_image_labels.append(bool(int(row[8])))
        else:
            all_image_labels.append([row[0], 0])

    if training == True:

        benign = []
        melanoma = []
        for i in range(len(all_image_labels)):
            if all_image_labels[i] == 0:
                benign.append(i)
            else:
                melanoma.append(i)

        if val_ratio == 0:
            
            train_idx = []
            for i in range(len(benign)):
                train_idx.append(benign[i])
                j = i - (math.trunc(i/len(melanoma))*len(melanoma))
                train_idx.append(melanoma[j])

            return all_image_paths, all_image_labels, all_image_info, train_idx

        else:
        
            dataset_len_b = len(benign)
            val_len_b = int(dataset_len_b*val_ratio)
            train_len_b = dataset_len_b - val_len_b
            train_idx_b = benign[val_len_b:dataset_len_b]
            val_idx_b = benign[0:val_len_b]

            dataset_len_m = len(melanoma)
            val_len_m = int(dataset_len_m*val_ratio)
            train_len_m = dataset_len_m - val_len_m
            train_idx_m = melanoma[val_len_m:dataset_len_m]
            val_idx_m = melanoma[0:val_len_m]

            ratio = int(len(train_idx_b)/len(train_idx_m))
            train_idx = []
            for i in range(len(train_idx_b)):
                train_idx.append(train_idx_b[i])
                j = i - (math.trunc(i/len(train_idx_m))*len(train_idx_m))
                train_idx.append(train_idx_m[j])

            ratio = int(len(val_idx_b)/len(val_idx_m))
            val_idx = []
            for i in range(len(val_idx_b)):
                val_idx.append(val_idx_b[i])
                j = i - (math.trunc(i/len(val_idx_m))*len(val_idx_m))
                val_idx.append(val_idx_m[j])

            return all_image_paths, all_image_labels, all_image_info, train_idx, val_idx
    else:
        train_idx = [i for i in range(len(all_image_paths))]
        return all_image_paths, all_image_labels, all_image_info, train_idx