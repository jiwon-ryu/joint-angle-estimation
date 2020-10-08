import numpy as np
import json
import glob
import csv

def extract_json(file):
    with open(file) as j:
        data = json.load(j)

        n_points = 4  # mcp, pip, dip, tip
        xy_set = []
        for i in range(n_points):
            value = data['objects'][i]['points']['exterior'][0]
            for j in range(2):  # x and y
                xy_set.append(value[j])
    return xy_set

path = '/users/jiwon/documents/강의/2020-2/기계시스템설계2/dataset/pilot/json/+50/'
file_list = glob.glob(path+'*.json')
print('json N: ', len(file_list))

keypoints = []
for f in file_list:
    keypoints.append(extract_json(f))

keypoints = np.array(keypoints)
keypoints = np.reshape(keypoints, (-1, 8))
np.savetxt('keypoints.csv', keypoints, fmt='%s', delimiter=',')
