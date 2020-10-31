'''
    1. input_folder의 rgb 이미지 -> grayscale에 gray로 저장
    2. grayscale -> resize -> jpg, csv로 저장
'''

import numpy as np
import csv
import glob
import cv2
from PIL import Image

input_folder = sorted(glob.glob('/users/jiwon/documents/강의/2020-2/기계시스템설계2/sync/bag_results/*/'))
input_folder = input_folder[0:len(input_folder)]
output_folder = sorted(glob.glob('/users/jiwon/documents/강의/2020-2/기계시스템설계2/grayscale/*/'))

def txtWriter(single_img, save_name):
    img = cv2.imread(single_img, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    img2list = ''
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (i == 0) & (j == 0):
                img2list += str(img[i, j])
            else:
                img2list += str(' ' + str(img[i, j]))

    f = open(save_name, 'w')
    f.write(img2list)
    f.close()

def rgb2gray(img, save_name):
    img = Image.open(img).convert('L')
    img_numpy = np.array(img, 'uint8')
    cv2.imwrite(save_name, img_numpy)    

def resize(img, save_name):
    image = cv2.imread(img)      
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_CUBIC)
    img_numpy = np.array(image, 'uint8')
    cv2.imwrite(save_name, img_numpy)

# convert to gray and save as jpg
for path, save_path in zip(input_folder, output_folder):
    for i in range(100):
       image = path + 'rgb/' + str(i*10+1) + '_rgb.jpg'
       save_name = save_path + str(i*10+1) + '_gray.jpg'
       rgb2gray(image, save_name)

# convert to gray -> resize gray jpg into 96x96 -> save as jpg & csv
input_folder = sorted(glob.glob('/users/jiwon/documents/강의/2020-2/기계시스템설계2/grayscale/*/'))
for path in input_folder:
    for i in range(100):
        # resize
        image = path + str(i*10+1) + '_gray.jpg'
        save_name = path + 'resized_' + str(i*10+1) + '_gray.jpg'
        resize(image, save_name)
        # csv
        image = save_name
        txtWriter(image, save_name)


    
    


    