import numpy as np
'''
    1. input_folder의 rgb 이미지 -> grayscale에 gray로 저장
    2. grayscale -> resize -> jpg, txt로 저장
'''

import csv
import glob
import cv2
from PIL import Image

input_folder = sorted(glob.glob('/users/jiwon/documents/강의/2020-2/기계시스템설계2/sync/bag_results/*/'))
input_folder = input_folder[0:len(input_folder)]
output_folder = sorted(glob.glob('/users/jiwon/documents/강의/2020-2/기계시스템설계2/grayscale/*/'))

def txtWriter(single_img, save_name):
    save_name1 = save_name + '.txt'
    save_name2 = save_name + '2.txt'
    
    img = cv2.imread(single_img, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    img2list = ''
    n_digits = 0
    stop_i = 0
    stop_j = 0
    breaker = False
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (i == 0) & (j == 0): # start: no ' '
                img2list += str(img[i, j])
                n_digits = len(img2list)            
            else: # others: with ' '
                if n_digits < 32767:
                    img2list += str(' ' + str(img[i, j]))
                    n_digits = len(img2list)
                    # if finished without overflowing -> save txt
                    if (i == img.shape[0]-1) & (j == img.shape[1]-1):
                        f = open(save_name1, 'w')
                        f.write(img2list)
                        f.close()
                else: # if overflowed
                    # save current data
                    f1 = open(save_name1, 'w')
                    f1.write(img2list)
                    f1.close()
                    # record stop point
                    stop_i = i
                    stop_j = j

                    # write the rest in a new file
                    if img2list[-1] == ' ':
                        img2list = ''
                    else:
                        img2list= ' '
                    for m in np.arange(stop_i, img.shape[0]):
                        for n in np.arange(stop_j, img.shape[1]):
                            if (m == stop_i) & (n == stop_j):
                                img2list += str(img[m, n])    
                            else:
                                img2list += str(' ' + str(img[m, n]))                                               
                                f2 = open(save_name2, 'w')
                                f2.write(img2list)
                                f2.close()
                    breaker = True
                    break
        if breaker == True:
            break

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
  
# convert to gray -> resize gray jpg into 96x96 -> save as jpg & txt
input_folder = sorted(glob.glob('/users/jiwon/documents/강의/2020-2/기계시스템설계2/grayscale/*/'))
for path in input_folder:
    for i in range(100):
        # resize
        image = path + str(i*10+1) + '_gray.jpg'
        save_name = path + 'resized_' + str(i*10+1) + '_gray.jpg'
        resize(image, save_name)
        # txt
        image = save_name
        save_name = path + 'resized_' + str(i*10+1) + '_gray'
        txtWriter(image, save_name)          
    
    


    