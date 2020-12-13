import numpy as np
import csv
from PIL import Image
import cv2
import glob

def saveDepth2Img(csv_file, save_name):
    f = open(csv_file, 'r')
    rdr = csv.reader(f)
    img_str = []
    for line in rdr:
        img_str.append(line)
    f.close()

    img_str = np.array(img_str)
    img = []
    for i in range(np.shape(img_str)[0]):
        img_row = []
        for j in range(np.shape(img_str)[1]):
            img_row.append(int(img_str[i, j]))
        img.append(img_row)
    img = np.array(img)
    img = 255.0 * img / np.max(img)
    cv2.imwrite(save_name,img)

def resize(img, save_name): # save_name, img 같게 해서 덮어쓰기 하기
    image = cv2.imread(img)      
    image = cv2.resize(image, (480, 480), interpolation=cv2.INTER_CUBIC)
    img_numpy = np.array(image, 'uint8')
    cv2.imwrite(save_name, img_numpy)


# saveDepth2Img 먼저 실행한 후 주석처리 하고 resize 진행하였음
input_folders = sorted(glob.glob('/users/jiwon/documents/강의/2020-2/기계시스템설계2/sync/bag_results/*/depth/'))
for folder in input_folders:
    print(folder)
    #files = sorted(glob.glob(folder+'*.csv'))
    idx = np.arange(1, 1001, 10)
    files = []
    for i in idx:
        files.append(folder + str(i) + '_depth.jpg')
    for single_file in files:
        #file_name = single_file[single_file.find('/depth/')+7:single_file.rfind('.')] # ex. 1_depth
        #saveDepth2Img(single_file, folder+file_name+'.jpg')
        resize(single_file, single_file)
