import numpy as np
import csv
import glob
import cv2

#path = '/Users/jiwon/Documents/MATLAB/sync/bag_results/e0a30m100/rgb/'
path = '/Users/jiwon/Documents/MATLAB/grayscale/e0a30m100/'
save_path = '/Users/jiwon/Documents/MATLAB/grayscale/e0a30m100/'
img_list = glob.glob(path+'*.jpg')

def txtWriter(single_img, save_name):
    img = cv2.imread(single_img, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    img2list = ''
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img2list += str(' ' + str(img[i, j]))

    f = open(save_name, 'w')
    f.write(img2list)
    f.close()

for i in np.arange(50,100):
    image = path + str(i*10+1) + '_rgb.jpg'
    save_name = save_path+str(i*10+1)+'_gray.txt'
    txtWriter(image, save_name)        
