import numpy as np
import cv2
import glob
 
path = '/users/jiwon/documents/matlab/grayscale/e0a30m100/'
#imglist = [os.path.join(path,file_name) for file_name in os.listdir(path)]
imglist = glob.glob(path+'*.jpg')
print('input N: ', len(imglist))

save_path = '/users/jiwon/documents/matlab/grayscale/e0a30m100/'
for img in imglist:
    img_name = img[img.rfind('/')+1::]
    image = cv2.imread(img)
    
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_CUBIC)
    img_numpy = np.array(image, 'uint8')
    cv2.imwrite(save_path+img_name, img_numpy)

output = glob.glob(save_path+'*.jpg')
print('output N: ', len(output))
