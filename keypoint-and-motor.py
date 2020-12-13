import numpy as np
import pandas as pd
import glob
import csv
import cv2

def idxOrder(folder):    
    # get idx order
    idx_list = []
    json_files = sorted(glob.glob(folder+'*.json')) # 이상한 순서긴 하지만 keypoints-and-img.py와 동일한 순서
    for j in json_files:
        idx = j[j.find('resized_')+8:j.find('_gray.')]
        idx_list.append(idx)
    return idx_list

def folderName(path): # 상황마다 str find 적절히 수정해서 쓸 것 -> 현재: annotation 기준
    train_or_test = path[path.find('/annotation/')+12:path.find('/annotation/')+16]
    if train_or_test == 'test':
        folder_name = path[path.find('/test/')+6:path.find('/ann/')]
    else:
        folder_name = path[path.find('/train/')+7:path.find('/ann/')]
    return folder_name

def matchingDepthFile(folder_name, idx):
    #folder_name = folderName(json_folder)
    depth_folder = '/users/jiwon/documents/강의/2020-2/기계시스템설계2/sync/bag_results/'+folder_name+'/depth/'
    matching_depth_file = depth_folder + str(idx) + '_depth.jpg'
    return matching_depth_file

def keypointZ(depth_jpg, keypoints):
    # get depth value for keypoints
    img = cv2.imread(depth_jpg, cv2.IMREAD_GRAYSCALE)
    z_set = []
    for i in range(4):
        row = int(keypoints[2*i+1]) # pixel 좌표는 [가로,세로]임에 주의
        col = int(keypoints[2*i])
        z_set.append(img[row-1, col-1])
    return z_set # [mcp_z, pip_z, dip_z, tip_z]

def dotTheta(v1, v2): # in degree
    return np.arccos(sum(v1*v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def jointAngles(keypoints3d):
    for i in range(len(keypoints3d)):
        keypoints3d[i] = float(keypoints3d[i])

    mcp = np.array(keypoints3d[0:3]) # [x, y, z]
    pip = np.array(keypoints3d[3:6])
    dip = np.array(keypoints3d[6:9])
    tip = np.array(keypoints3d[9:12])

    proximal = pip - mcp
    intermediate = dip - pip
    distal = tip - dip

    pip_angle = dotTheta(proximal, intermediate) * 180 / np.pi
    dip_angle = dotTheta(intermediate, distal) * 180 / np.pi
     
    return [pip_angle, dip_angle]

# prepare motor and joint angle data
motor_path = '/users/jiwon/documents/강의/2020-2/기계시스템설계2/sync/final_sync_data/'
def matchingMotorVicon(motor_csv, idx):
    idx = int(idx)
    df = pd.read_csv(motor_csv).drop(['time','id'], axis=1)
    motor_vicon = df.iloc[idx-1] # ex. resized_1_gray -> get df.iloc[0]
    motor_vicon_list = []
    for elem in motor_vicon:
        motor_vicon_list.append(elem)
    return motor_vicon_list  # list, [v1, v2, v3, ..., mcp_vicon, pip_vicon, dip_vicon]

###

# 1. prepare keypoint data (x, y and z-depth) - train: use train, test: use predicted result (drop gt)
path = '/users/jiwon/documents/강의/2020-2/기계시스템설계2/dataset/real/480_revised/'
test_keypoints = pd.read_csv(path+'keypoints_result.csv')  # integer x,y
columns_to_drop = ['mcp_x','mcp_y','pip_x','pip_y','dip_x','dip_y','tip_x','tip_y'] # remove gt data and only use predicted result
pred_keypoints = test_keypoints.drop(columns_to_drop, axis=1)
pred_keypoints.columns = columns_to_drop # unify column names btw pred & train
train_keypoints = pd.read_csv(path+'train_input.csv')
train_keypoints = train_keypoints.drop('img', axis=1)

keypoints = pd.concat([pred_keypoints, train_keypoints], axis=0)
print(keypoints)
keypoints = np.array(keypoints).reshape(-1,8)

# prepare depth data (z) - 1) as image, 2) resize, 3) get z values for keypoints -> 다른 파일에서 했음 (save-depth-as-img.py)

# 2. keypoint_folders: keypoint path -> to find idx order
keypoint_test_folders = sorted(glob.glob('/users/jiwon/documents/강의/2020-2/기계시스템설계2/dataset/real/annotation/test/*/ann/'))
keypoint_train_folders = sorted(glob.glob('/users/jiwon/documents/강의/2020-2/기계시스템설계2/dataset/real/annotation/train/*/ann/'))
keypoint_folders = []
for test_folder in keypoint_test_folders:
    keypoint_folders.append(test_folder)
for train_folder in keypoint_train_folders:
    keypoint_folders.append(train_folder)

folder_idx_set = []  # [['e0a0m100','#'],...['e100a0m0','###']]
for folder in keypoint_folders:
    folder_name = folderName(folder)
    json_files = sorted(glob.glob(folder+'*.json'))
    for j in json_files:
        folder_idx = []
        folder_idx.append(folder_name) # ex. e0a0m100
        file_id = j[j.find('resized_')+8:j.find('_gray.')] # ex. 1, 2, 3 ...
        folder_idx.append(file_id)
        folder_idx_set.append(folder_idx)

# 3. get z, compute joint angles, and get matching motor & vicon data
merged_data = [['pip_pred','dip_pred','v1','v2','v3','p1','p2','p3','f1','f2','f3','mcp','pip','dip']]
for i in range(len(folder_idx_set)):
    # 1) finding z value
    id_set = folder_idx_set[i]
    folder = id_set[0]
    idx = id_set[1]
    corres_keypoints = keypoints[i].tolist()
    depth_jpg = '/users/jiwon/documents/강의/2020-2/기계시스템설계2/sync/bag_results/'+folder+'/depth/'+idx+'_depth.jpg'
    keypoints_z = keypointZ(depth_jpg, corres_keypoints) # [mcp_z, pip_z, dip_z, tip_z]
    
    for j in range(4):
        corres_keypoints.insert(3*i+2, keypoints_z[j])
    keypoints_3d = corres_keypoints # [mcp_x, y, z, pip_x, y, z, ..., tip_x, y, z]

    # 2) calculate joint angles
    joint_angles = jointAngles(keypoints_3d) # [pip, dip] in degree

    # 3) get matching motor & vicon data
    motor_csv = motor_path + folder + '.csv'
    motor_n_vicon = matchingMotorVicon(motor_csv, idx) # [v1,v2,v3, ..., mcp,pip,dip]

    # 4) merge all (joint_angles & motor_n_vicon)
    merged = [] # [pip, dip, v1,v2,..., mcp, pip, dip]
    for elem1 in joint_angles:
        merged.append(elem1)
    for elem2 in motor_n_vicon:
        merged.append(elem2)
    
    merged_data.append(merged)

# save merged data as csv
merged_data = np.array(merged_data).reshape(-1,14)
save_path = '/users/jiwon/documents/강의/2020-2/기계시스템설계2/dataset/real/480_revised/merged_data.csv'
np.savetxt(save_path, merged_data, fmt='%s', delimiter=',')

    






    

    

    


