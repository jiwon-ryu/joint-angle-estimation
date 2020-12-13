'''
    keypoints(.json) + image(.jpg) -> train_input & test_input (.csv)
    
    CHANGE_ME(train/test)
        1. keypoint_folder, save_path
        2. choose option 1 / 2
'''
import numpy as np
import glob
import json

def extract_json(json_file):
    with open(json_file) as j:
        data = json.load(j)
        n_points = 4  # mcp, pip, dip, tip
        xy_set = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(n_points):  # i:
            label = data['objects'][i]['classTitle']
            value = data['objects'][i]['points']['exterior'][0]
            if label == 'mcp': 
                xy_set[0] = value[0]
                xy_set[1] = value[1]
            elif label == 'pip':
                xy_set[2] = value[0]
                xy_set[3] = value[1]
            elif label == 'dip':
                xy_set[4] = value[0]
                xy_set[5] = value[1]
            elif label == 'tip':
                xy_set[6] = value[0]
                xy_set[7] = value[1]
        for i in range(len(xy_set)):
            xy_set[i] = (480/96) * round(xy_set[i])
    return xy_set

def append_img_to_keypoints(keypoint, img_csv):
    # read img (csv)
    csv_file = open(img_csv, 'r')
    img = ''
    for i in csv_file:
        img = i

    # append img in keypoints-set
    img_keypoint_set = keypoint
    img_keypoint_set.append(img)
    return img_keypoint_set

# folders
keypoint_folder = sorted(glob.glob('/users/jiwon/documents/강의/2020-2/기계시스템설계2/dataset/real/annotation/test/*/ann/'))
img_folder = sorted(glob.glob('/users/jiwon/documents/강의/2020-2/기계시스템설계2/grayscale/*/'))

combined_stack_altogether = [['mcp_x','mcp_y','pip_x','pip_y','dip_x','dip_y','tip_x','tip_y','img']]

test_stack_altogether = [['ImageId', 'img']] # for test data
nTest = 0 # for test data

for keypoint_path, img_path in zip(keypoint_folder, img_folder):
    folder_name = img_path[img_path.rfind('/',0,len(img_path)-1)+1:img_path.rfind('/')] # ex. e0a0m100

    # keypoint_list, img_list: file list in single folder
    keypoint_list = sorted(glob.glob(keypoint_path + '*.json')) # path of json files
    img_list = [] # path of jpg files
    for keypoint in keypoint_list:
        common_name = keypoint[keypoint.rfind('/')+1:keypoint.find('.')] # name (ex. resized_gray_1)
        img_list.append(img_path + common_name + '.csv')

    # for every json & csv, save combined csv
    combined_stack = [['mcp_x','mcp_y','pip_x','pip_y','dip_x','dip_y','tip_x','tip_y','img']]

    for k, i in zip(keypoint_list, img_list):
        print('processing: ', i)
        xy_set = extract_json(k)
        img_keypoint_set = append_img_to_keypoints(xy_set, i) # merge - [x,y,x,y,...,img]
        combined_stack.append(img_keypoint_set) # stack - [[key],[file1],[file2],...,[fileN]]
        combined_stack_altogether.append(img_keypoint_set) # save all data as one file 

        # test
        test_stack = [] # for test data
        nTest += 1
        test_stack.append(nTest)
        id_img_set = append_img_to_keypoints(test_stack, i) # merge - [id, img]
        test_stack_altogether.append(id_img_set) # save all data as one file
    combined_stack = np.array(combined_stack).reshape(-1,9)
    
    '''
    # save result for each folders -> 주석처리해도 상관없음
    save_path = '/users/jiwon/documents/강의/2020-2/기계시스템설계2/dataset/real/'
    np.savetxt(folder_name+'.csv', combined_stack, fmt='%s', delimiter=',')
    '''

'''
# OPTION 1: save result as a whole (train data)
combined_stack_altogether = np.array(combined_stack_altogether).reshape(-1, 9)
#save_path = '/users/jiwon/documents/강의/2020-2/기계시스템설계2/dataset/real/gt_keypoints.csv'
save_path = '/users/jiwon/documents/강의/2020-2/기계시스템설계2/dataset/real/train_input.csv'
np.savetxt(save_path, combined_stack_altogether, fmt='%s', delimiter=',')
'''

# OPTION 2: save result as a whole (test data)
test_stack_altogether = np.array(test_stack_altogether).reshape(-1, 2)
save_path = '/users/jiwon/documents/강의/2020-2/기계시스템설계2/dataset/real/test_input.csv'
np.savetxt(save_path, test_stack_altogether, fmt='%s', delimiter=',')
