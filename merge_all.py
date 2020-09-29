import numpy as np
import pandas as pd
import glob as glob

### path configuration ###

# rgb path
rgb_path = glob.glob('/users/jiwon/documents/matlab/sync/bag_results/e*/rgb')
print(len(rgb_path))  # 15

# matching_can_index path
matching_can_path = glob.glob('/users/jiwon/documents/matlab/sync/bag_results/matching_can_index/*')
print(len(matching_can_path))  # 15

# can_output path
can_data_path = glob.glob('/users/jiwon/documents/matlab/sync/can_outputs/*')
print(len(can_data_path))  # 15

# vicon_joint_angle path (already high signal only)
vicon_path = glob.glob('/users/jiwon/documents/matlab/sync/vicon_high_joint_angles/*.csv')
print(len(vicon_path))  # 15

# valid_can path
valid_can_path = glob.glob('/users/jiwon/documents/matlab/sync/valid_can_data/*.csv')

# file name list
file_names = []
for v in vicon_path:
    index_end = v.find('_joint_angles.csv')
    index_start = v.rfind('/')
    file_names.append(v[index_start+1:index_end])
print(file_names)

# order check
rgb_list = []
matching_can_list = []
can_list = []
vicon_list = []
valid_can_list = []
for f in file_names:
    for r, m, c, v, vc in zip(rgb_path, matching_can_path, can_data_path, vicon_path, valid_can_path):
        if f in r:
            rgb_list.append(r)
        if f in m:
            matching_can_list.append(m)
        if f in c:
            can_list.append(c)
        if f in v:
            vicon_list.append(v)
        if f in vc:
            valid_can_list.append(vc)

### find matching VICON joint angles (CAN->VICON Sync) -> "valid_vicon"
for c, m, v, vc, f in zip(can_list, matching_can_list, vicon_list, valid_can_list, file_names):
    can_data = pd.read_csv(c)  # use: get id=50 index
    can_data = np.array(can_data)
    matching_can_data = pd.read_csv(m)  # use: position of valid CAN w.r.t. raw can_data (cf. MATLAB index)
    matching_can_data = np.array(matching_can_data)
    valid_can_data = pd.read_csv(vc)  # only valid id=80 data (matches with realsense)
    valid_can_data = np.array(valid_can_data)
    vicon_data = pd.read_csv(v)  # joint angles with id=50 group number
    vicon_data = np.array(vicon_data)

    # num_can
    can_50 = []  # can_data index of id=50
    num_can = []  # num of can_data(id=80) of each id=50 group
    for i in range(can_data.shape[0]):
        if can_data[i, 1] == 50:
            can_50.append(i)
    for i in range(len(can_50)-1):
        num_can.append(can_50[i+1] - can_50[i] - 1)  # 50~50 사이의 80 개수
    num_can.append((can_data.shape[0]-1) - can_50[-1])  # 마지막 50에 해당하는 80 개수 (last index  - last id=50 index)
    
    # matching_can_data -> matlab 기준 index이므로 1씩 빼야함
    matching_can_data -= 1

    # ratio of valid_can
    # 1. find which group it belongs to
    group = []  # 몇번째 can_50인지
    group_50_idx = []  # 해당 can_50의 index(can_data 상에서)
    for i in matching_can_data[:, 0]:
        for j in range(len(can_50)):
            if i >= can_50[j]:
                group_temp = j
                group_50_idx_temp = can_50[j]
        group.append(group_temp)
        group_50_idx.append(group_50_idx_temp)
    # 2. get ratio
    ratio = []
    for i in range(matching_can_data.shape[0]):
        ratio.append( (matching_can_data[i, 0] - group_50_idx[i]) / num_can[group[i]] )

    # num_vicon - id=50 별 vicon 개수 (**주의: vicon 파일에 id=50 개수가 더 많은 경우 있음 -> 뒷부분 자르기)
    num_vicon = []
    for i in np.arange(1, len(can_50)+1):
        count = 0
        for j in range(vicon_data.shape[0]):
            if vicon_data[j, 0] == i:
                count += 1
        num_vicon.append(count)
    
    # vicon_start - 시작점
    vicon_start = []
    vicon_start.append(0)
    for i in range(sum(num_vicon)-1):
        if (vicon_data[i+1, 0] - vicon_data[i, 0]) == 1:
            vicon_start.append(i+1)

    # matching_vicon_index
    matching_vicon_index = []        
    for i in range(matching_can_data.shape[0]):
        vicon_index = vicon_start[group[i]] + int(ratio[i] * num_vicon[group[i]])-1  # 오류나면 -1하기
        matching_vicon_index.append(vicon_index)

    # valid vicon
    valid_vicon = []
    print(f)
    for i in matching_vicon_index:
        temp = []
        for j in range(3):
            temp.append(vicon_data[i, j+1])  # mcp, pip, dip joint angles
        valid_vicon.append(temp)
    valid_vicon = np.array(valid_vicon)

    # merge real_valid_can and valid_vicon
    merged = np.concatenate((valid_can_data, valid_vicon), axis=1)
    save_path = '/users/jiwon/documents/matlab/sync/final_sync_data/' + f + '.csv'
    np.savetxt(save_path, merged, delimiter=',', fmt='%s')

