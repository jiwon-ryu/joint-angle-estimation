import numpy as np
import pandas as pd
import glob
import csv

# read can data
can_list = glob.glob('/Users/jiwon/Desktop/test/*.csv')

can_example = can_list[0]
print(can_list)

for c in can_list:
    print(c)  # processing file name
    can_name = c[c.rfind('/')+1::]
    save_path = '/Users/jiwon/Desktop/motor/' + 'motor_' + can_name  # save path of valid motor data - used at the end

    can_file = pd.read_csv(c)
    can_1 = can_file.iloc[:, 1:3]  # time, id
    can_2 = can_file.iloc[:, 12:20]  # 8 columns
    can_file = pd.concat([can_1, can_2], axis=1)  # merged
    can_file = can_file.to_numpy()

    # find start/end time and signals
    start_rows = []  # id: 50
    start_time = []
    can_file[:, 0] = 100 * ((can_file[:, 0] / 100) % 1)  # time: xx.xxx...
    for r in range(can_file.shape[0]):
        if can_file[r][1] == 50:
            start_rows.append(r)
            start_time.append(can_file[r, 0])

    end_rows = []  # id: 601
    count = 0
    for i in range(len(start_rows)):
        if i < len(start_rows)-1:
            for r in np.arange(start_rows[i], start_rows[i+1]):
                if can_file[r, 1] == 601:
                    for x in np.arange(r, start_rows[i], -1):
                        if can_file[x, 1] == 80:
                            last_80 = x
                            if (r - x) == 7:
                                end_rows.append(r)
                                break
                            else:
                                end_rows.append(x)
                                break
                    break
        else:  # last start-end
            for r in np.arange(start_rows[i], can_file.shape[0]):            
                if can_file[r, 1] == 601:
                    for x in np.arange(r, start_rows[i], -1):
                        if can_file[x, 1] == 80:
                            last_80 = x
                            if (r - x) == 7:
                                end_rows.append(r)
                                break
                            else:
                                end_rows.append(x)
                                break
                    break

    # extract from the very first to the very end
    can_file = can_file.tolist()
    valid_signals = []
    for i in range(len(start_rows)):
        valid_signals.append(can_file[start_rows[i]:end_rows[i]][:])

    for i in range(len(valid_signals)):
        for j in range(len(valid_signals[i])):
            valid_signals[i][j][2] = int(valid_signals[i][j][2])  # change str -> int
                
    # velocity, position (regardless of 'id')
    for i in range(len(valid_signals)):
        for j in range(len(valid_signals[i])):
            # 1. velocity
            sum = valid_signals[i][j][2] + valid_signals[i][j][3] * 256 + valid_signals[i][j][4] * (256 ** 2) + valid_signals[i][j][5] * (256 ** 3)
            if valid_signals[i][j][5] < 128:  # positive      
                valid_signals[i][j].append(sum)

            if valid_signals[i][j][5] >= 128:  # negetive
                valid_signals[i][j].append(sum - 2 * 128 * (256 ** 3))
            
            # 2. position
            sum = valid_signals[i][j][6] + valid_signals[i][j][7] * 256 + valid_signals[i][j][8] * (256 ** 2) + valid_signals[i][j][9] * (256 ** 3)
            if valid_signals[i][j][9] < 128:  # positive       
                valid_signals[i][j].append(sum)

            if valid_signals[i][j][9] >= 128:  # negative
                valid_signals[i][j].append(sum - 2 * 128 * (256 ** 3))

    # new_valid_signals: id=50, 80 only.
    # id=80 row contains all 9 motor data
    for i in range(len(valid_signals)):
        for j in range(len(valid_signals[i])):
            if valid_signals[i][j][1] == 80:
                del valid_signals[i][j][2:12]
                for k in range(9):   # pre-allcation of 9 motor data with zeros
                    valid_signals[i][j].append(0)            
                for k in np.arange(1, 7):  # replace zeros with actual motor data                
                    if valid_signals[i][j+k][1] == 281:
                        valid_signals[i][j][2] = valid_signals[i][j+k][10] # motor 1 - vel
                        valid_signals[i][j][5] = valid_signals[i][j+k][11] # motor 1 - pos
                    elif valid_signals[i][j+k][1] == 283:
                        valid_signals[i][j][3] = valid_signals[i][j+k][10] # motor 2 - vel
                        valid_signals[i][j][6] = valid_signals[i][j+k][11] # motor 2 - pos
                    elif valid_signals[i][j+k][1] == 285:
                        valid_signals[i][j][4] = valid_signals[i][j+k][10] # motor 3 - vel
                        valid_signals[i][j][7] = valid_signals[i][j+k][11] # motor 3 - pos
                    elif valid_signals[i][j+k][1] == 401:
                        valid_signals[i][j][8] = valid_signals[i][j+k][10] # motor 1 - target vel                    
                    elif valid_signals[i][j+k][1] == 403:
                        valid_signals[i][j][9] = valid_signals[i][j+k][10] # motor 2 - target vel
                    elif valid_signals[i][j+k][1] == 405:
                        valid_signals[i][j][10] = valid_signals[i][j+k][10] # motor 3 - target vel

    new_valid_signals = [] # id=50, 80 only. 2D array.
    for i in range(len(valid_signals)):
        valid_signals[i][0].remove(valid_signals[i][0][-1])  # make id=50 and id=80 same dimension
        new_valid_signals.append(valid_signals[i][0])  # append id=50 row
        for j in range(len(valid_signals[i])):
            if valid_signals[i][j][1] == 80:
                new_valid_signals.append(valid_signals[i][j])  # append id=80 rows

    # type test - if neither int nor float -> print the location
    for i in range(len(new_valid_signals)):
        for j in range(len(new_valid_signals[i])):
            if (type(new_valid_signals[i][j]) != int) and (type(new_valid_signals[i][j]) != float):
                print(i, j)

    # save as csv (new_valid_signals)
    new_valid_signals = np.array(new_valid_signals)
    np.savetxt(save_path, new_valid_signals, delimiter=',', fmt='%s')


           
        



        









