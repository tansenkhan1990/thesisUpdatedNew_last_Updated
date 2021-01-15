import pandas as pd
import numpy as np
import os

def swap_columns(data, first_column, second_column):
    swapped = data.columns
    swapped_data = data[swapped[np.r_[second_column, first_column, 2:len(swapped)]]]
    return swapped_data

def write_to_txt_file(path, data):
    """
    :param

    path: path where to be saved
    data: triples to be written in txt


    """
    f = open(path, "w")
    for i in range(data.shape[0]):
        line = ''
        for j in range(data.shape[1]):
            if(j==0):
                line = str(data[i][j])
            else:
                line = line + '\t' + str(data[i][j])
        f.write(line)
        f.write("\n")
        print(line)
    f.close()

data_dir = '/home/mirza/PycharmProjects/frame_work/dataset/youtube_dataset/mapped'
######################################################################################
save_file_name = 'original_quadropoles.txt'
save_data_dir = os.path.join(data_dir,save_file_name)

data = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/youtube_dataset/original_quadropoles.txt', header=None, sep=' ')
swapped_data = swap_columns(data, 0, 1)

write_to_txt_file(save_data_dir, np.array(swapped_data))
######################################################################################
save_file_name = 'test.txt'
save_data_dir = os.path.join(data_dir,save_file_name)

test_data = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/youtube_dataset/test.txt', header=None, sep=' ')
swapped_test_data = swap_columns(test_data, 0, 1)
swapped_test_data = swapped_test_data.loc[swapped_test_data[3]==1]
swapped_test_data = swapped_test_data.drop(columns = [3])

write_to_txt_file(save_data_dir, np.array(swapped_test_data))
################################################################################################
save_file_name = 'valid.txt'
save_data_dir = os.path.join(data_dir,save_file_name)

valid_data = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/youtube_dataset/valid.txt', header=None, sep=' ')
swapped_valid_data = swap_columns(valid_data, 0, 1)
swapped_valid_data = swapped_valid_data.loc[swapped_valid_data[3]==1]
swapped_valid_data = swapped_valid_data.drop(columns = [3])

write_to_txt_file(save_data_dir, np.array(swapped_valid_data))
