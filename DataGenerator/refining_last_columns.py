import numpy as np
import pandas as pd
import os

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

save_dir = '/dataset/yago3_10/mapped_two'
data_dir = '/dataset/yago3_10/mapped'
train = pd.read_table(os.path.join(data_dir, 'train.txt'), header=None)
train_array = train.iloc[:,:3].values
test = pd.read_table(os.path.join(data_dir, 'test.txt'), header=None)
test_array = test.iloc[:,:3].values

write_to_txt_file(os.path.join(save_dir,'train.txt'),train_array)
write_to_txt_file(os.path.join(save_dir,'test.txt'),test_array)
