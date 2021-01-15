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
        #print(line)
    f.close()

base_dir = '/home/tansen/my files/thesis/mirza_framework_for_rotateE/rule_learning_framework-master/dataset/yago3_10/mapped_two/'

train_data = pd.DataFrame(np.array(pd.read_table(os.path.join(base_dir, 'train.txt'), header=None).drop_duplicates()))
test_data =pd.DataFrame(np.array(pd.read_table(os.path.join(base_dir, 'test.txt'), header=None).drop_duplicates()))



temp = pd.merge(test_data, train_data, how='outer', indicator=True)
temp = temp.loc[temp._merge == 'left_only', [0,1,2]]
write_to_txt_file(os.path.join(base_dir, 'test_revised.txt'), np.array(temp))