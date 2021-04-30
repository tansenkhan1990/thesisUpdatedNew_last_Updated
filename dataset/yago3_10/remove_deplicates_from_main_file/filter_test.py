import pandas as pd
import os
import numpy as np


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

data_dir = '/home/tansen/Desktop/remove_deplicates_from_main_file'

maindata = pd.read_table(os.path.join(data_dir,'original_quadropoles.txt'),header= None, dtype=str)
data = np.array(maindata).astype(str)
data = np.unique(data, axis=0)
write_to_txt_file(os.path.join(data_dir, 'original_quadropoles.txt'), data)

# maindata = pd.DataFrame(maindata)
#
# duplicatedData = maindata[maindata.duplicated()]
#
# removeDuplicateData = maindata.drop_duplicates()
#
# dup2 = duplicatedData = removeDuplicateData[removeDuplicateData.duplicated()]
#
# print(maindata.head(10))
#
# print(removeDuplicateData.sum())
#
# print(duplicatedData.head())

print(dup2)