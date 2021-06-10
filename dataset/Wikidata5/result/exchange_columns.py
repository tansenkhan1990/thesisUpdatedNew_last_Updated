import pandas as pd
import os
import  numpy as np
dataset = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/DBPedia5/dbpedia.txt',header=None, dtype=str)

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


dataset['temp'] = dataset[3]
dataset[3] = dataset[4]
dataset[4] = dataset['temp']
dataset.drop(columns=['temp'], inplace=True)
data_dir = '/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/DBPedia5'
# maindata = pd.read_table(os.path.join(data_dir,'yagoUpdated.txt'),header= None, dtype=str)
data = np.array(dataset).astype(str)
# data = np.unique(dataset, axis=0)
write_to_txt_file(os.path.join(data_dir, 'dbpediaiUpdated.txt'), data)
