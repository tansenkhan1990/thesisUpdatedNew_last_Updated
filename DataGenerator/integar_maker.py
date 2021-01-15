import pandas as pd
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
        print(line)
    f.close()

entity_to_id = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/WN18RR/entities.dict', header=None, dtype=int)
entity_to_id = np.array(entity_to_id)
write_to_txt_file('/home/mirza/PycharmProjects/frame_work/dataset/WN18RR/entity2id_rev.txt',entity_to_id)
