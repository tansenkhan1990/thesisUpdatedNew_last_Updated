import pandas as pd
import numpy as np
from sklearn.utils import shuffle
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


def mapped_id_to_original_fifthopoles(fifthopoles, entity_to_id, rel_to_id, tm_to_id, loc_to_id):
    """
    :param

    mapped_pos_train_tripels: the mapped ids of triples
    entity_to_id: Dictionary for entities
    rel_to_id: Dictionary for relation dictionary

    :returns

    the original triples: the original triples

    """
    quadropoles = np.array(fifthopoles)
    sub = []
    rel = []
    obj = []
    tm = []
    loc = []
    entity_to_id = dict(zip(entity_to_id[0].astype(int), entity_to_id[1]))
    rel_to_id = dict(zip(rel_to_id[0].astype(int), rel_to_id[1]))
    tm_to_id = dict(zip(tm_to_id[0].astype(int), tm_to_id[1]))
    loc_to_id = dict(zip(loc_to_id[0].astype(int), loc_to_id[1]))

    for i in range(len(quadropoles)):
        sub.append(entity_to_id[quadropoles[i][0]])
        rel.append(rel_to_id[quadropoles[i][1]])
        obj.append(entity_to_id[quadropoles[i][2]])
        tm.append(tm_to_id[quadropoles[i][3]])
        loc.append(loc_to_id[quadropoles[i][4]])

    original_fifthopoles = np.c_[sub, rel, obj , tm, loc]
    return pd.DataFrame(original_fifthopoles)


# def swap_columns(df, c1, c2):
#     df['temp'] = df[c1]
#     df[c1] = df[c2]
#     df[c2] = df['temp']
#     df.drop(columns=['temp'], inplace=True)


#for wikipedia data
# data_dir = '/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/result'
#for dbpedia data
data_dir = '/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/result'
save_data_dir = 'mapped'
train = pd.read_table(os.path.join(data_dir, 'train.txt'), header=None)
test = pd.read_table(os.path.join(data_dir, 'test.txt'), header=None)
entity_to_id = pd.read_table(os.path.join(data_dir, 'entities.dict'), header=None)
rel_to_id = pd.read_table(os.path.join(data_dir, 'relations.dict'), header=None)
tm_to_id = pd.read_table(os.path.join(data_dir, 'times.dict'), header=None)
loc_to_id = pd.read_table(os.path.join(data_dir, 'locations.dict'), header=None)

train_original = mapped_id_to_original_fifthopoles(train, entity_to_id, rel_to_id, tm_to_id, loc_to_id)
test_original = mapped_id_to_original_fifthopoles(test, entity_to_id, rel_to_id, tm_to_id, loc_to_id)

#for wikipedia data
# train_original.to_csv('/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/result/train_original.txt', index=False, header=None, sep = '\t')
# test_original.to_csv('/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/result/test_original.txt', index=False, header=None, sep='\t')

#for dbpedia data
train_original.to_csv('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/result/train_original.txt', index=False, header=None, sep = '\t')
test_original.to_csv('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/result/test_original.txt', index=False, header=None, sep='\t')
