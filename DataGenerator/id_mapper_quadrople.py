import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split

def create_mappings(quadropoles):
    """
      :param

      triples: the training triple in rdf format

      :return entity to id dictionary, relation to id dictionary:
      """
    entities = np.unique(np.ndarray.flatten(np.concatenate([quadropoles[:, 0:1], quadropoles[:, 2:3]])))
    relations = np.unique(np.ndarray.flatten(quadropoles[:, 1:2]).tolist())
    time = np.unique(np.ndarray.flatten(quadropoles[:, 3:4]).tolist())
    location =  np.unique(np.ndarray.flatten(quadropoles[:, 4:5]).tolist())
    entity_to_id = {value: key for key, value in enumerate(entities)}
    rel_to_id = {value: key for key, value in enumerate(relations)}
    time_to_id = {value: key for key, value in enumerate(time)}
    location_to_id = {value: key for key, value in enumerate(location)}
    return entity_to_id, rel_to_id, time_to_id, location_to_id

def create_mapped_triples(triples, entity_to_id=None, rel_to_id=None):
    """
    :param

    triples: the training triple in rdf format
    entity_to_id: the dictionary of entity
    rel_to_id: the dictionary of relation to id

    :return mapped triples of ids, entity to id dictionary, relation to id dictionary

    """
    if entity_to_id is None or rel_to_id is None:
        entity_to_id, rel_to_id = create_mappings(triples)
    subject_column = np.vectorize(entity_to_id.get)(triples[:, 0:1])
    relation_column = np.vectorize(rel_to_id.get)(triples[:, 1:2])
    object_column = np.vectorize(entity_to_id.get)(triples[:, 2:3])
    triples_of_ids = np.concatenate([subject_column, relation_column, object_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    triples_of_ids = np.unique(ar=triples_of_ids, axis=0)

    return triples_of_ids, entity_to_id, rel_to_id

def write_dic(path,d):
    """
    :param

    path: path where to be saved
    d: dictionary to be written in txt

    """
    f=open (path,"w")
    keys=d.keys()
    for k in keys:
        print(str(k)+'\t'+str(d[k]))
        f.write(str(k)+'\t'+str(d[k]))
        f.write("\n")
    f.close()

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

def create_mapped_id_to_triples(mapped_pos_train_tripels, entity_to_id, rel_to_id):
    """
    :param

    mapped_pos_train_tripels: the mapped ids of triples
    entity_to_id: Dictionary for entities
    rel_to_id: Dictionary for relation dictionary

    :returns

    the original triples: the original triples

    """
    sub = []
    rel = []
    obj = []
    entity_to_id = dict(zip(entity_to_id[0], entity_to_id[1]))
    rel_to_id = dict(zip(rel_to_id[0], rel_to_id[1]))

    for i in range(len(mapped_pos_train_tripels)):
        sub.append(entity_to_id[mapped_pos_train_tripels[0][i]])
        rel.append(rel_to_id[mapped_pos_train_tripels[1][i]])
        obj.append(entity_to_id[mapped_pos_train_tripels[2][i]])

    original_triples = np.c_[sub, rel, obj]
    return pd.DataFrame(original_triples)

def original_id_to_triples(triples, entity_to_id, rel_to_id):
    """
    :param

    mapped_pos_train_tripels: the mapped ids of triples
    entity_to_id: Dictionary for entities
    rel_to_id: Dictionary for relation dictionary

    :returns

    the original triples: the original triples

    """
    int_flag_sub = False
    triples = np.array(triples)
    sub = []
    rel = []
    obj = []
    entity_to_id = dict(zip(entity_to_id[1], entity_to_id[0]))
    rel_to_id = dict(zip(rel_to_id[1], rel_to_id[0]))

    if (set(map(type, entity_to_id)) == {int}):
        int_flag_sub = True
    else:
        int_flag_sub = False

    if int_flag_sub == True:
        for i in range(len(triples)):
            sub.append(entity_to_id[int(triples[i][0])])
            obj.append(entity_to_id[int(triples[i][2])])
            rel.append(rel_to_id[triples[i][1]])
    else:
        for i in range(len(triples)):
            sub.append(entity_to_id[(triples[i][0])])
            obj.append(entity_to_id[(triples[i][2])])
            rel.append(rel_to_id[triples[i][1]])

    original_triples = np.c_[sub, rel, obj]
    return pd.DataFrame(original_triples)

def original_id_to_quadropoles_str(quadropoles, entity_to_id, rel_to_id, time_to_id, location_to_id):
    """
    :param

    mapped_pos_train_tripels: the mapped ids of triples
    entity_to_id: Dictionary for entities
    rel_to_id: Dictionary for relation dictionary

    :returns

    the original triples: the original triples

    """
    quadropoles = np.array(quadropoles)
    sub = []
    rel = []
    obj = []
    time = []
    location = []
    entity_to_id = dict(zip(entity_to_id[1], entity_to_id[0]))
    rel_to_id = dict(zip(rel_to_id[1], rel_to_id[0]))
    time_to_id = dict(zip(time_to_id[1], time_to_id[0]))
    location_to_id = dict(zip(location_to_id[1], location_to_id[0]))

    for i in range(len(quadropoles)):
        sub.append(entity_to_id[quadropoles[i][0]])
        rel.append(rel_to_id[quadropoles[i][1]])
        obj.append(entity_to_id[quadropoles[i][2]])
        time.append(time_to_id[quadropoles[i][3]])
        location.append(location_to_id[quadropoles[i][4]])

    original_quadropoles = np.c_[sub, rel, obj, time ,location]
    return pd.DataFrame(original_quadropoles)

def original_id_to_triples_str(triples, entity_to_id, rel_to_id):
    """
    :param

    mapped_pos_train_tripels: the mapped ids of triples
    entity_to_id: Dictionary for entities
    rel_to_id: Dictionary for relation dictionary

    :returns

    the original triples: the original triples

    """
    triples = np.array(triples)
    sub = []
    rel = []
    obj = []
    entity_to_id = dict(zip(entity_to_id[1], entity_to_id[0]))
    #print(entity_to_id)
    #exit()
    rel_to_id = dict(zip(rel_to_id[1], rel_to_id[0]))

    for i in range(len(triples)):
        sub.append(entity_to_id[triples[i][0]])
        rel.append(rel_to_id[triples[i][1]])
        obj.append(entity_to_id[triples[i][2]])

    original_triples = np.c_[sub, rel, obj]
    return pd.DataFrame(original_triples)

def mapped_id_to_original_triples(triples, entity_to_id, rel_to_id):
    """
    :param

    mapped_pos_train_tripels: the mapped ids of triples
    entity_to_id: Dictionary for entities
    rel_to_id: Dictionary for relation dictionary

    :returns

    the original triples: the original triples

    """
    triples = np.array(triples)
    sub = []
    rel = []
    obj = []
    entity_to_id = dict(zip(entity_to_id[0], entity_to_id[1]))
    rel_to_id = dict(zip(rel_to_id[0], rel_to_id[1]))

    for i in range(len(triples)):
        sub.append(entity_to_id[triples[i][0]])
        rel.append(rel_to_id[triples[i][1]])
        obj.append(entity_to_id[triples[i][2]])

    original_triples = np.c_[sub, rel, obj]
    return pd.DataFrame(original_triples)

# def swap_columns(df, c1, c2):
#     df['temp'] = df[c1]
#     df[c1] = df[c2]
#     df[c2] = df['temp']
#     df.drop(columns=['temp'], inplace=True)
# for wiki data
# data_dir = '/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata'
# save_data_dir = '/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/result'
#for yago dataset
data_dir = 'dataset/yago5'
save_data_dir = 'result'

#for dbpedia data
# data_dir = '/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/dbpediadata'
# save_data_dir = '/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/dbpediadata/result'

save_data_dir = os.path.join(data_dir,save_data_dir)
#for wikipedia data
# train_data_dir = os.path.join(data_dir, 'wikidata.txt')

#for dbpedia data
# train_data_dir = os.path.join(data_dir, 'dbpedia.txt')

#for yago 
train_data_dir = os.path.join(data_dir, 'yago.txt')

#for wikidata data
# quadropoles = pd.read_csv(train_data_dir, header=None, dtype=str)
#for dbpedia data
quadropoles = pd.read_csv(train_data_dir, header=None, sep='\t', dtype=str)
quadropoles = quadropoles.dropna(how='any',axis=0)

#train_pos, test_pos = train_test_split(pos_triples, test_size=0.2)
entity_to_id, rel_to_id, time_to_id, location_to_id = create_mappings(np.array(quadropoles))
ent2id = dict((v,k) for k,v in entity_to_id.items())
rel2id = dict((v,k) for k,v in rel_to_id.items())
time2id = dict((v,k) for k,v in time_to_id.items())
loc2id = dict((v,k) for k,v in location_to_id.items())

write_entity_to_id = os.path.join(data_dir, 'entities.dict')
write_relation_to_id = os.path.join(data_dir, 'relations.dict')
time_to_id = os.path.join(data_dir, 'times.dict')
location_to_id = os.path.join(data_dir, 'locations.dict')
write_dic(write_entity_to_id ,ent2id)
write_dic(write_relation_to_id,rel2id)
write_dic(time_to_id ,time2id)
write_dic(location_to_id,loc2id)

#new function for tran validate and test
# train_pos, test_pos = train_test_split(quadropoles, test_size=0.15)
train_pos, valid_pos, test_pos = np.split(quadropoles.sample(frac=1), [int(.6*len(quadropoles)), int(.8*len(quadropoles))])
base_path = 'dataset/yago5'
entity_to_id = pd.read_table(os.path.join(base_path,'entities.dict'), header=None, dtype=str)
rel_to_id = pd.read_table(os.path.join(base_path,'relations.dict'), header=None, dtype=str)
time_to_id = pd.read_table(os.path.join(base_path,'times.dict'), header=None, dtype=str)
location_to_id = pd.read_table(os.path.join(base_path,'locations.dict'), header=None, dtype=str)

train = original_id_to_quadropoles_str(train_pos, entity_to_id, rel_to_id, time_to_id, location_to_id)
test = original_id_to_quadropoles_str(test_pos, entity_to_id, rel_to_id, time_to_id, location_to_id)
valid = original_id_to_quadropoles_str(valid_pos, entity_to_id, rel_to_id, time_to_id, location_to_id)

train_save_dir = os.path.join(save_data_dir, 'train.txt')
test_save_dir = os.path.join(save_data_dir, 'test.txt')
valid_save_dir = os.path.join(save_data_dir, 'valid.txt')
write_to_txt_file(train_save_dir, np.array(train))
write_to_txt_file(test_save_dir, np.array(test))
write_to_txt_file(valid_save_dir, np.array(valid))


