import numpy as np
import pandas as pd
import os

def create_mappings(triples):
    """
      :param

      triples: the training triple in rdf format

      :return entity to id dictionary, relation to id dictionary:
      """
    entities = np.unique(np.ndarray.flatten(np.concatenate([triples[:, 0:1], triples[:, 2:3]])))
    relations = np.unique(np.ndarray.flatten(triples[:, 1:2]).tolist())
    entity_to_id = {value: key for key, value in enumerate(entities)}
    rel_to_id = {value: key for key, value in enumerate(relations)}
    return entity_to_id, rel_to_id
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



data_dir = '/home/mirza/PycharmProjects/frame_work/dataset/new_data_mojtaba'
file_data_dir = os.path.join(data_dir,'data.txt')
save_data_dir = 'mapped'

all_triples = np.loadtxt(fname=file_data_dir, dtype=str, comments='@Comment@ Subject Predicate Object')
all_triples = np.delete(all_triples, -1, axis=1)
all_triples = pd.DataFrame(all_triples, columns=["subject", "predicate", "object"])
all_triples = np.array(all_triples)

entity_to_id, rel_to_id = create_mappings(triples=all_triples)

ent2id = dict((v,k) for k,v in entity_to_id.items())
rel2id = dict((v,k) for k,v in rel_to_id.items())

write_entity_to_id = os.path.join(data_dir, 'entities.dict')
write_relation_to_id = os.path.join(data_dir, 'relations.dict')
write_dic(write_entity_to_id ,ent2id)
write_dic(write_relation_to_id,rel2id)

all_triples = pd.DataFrame(all_triples)
sample_test = all_triples.sample(frac = 0.10)


#data_fraction_index = np.array(data_fraction.index)
#data_less = data.drop(data.index[data_fraction_index])
all_triples_test_less = all_triples[~all_triples.isin(sample_test)].dropna()

sample_valid = all_triples_test_less.sample(frac = 0.10)

all_triples_test_valid_less = all_triples_test_less[~all_triples_test_less.isin(sample_valid)].dropna()

write_train_dir = os.path.join(data_dir, 'original_quadropoles.txt')
write_test_dir = os.path.join(data_dir, 'test.txt')
write_valid_dir = os.path.join(data_dir, 'valid.txt')

write_to_txt_file(write_train_dir,np.array(all_triples_test_valid_less))
write_to_txt_file(write_test_dir,np.array(sample_test))
write_to_txt_file(write_valid_dir,np.array(sample_valid))
