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


def mapped_id_to_original_groundings(quadropoles, entity_to_id, rel_to_id):
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
    rel_1 = []
    rel_2 = []
    obj = []
    entity_to_id = dict(zip(entity_to_id[0].astype(int), entity_to_id[1]))
    #print(entity_to_id)
    #exit()
    rel_to_id = dict(zip(rel_to_id[0].astype(int), rel_to_id[1]))

    for i in range(len(quadropoles)):
        sub.append(entity_to_id[quadropoles[i][0]])
        rel_1.append(rel_to_id[quadropoles[i][2]])
        rel_2.append(rel_to_id[quadropoles[i][3]])
        obj.append(entity_to_id[quadropoles[i][1]])

    original_groundings = np.c_[sub, obj, rel_1 , rel_2]
    return pd.DataFrame(original_groundings)


data_dir = '/home/mirza/PycharmProjects/frame_work/dataset/fb15k'
#save_data_dir = os.path.join(data_dir,save_data_dir)


entity_to_id_dir = os.path.join(data_dir, 'entities.dict')
relation_to_id_dir = os.path.join(data_dir, 'relations.dict')
#entity_to_id = pd.read_table(entity_to_id_dir, header=None)
entity_to_id = np.loadtxt(entity_to_id_dir, dtype=str)
entity_to_id = pd.DataFrame(entity_to_id)
#rel_to_id = pd.read_table(relation_to_id_dir, header=None)
rel_to_id = np.loadtxt(relation_to_id_dir, dtype=str)
rel_to_id = pd.DataFrame(rel_to_id)

grounding_implication = np.loadtxt(os.path.join(data_dir, 'groundings_implication.txt'))
original_grounding_implication = mapped_id_to_original_groundings(grounding_implication, entity_to_id, rel_to_id)
grounding_inverse= np.loadtxt(os.path.join(data_dir, 'groundings_inverse_test.txt'))
original_grounding_inverse = mapped_id_to_original_groundings(grounding_inverse, entity_to_id, rel_to_id)
grounding_symmetric = np.loadtxt(os.path.join(data_dir, 'groundings_symmetric.txt'))
original_grounding_symmetric = mapped_id_to_original_groundings(grounding_symmetric, entity_to_id, rel_to_id)
grounding_equivalence = np.loadtxt(os.path.join(data_dir, 'groundings_equivalence.txt'))
original_grounding_equivalence = mapped_id_to_original_groundings(grounding_equivalence, entity_to_id, rel_to_id)


write_implication_grounding_dir = os.path.join(data_dir, 'groundings_implication_original.txt')
write_inverse_grounding_dir = os.path.join(data_dir, 'groundings_inverse_original_test.txt')
write_symmetric_grounding_dir = os.path.join(data_dir, 'groundings_symmetric_original.txt')
write_equivalence_grounding_dir = os.path.join(data_dir, 'groundings_equivalence_original.txt')


write_to_txt_file(write_implication_grounding_dir ,np.array(original_grounding_implication))
write_to_txt_file(write_inverse_grounding_dir ,np.array(original_grounding_inverse))
write_to_txt_file(write_symmetric_grounding_dir ,np.array(original_grounding_symmetric))
write_to_txt_file(write_equivalence_grounding_dir ,np.array(original_grounding_equivalence))
