import numpy as np
import os
import pandas as pd


def fetch_conclusion(grounding_data, grounding_type):

    conclusion = []

    if grounding_type == 'implication':
        for triple in grounding_data:
            conclusion.append(np.array([triple[0], triple[3], triple[1]]))

    elif grounding_type == 'inverse':
        for triple in grounding_data:
            conclusion.append(np.array([triple[1], triple[3], triple[0]]))

    elif grounding_type == 'symmetric':
        for triple in grounding_data:
            conclusion.append(np.array([triple[1], triple[2], triple[0]]))

    elif grounding_type == 'equivalence':
        for triple in grounding_data:
            conclusion.append(np.array([triple[0], triple[3], triple[1]]))

    else:
        print('wrong type of grounding has been inserted! The program will be stop now')
        exit()

    conclusion = np.array(conclusion)
    conclusion = pd.DataFrame(conclusion)

    return conclusion


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




def corrupt(data, entity_table):

    all_entities = entity_table[1]
    all_unique_entities = list(set(all_entities))

    corrupted_triples = []

    for triple in np.array(data):
        corrupt_ind = np.random.choice([0, 2], p=[0.5, 0.5])
        candidate = np.random.choice([x for x in all_unique_entities if x != triple[corrupt_ind]])
        # noisy_triple = np.copy(triple)
        triple[corrupt_ind] = candidate
        corrupted_triples.append(triple)

    corrupted_triples_triples = np.array(corrupted_triples)
    corrupted_triples = pd.DataFrame(corrupted_triples_triples)

    return corrupted_triples

def add_grounded_noise(triple_data, grounding_data , entity_table, grounding_corrupt_percentage = 0.05):

    grounding_data_fraction = grounding_data.sample(frac=grounding_corrupt_percentage)
    corrupt_grounding_conclusion_data = corrupt(grounding_data_fraction, entity_table)

    noisy_training_triple = triple_data.append(corrupt_grounding_conclusion_data, ignore_index=True)

    return noisy_training_triple, corrupt_grounding_conclusion_data



data_dir = '/home/mirza/PycharmProjects/frame_work/dataset/FB15k_noise_included'

training_data_dir = '/home/mirza/PycharmProjects/frame_work/dataset/FB15k_noise_included/original_quadropoles.txt'
triple_data = pd.read_table(training_data_dir, header=None, dtype=str)

entity_table = pd.read_table(os.path.join(data_dir,'entities.dict'), header=None, dtype=str)

groundings_implication = pd.read_table(os.path.join(data_dir,'groundings_implication_original.txt'), header=None, dtype=str)
groundings_implication = np.array(groundings_implication)

groundings_inverse = pd.read_table(os.path.join(data_dir,'groundings_inverse_original.txt'), header=None, dtype=str)
groundings_inverse = np.array(groundings_inverse)

groundings_symmetric = pd.read_table(os.path.join(data_dir,'groundings_symmetric_original.txt'), header=None, dtype=str)
groundings_symmetric = np.array(groundings_symmetric)

groundings_equivalence = pd.read_table(os.path.join(data_dir,'groundings_equivalence_original.txt'), header=None, dtype=str)
groundings_equivalence = np.array(groundings_equivalence)



conclusion_implication = fetch_conclusion(groundings_implication, grounding_type='implication')
corrupted_triple_implication_grounding, corrupt_grounding_conclusion_data_implication = add_grounded_noise(triple_data, grounding_data= conclusion_implication, entity_table = entity_table ,grounding_corrupt_percentage = 0.25)
#print(corrupt_grounding_conclusion_data_implication.shape)

conclusion_inverse = fetch_conclusion(groundings_inverse, grounding_type='inverse')
corrupted_triple_inverse_grounding, corrupt_grounding_conclusion_data_inverse = add_grounded_noise(triple_data, grounding_data= conclusion_inverse, entity_table = entity_table ,grounding_corrupt_percentage = 0.25)

conclusion_symmetric = fetch_conclusion(groundings_symmetric, grounding_type='symmetric')
corrupted_triple_symmetric_grounding, corrupt_grounding_conclusion_data_symmetric = add_grounded_noise(triple_data, grounding_data= conclusion_symmetric, entity_table = entity_table ,grounding_corrupt_percentage = 0.25)

conclusion_equivalence = fetch_conclusion(groundings_equivalence, grounding_type='equivalence')
corrupted_triple_equivalence_grounding, corrupt_grounding_conclusion_data_equivalence = add_grounded_noise(triple_data, grounding_data= conclusion_equivalence, entity_table = entity_table ,grounding_corrupt_percentage = 0.25)


all_corrupted_conclusion_triples = np.vstack((corrupt_grounding_conclusion_data_implication, corrupt_grounding_conclusion_data_inverse, corrupt_grounding_conclusion_data_symmetric, corrupt_grounding_conclusion_data_equivalence))
#all_corrupted_conclusion_triples = np.vstack(( corrupt_grounding_conclusion_data_inverse, corrupt_grounding_conclusion_data_symmetric))
#print(all_corrupted_conclusion_triples.shape)
noisy_training_triple_all_conclusion = triple_data.append(pd.DataFrame(all_corrupted_conclusion_triples), ignore_index=False)

#write_data_dir_implication = os.path.join(data_dir,'train_25p_implication_conclusion.txt')
#write_data_dir_inverse = os.path.join(data_dir,'train_25p_inverse_conclusion.txt')
#write_data_dir_symmetric = os.path.join(data_dir,'train_25p_symmetric_conclusion.txt')
#write_data_dir_equivalence = os.path.join(data_dir,'train_25p_equivalence_conclusion.txt')
write_corrupted_data_all = os.path.join(data_dir,'corrupted_25p_all.txt')
write_data_dir_all = os.path.join(data_dir,'train_25p_all.txt')

#write_to_txt_file(write_data_dir_implication,np.array(corrupted_triple_implication_grounding))
#write_to_txt_file(write_data_dir_inverse,np.array(corrupted_triple_inverse_grounding))
#write_to_txt_file(write_data_dir_symmetric,np.array(corrupted_triple_symmetric_grounding))
write_to_txt_file(write_corrupted_data_all,np.array(all_corrupted_conclusion_triples))
write_to_txt_file(write_data_dir_all,np.array(noisy_training_triple_all_conclusion))


