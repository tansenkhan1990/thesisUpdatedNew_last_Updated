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


def geneate_nosiy_data(data, entity_table, percentage = 0.05, base_dir = ''):

    data_fraction = data.sample(frac = percentage)

    #data_fraction_index = np.array(data_fraction.index)

    #data_less = data.drop(data.index[data_fraction_index])
    data_less = data[~data.isin(data_fraction)].dropna()

    #data_less_to_be_corrupted = data_less.sample(frac = percentage)

    #data_less_noise_reducted = data_less[~data_less.isin(data_less_to_be_corrupted)].dropna()

    all_entities = entity_table[1]
    all_unique_entities = list(set(all_entities))

    noisy_triples = []

    for triple in np.array(data_fraction):

        corrupt_ind = np.random.choice([0, 2], p=[0.5, 0.5])
        candidate = np.random.choice([x for x in all_unique_entities if x != triple[corrupt_ind]])
        #noisy_triple = np.copy(triple)
        triple[corrupt_ind] = candidate
        noisy_triples.append(triple)

    noisy_triples = np.array(noisy_triples)
    noisy_triples = pd.DataFrame(noisy_triples)

    noisy_training_triple = data_less.append(noisy_triples, ignore_index=False)

    write_to_txt_file(os.path.join(base_dir,'FN.txt'), np.array(data_fraction))
    write_to_txt_file(os.path.join(base_dir,'corrupted_25p_all.txt'), np.array(noisy_triples))
    write_to_txt_file(os.path.join(base_dir,'train_25p_all.txt'), np.array(noisy_training_triple))


base_dir = '/home/mirza/PycharmProjects/frame_work/dataset/wn18rr'
data = pd.read_table(os.path.join(base_dir,'original_quadropoles.txt'), header=None, dtype=str)
entity_table = pd.read_table(os.path.join(base_dir,'entities.dict'), header=None, dtype=str)

geneate_nosiy_data(data,entity_table,0.25, base_dir)




