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





data_dir = '/home/mirza/PycharmProjects/frame_work/dataset/wn18_noise_included'
#file = 'test_fb15k'
#training_data_dir = '/home/mirza/PycharmProjects/frame_work/dataset/wn18_noise_included/original_quadropoles.txt'
#triple_data = pd.read_table(training_data_dir, header=None)

entity_table = pd.read_table(os.path.join(data_dir,'entities.dict'), header=None, dtype=str)

groundings_implication = pd.read_table(os.path.join(data_dir,'groundings_implication_original.txt'), header=None,dtype=str)
groundings_implication = np.array(groundings_implication)

groundings_inverse = pd.read_table(os.path.join(data_dir,'groundings_inverse_original.txt'), header=None, dtype=str)
groundings_inverse = np.array(groundings_inverse)

groundings_symmetric = pd.read_table(os.path.join(data_dir,'groundings_symmetric_original.txt'), header=None, dtype=str)
groundings_symmetric = np.array(groundings_symmetric)

groundings_equivalence = pd.read_table(os.path.join(data_dir,'groundings_equivalence_original.txt'), header=None, dtype=str)
groundings_equivalence = np.array(groundings_equivalence)

conclusion_implication = fetch_conclusion(groundings_implication, grounding_type='implication')
#corrupted_triple_implication_grounding = add_grounded_noise(triple_data, grounding_data= conclusion_implication, entity_table = entity_table ,grounding_corrupt_percentage = 0.05)

conclusion_inverse = fetch_conclusion(groundings_inverse, grounding_type='inverse')
#corrupted_triple_inverse_grounding = add_grounded_noise(triple_data, grounding_data= conclusion_inverse, entity_table = entity_table ,grounding_corrupt_percentage = 0.05)

conclusion_symmetric = fetch_conclusion(groundings_symmetric, grounding_type='symmetric')
#corrupted_triple_symmetric_grounding = add_grounded_noise(triple_data, grounding_data= conclusion_symmetric, entity_table = entity_table ,grounding_corrupt_percentage = 0.05)

conclusion_equivalence = fetch_conclusion(groundings_equivalence, grounding_type='equivalence')
#corrupted_triple_equivalence_grounding = add_grounded_noise(triple_data, grounding_data= conclusion_equivalence, entity_table = entity_table ,grounding_corrupt_percentage = 0.05)


write_data_dir_implication = os.path.join(data_dir,'test_wn18_implication_conclusion_100%_true.txt')
write_data_dir_inverse = os.path.join(data_dir,'test_wn18_inverse_conclusion_100%_true.txt')
write_data_dir_symmetric = os.path.join(data_dir,'test_wn18_symmetric_conclusion_100%_true.txt')
write_data_dir_equivalence = os.path.join(data_dir,'test_wn18_equivalence_conclusion_100%_true.txt')

write_to_txt_file(write_data_dir_implication,np.array(conclusion_implication))
write_to_txt_file(write_data_dir_inverse,np.array(conclusion_inverse ))
write_to_txt_file(write_data_dir_symmetric,np.array(conclusion_symmetric))
write_to_txt_file(write_data_dir_equivalence,np.array(conclusion_equivalence))

