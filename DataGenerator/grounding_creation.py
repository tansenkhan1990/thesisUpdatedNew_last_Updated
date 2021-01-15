import os
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

def prepare_training_data(training_data):
    training_triples = list(zip([h for h in training_data[0]],
                                [t for t in training_data[2]],
                                [r for r in training_data[1]]))
    return training_triples


def ground_truth_generation(rule_bag, training_triples):
    implications, inverses, symmetrics, equivalences = [], [], [], []

    existing_triples = {}
    for triple in training_triples:
        #mark triples which exists in the dataset
        existing_triples[triple[0], triple[1], triple[2]] = 1

    for triple in training_triples:
        for rule in rule_bag:
            if triple[2] == rule[0]:
                if rule[2] == 0:
                    try:
                        exists = existing_triples[triple[0], triple[1], rule[1]]
                    except:
                        implication = [triple[0], triple[1], rule[0], rule[1]]
                        implications.append(implication)
                if rule[2] == 1:
                    try:
                        exists = existing_triples[triple[1], triple[0], rule[1]]
                    except:
                        inverse = [triple[0], triple[1], rule[0], rule[1]]
                        inverses.append(inverse)
                if rule[2] == 2:
                    try:
                        exists = existing_triples[triple[1], triple[0], rule[1]]
                    except:
                        symmetric = [triple[0], triple[1], rule[0], rule[1]]
                        symmetrics.append(symmetric)
                if rule[2] == 3:
                    try:
                        exists = existing_triples[triple[0], triple[1], rule[1]]
                    except:
                        equivalence = [triple[0], triple[1], rule[0], rule[1]]
                        equivalences.append(equivalence)

    return implications, inverses, symmetrics, equivalences

def create_rulebag(patterns):
    rule_bag = []
    for elements in patterns:
        parts = elements[0].split()
        print(parts)
        try:
            r1 = relation_dict[parts[1]]
            r2 = relation_dict[parts[5]]
            h1 = parts[0]
            h2 = parts[4]
            t1 = parts[2]
            t2 = parts[6]
        except:
            continue
        if h1 == h2 and t1 == t2 and r1 != r2:
            category = [r1, r2, 0]  # implication
            rule_bag.append(category)
            print('implication')
        if h1 == t2 and t1 == h2 and r2 != r1:
            category = [r1, r2, 1]  # inverse
            rule_bag.append(category)
            #inverse.append(singlerule)
            print('inverse')
        if h1 == t2 and t1 == h2 and r2 == r1:
            category = [r1, r2, 2]  # symmetric
            rule_bag.append(category)
            #symmetric.append(singlerule)
            print('symmetric')

    for i in range(0, len(rule_bag)):
        if rule_bag[i][2] == 0:
            for j in range(i, len(rule_bag)):
                if rule_bag[j][2] == 0 and rule_bag[j][0] == rule_bag[i][1] and rule_bag[j][1] == rule_bag[i][0]:
                    rule_bag[i][2] = 3
                    rule_bag[j][2] = 3

    return rule_bag



data_dir = '/home/mirza/PycharmProjects/frame_work/dataset/kinship'
patterns = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/kinship/kinship_train_rules_all_thresholded_rule.tsv', header=None).values
relation_df_data_dir = os.path.join(data_dir, 'relations.dict')
relation_df = pd.read_table(relation_df_data_dir, header=None)
relation_dict = dict(zip(relation_df[1], relation_df[0]))
n_rule = len(patterns)

rule_bag = create_rulebag(patterns)
rule_bag=np.array(rule_bag)

training_data_dir = os.path.join(data_dir, 'original_quadropoles.txt')
training_df = pd.read_table(training_data_dir, header=None)
training_triples = prepare_training_data(training_df)

implications, inverses, symmetrics, equivalences = ground_truth_generation(rule_bag,training_triples)

implication_grounding_data_dir = os.path.join(data_dir, 'groundings_implication.txt')
inverse_grounding_data_dir = os.path.join(data_dir, 'groundings_inverse.txt')
symmetric_grounding_data_dir = os.path.join(data_dir, 'groundings_symmetric.txt')
equivalence_grounding_data_dir = os.path.join(data_dir, 'groundings_equivalence.txt')

write_to_txt_file(implication_grounding_data_dir,np.array(implications))
write_to_txt_file(inverse_grounding_data_dir,np.array(inverses))
write_to_txt_file(symmetric_grounding_data_dir,np.array(symmetrics))
write_to_txt_file(equivalence_grounding_data_dir,np.array(equivalences))