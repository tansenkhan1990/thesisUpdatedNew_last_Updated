import numpy as np
import pandas as pd
import os


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

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


data_dir = '/home/mirza/PycharmProjects/frame_work/dataset/fb15k'
train_data_dir = os.path.join(data_dir,'original_quadropoles.txt')
test_data_dir = os.path.join(data_dir,'test.txt')
validataion_data_dir = os.path.join(data_dir,'valid.txt')
ent2id_dir = os.path.join(data_dir, 'entities.dict')
relation2id_dir = os.path.join(data_dir, 'relations.dict')
rule_implication_dir = os.path.join(data_dir, 'rule_implication.txt')
rule_inverse_dir = os.path.join(data_dir, 'rule_inverse.txt')
rule_symmetric_dir = os.path.join(data_dir, 'rule_symmetric.txt')
rule_equivalence_dir = os.path.join(data_dir, 'rule_equivalence.txt')
groundings_implication_dir = os.path.join(data_dir, 'groundings_implication.txt')
groundings_inverse_dir = os.path.join(data_dir, 'groundings_inverse.txt')
groundings_symmetric_dir = os.path.join(data_dir, 'groundings_symmetric.txt')
groundings_equivalence_dir = os.path.join(data_dir, 'groundings_equivalence.txt')
test_set_refined_dir = os.path.join(data_dir,'test_refined.txt')

train_triples = np.loadtxt(fname= train_data_dir, dtype=str, comments='@Comment@ Subject Predicate Object')
test_triples = np.loadtxt(fname=test_data_dir, dtype=str, comments='@Comment@ Subject Predicate Object')
validation_triples = np.loadtxt(fname=validataion_data_dir, dtype=str, comments='@Comment@ Subject Predicate Object')
entity_to_id = np.loadtxt(fname=ent2id_dir, dtype=str, comments='@Comment@ id Entity')
relation_to_id = np.loadtxt(fname=relation2id_dir, dtype=str, comments='@Comment@ id Entity')
implication_rules = np.loadtxt(fname=rule_implication_dir, dtype=str, comments='@Comment@ Subject Predicate Object')
inverse_rules = np.loadtxt(fname=rule_inverse_dir, dtype=str, comments='@Comment@ Subject Predicate Object')
symmetric_rules = np.loadtxt(fname=rule_symmetric_dir, dtype=str, comments='@Comment@ Subject Predicate Object')
equivalence_rules = np.loadtxt(fname=rule_equivalence_dir, dtype=str, comments='@Comment@ Subject Predicate Object')
implication_groundings = np.loadtxt(fname=groundings_implication_dir, dtype=str, comments='@Comment@ H T R1 R2')
inverse_groundings = np.loadtxt(fname=groundings_inverse_dir, dtype=str, comments='@Comment@ H T R1 R2')
symmetric_groundings = np.loadtxt(fname=groundings_symmetric_dir, dtype=str, comments='@Comment@ H T R1 R2')
equivalence_groundings = np.loadtxt(fname=groundings_equivalence_dir, dtype=str, comments='@Comment@ H T R1 R2')

print('# Total Triples = ', (len(train_triples) + len(test_triples) + len(validation_triples)))
print('# training data = ', len(train_triples))
print('# test data = ', len(test_triples))
print('# validation data = ', len(validation_triples))
print('# entity = ', len(entity_to_id))
print('# relation = ', len(relation_to_id))
print('# implication rule = ', len(implication_rules))
print('# inverse rule = ', len(inverse_rules))
print('# symmetric rule = ', len(symmetric_rules))
print('# equivalence rule = ', len(equivalence_rules))
print('# implication groundings = ', len(implication_groundings))
print('# inverse groundings = ', len(inverse_groundings))
print('# symmetric groundings = ', len(symmetric_groundings))
print('# equivalence groundings = ', len(equivalence_groundings))


groundings_implication = np.loadtxt(groundings_implication_dir)
groundings_inverse = np.loadtxt(groundings_inverse_dir)
groundings_symmetric = np.loadtxt(groundings_symmetric_dir)
groundings_equivalence = np.loadtxt(groundings_equivalence_dir)
#test_triples = np.loadtxt(test_data_dir)


match_list = []
test_triples = pd.read_table(test_data_dir, header=None)
test_triples = np.array(test_triples)
test_triples_clone = np.copy(test_triples)

total_leakage = 0

#from grounding format h t r1 r2 we should check whether h r2 t exist in the test triples
#Get the number of implication leakage
non_reverse_triple = []
for triple in groundings_implication:
    #reverse_triple.append(np.array([triple[1],triple[2],triple[0]]))
    non_reverse_triple.append(np.array([triple[0],triple[3],triple[1]]))

non_reverse_triple = np.array(non_reverse_triple)
#print('having duplicate rows ', len(non_reverse_triple))
if (groundings_implication!=0).any():
    non_reverse_triple = unique_rows(non_reverse_triple)
    #print('not having duplicate rows ', len(non_reverse_triple))


observed_triples = {}
for triple in test_triples:
    observed_triples[triple[0],triple[1],triple[2]]=1

#print(len(observed_triples))

count = 0

for triple in non_reverse_triple:
    if (triple[0], triple[1], triple[2]) in observed_triples:
        if (groundings_implication!=0).any():
            match_list.append(np.array([triple[0], triple[1], triple[2]]))
        count+=1
    else:
        continue

'''
#print(len(refined_test_data))
#print(len(np.array(match_list)))
match_list = np.array(match_list).astype(int)
#test_triples_clone = pd.DataFrame(test_triples_clone)
#x = 0
#match_count = 0
for triple in match_list:
    test_triples_clone = np.delete(test_triples_clone, np.where(np.bitwise_and(np.bitwise_and((test_triples_clone[:, 0] == triple[0]), (test_triples_clone[:, 1] == triple[1])), (test_triples_clone[:, 2] == triple[2])))[0], 0)

print(len(np.array(test_triples_clone)))
'''

#print('total match found in test set', match_found_in_test_set)
total_leakage += count

print('# of implication Leakage = ', count)
print('% of implication Leakage = ', count/len(test_triples))


#############
'''
observed_triples = {}
for triple in test_triples_clone:
    observed_triples[triple[0],triple[1],triple[2]]=1
count = 0
for triple in non_reverse_triple:
    if (triple[0], triple[1], triple[2]) in observed_triples:

        #found = observed_triples[triple[0], triple[1], triple[2]]
        #here I have to remove the test files
        # refined_test_data = test_triples[
        #     np.logical_not(np.logical_and(
        #         np.logical_and(test_triples[:, 0] == triple[0], test_triples[:, 1] == triple[1]),
        #         np.logical_and(test_triples[:, 0] == triple[2])
        #     ))
        # ]
        #mask_array = (test_triples_clone[:, :] != [triple[0], triple[1], triple[2]]).all(axis=1)
        #test_triples_clone = test_triples_clone[mask_array]
        #match_found_in_test_set+=np.sum(mask_array[:] == False)
        #print(np.sum(mask_array[:] == False))
        #print('triple ', triple, ' is removed from test')
        count+=1
    else:
        continue
#print(len(refined_test_data))
total_leakage += count
print('# of test_triples after removal of implication leakage: ', len(test_triples_clone))
print('# of implication Leakage = ', count)
print('% of implication Leakage = ', count/len(test_triples_clone))
'''
##############


#from grounding format h t r1 r2 we should check whether t r2 h exist in the test triples
#Get the number of inverse leakage
reverse_triple = []
for triple in groundings_inverse:
    reverse_triple.append(np.array([triple[1],triple[3],triple[0]]))
    #reverse_triple.append(np.array([triple[1],triple[3],triple[0]]))

reverse_triple = np.array(reverse_triple)
if (groundings_inverse!=0).any():
    reverse_triple = unique_rows(reverse_triple)


observed_triples = {}
for triple in test_triples:
    observed_triples[triple[0],triple[1],triple[2]]=1

count = 0
for triple in reverse_triple:
    if (triple[0], triple[1], triple[2]) in observed_triples:
        if (groundings_inverse!=0).any():
            match_list.append(np.array([triple[0], triple[1], triple[2]]))
        count+=1
    else:
        continue

total_leakage += count
print('# of Inverse Leakage = ', count)
print('% of Inverse Leakage = ', count/len(test_triples))

#from grounding format h t r1 r2 we should check whether t r1 h exist in the test triples
#Get the number of symmetric leakage
reverse_triple = []
for triple in groundings_symmetric:
    reverse_triple.append(np.array([triple[1],triple[2],triple[0]]))
    #reverse_triple.append(np.array([triple[1],triple[3],triple[0]]))

reverse_triple = np.array(reverse_triple)
if (groundings_symmetric!=0).any():
    reverse_triple = unique_rows(reverse_triple)

observed_triples = {}
for triple in test_triples:
    observed_triples[triple[0],triple[1],triple[2]]=1

count = 0
for triple in reverse_triple:
    if (triple[0], triple[1], triple[2]) in observed_triples:
        if (groundings_symmetric!=0).any():
            match_list.append(np.array([triple[0], triple[1], triple[2]]))
        count+=1
    else:
        continue

total_leakage += count
print('# of Symmteric Leakage = ', count)
print('% of symmetric Leakage = ', count/len(test_triples))


#from grounding format h t r1 r2 we should check whether h r2 t exist in the test triples
#Get the number of equivalence leakage
non_reverse_triple = []
for triple in groundings_equivalence:
    #reverse_triple.append(np.array([triple[1],triple[2],triple[0]]))
    non_reverse_triple.append(np.array([triple[0],triple[3],triple[1]]))

non_reverse_triple = np.array(non_reverse_triple)
if (groundings_equivalence!=0).any():
    non_reverse_triple = unique_rows(non_reverse_triple)


observed_triples = {}
for triple in test_triples:
    observed_triples[triple[0],triple[1],triple[2]]=1

count = 0
for triple in non_reverse_triple:
    if (triple[0], triple[1], triple[2]) in observed_triples:
        if (groundings_equivalence!=0).any():
            match_list.append(np.array([triple[0], triple[1], triple[2]]))
        count+=1
    else:
        continue

total_leakage += count
print('# of Equivalence Leakage = ', count)
print('% of Equivalence Leakage = ', count/len(test_triples))

print('# of Total Leakage = ', total_leakage)
print('% of leakage = ', ((total_leakage/len(test_triples))))




#print(len(refined_test_data))
#print(len(np.array(match_list)))
match_list = np.array(match_list).astype(int)
#test_triples_clone = pd.DataFrame(test_triples_clone)
#x = 0
#match_count = 0
for triple in match_list:
    test_triples_clone = np.delete(test_triples_clone, np.where(np.bitwise_and(np.bitwise_and((test_triples_clone[:, 0] == triple[0]), (test_triples_clone[:, 1] == triple[1])), (test_triples_clone[:, 2] == triple[2])))[0], 0)


test_triples_clone = np.array(test_triples_clone)
write_to_txt_file(test_set_refined_dir, test_triples_clone)
print('# of test_triples before removal leakage: ', len(test_triples))
print('# of test_triples after removal leakage: ', len(test_triples_clone))



'''
##########################################
import numpy as np
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

x = np.array([[1,2,3],[1,2,3],
        [4,5,6],
        [7,8,9]])


uniques = unique_rows(non_reverse_triple)
print(len(test_triples))
print(len(uniques))


import numpy as np
unq, cnt = np.unique(test_triples_clone, axis=0, return_counts=True)

y = np.copy(x)

mask_array = (y[:,:]!=[1,2,3]).all(axis=1)
print(np.sum(mask_array[:]==False))
y = y[mask_array]


a = np.array([[1,2],[10,20],[100,200], [1,2]])
a = unique_rows(a)
####
'''

