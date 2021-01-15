import numpy as np
import pandas as pd


data = pd.read_table('/media/rony/Feel Free/Others/Temp/jan25_rule_learning/rule_learning_framework-master/dataset/fb15k/fb15k_triples.train',header=None)


to_find = data.loc[data[1] == '/film/film_subject/films']
to_find = np.array(to_find)
to_match = data.loc[data[1] == '/film/film/subjects']
to_match =np.array(to_match)
for i in range(len(to_find)):
    triple = to_find[i]
    triple_subject = triple[0]
    triple_relation = triple[1]
    triple_subject2 = triple[2]
    to_be_match_triple = np.array([triple_subject2, triple_relation, triple_subject])
    #print(triple[i])
    for j in range(len((to_match))):
        to_matched = to_match[j]
        subject = to_matched[0]
        subject2 = to_matched[2]
        relation = to_matched[1]
        matching_triple = np.array([subject, relation, subject2])
        #print(matching_triple)
        if (all(to_be_match_triple == matching_triple)):
            print(to_be_match_triple + 'symmetric implies' + matching_triple)

# A = np.array([3,4,5])
# B = np.array([4,4,5])
#
# if all((A == B)):
#     print('Array match')
# else:
#     print('Array doesnot match')