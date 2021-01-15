# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:49:09 2019

@author: 86187
"""

import os
import pandas as pd
import numpy as np
import random
'''
with open("/home/turzo/PycharmProjects/LogicENN_2.0/data/fb15k237/fb15k_rule.txt", "r") as f:
    rules = f.readlines()
'''
data_dir = '/home/mirza/PycharmProjects/frame_work/dataset/wn18_noise_included'
rules = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/wn18_noise_included/wn18_rules_all_thresholded_rule.tsv', header=None).values
relation_df_data_dir = os.path.join(data_dir, 'relations.dict')
relation_df = pd.read_table(relation_df_data_dir, header=None)
relation_dict = dict(zip(relation_df[1], relation_df[0]))
n_rule = len(rules)
simplerules = []
inverse = []
symmetric = []


for rule in rules:
    simplerule = rule[0].split()
    print(simplerule)
    try:
        s = relation_dict[simplerule[1]]
        p = relation_dict[simplerule[5]]
    except:
        continue
    if simplerule[0]==simplerule[4] and simplerule[2]==simplerule[6] and p!=s:
        singlerule = [s,p,0] #implication
        simplerules.append(singlerule)
        print('implication')
    if simplerule[0]==simplerule[6] and simplerule[2]==simplerule[4]and p!=s:
        singlerule = [s,p,1] # inverse
        simplerules.append(singlerule)
        inverse.append(singlerule)
        print('inverse')
    if simplerule[0]==simplerule[6] and simplerule[2]==simplerule[4]and p==s:
        singlerule = [s,p,2] # symmetric
        simplerules.append(singlerule)
        symmetric.append(singlerule)
        print('symmetric')

for i in range(0,len(simplerules)):
    if simplerules[i][2]==0:
        for j in range(i,len(simplerules)):
            if simplerules[j][2]==0 and simplerules[j][0]==simplerules[i][1] and simplerules[j][1]==simplerules[i][0]:
                simplerules[i][2]=3
                simplerules[j][2]=3

inverse=[]
equality=[]
symmetric=[]
implication=[]
n_imp=0
n_inv=0
n_sym=0
n_equ=0
for simplerule in simplerules:
    if simplerule[2]==1:
        inverse.append(simplerule)
        n_inv+=1
    if simplerule[2]==2:
        symmetric.append(simplerule)
        n_sym+=1
    if simplerule[2]==3:
        equality.append(simplerule)
        n_equ+=1
    if simplerule[2] == 0:
        implication.append(simplerule)
        n_imp+=1




implication_data_dir = os.path.join(data_dir, 'rule_implication.txt')
inverse_data_dir = os.path.join(data_dir, 'rule_inverse.txt')
symmetric_data_dir = os.path.join(data_dir, 'rule_symmetric.txt')
equivalence_data_dir = os.path.join(data_dir, 'rule_equivalence.txt')


np.savetxt(implication_data_dir,np.array(implication))
np.savetxt(inverse_data_dir,np.array(inverse))
np.savetxt(symmetric_data_dir,np.array(symmetric))
np.savetxt(equivalence_data_dir,np.array(equality))


simplerules=np.array(simplerules)
mapped_data_dir = os.path.join(data_dir,'mapped')
training_data_dir = os.path.join(mapped_data_dir, 'original_quadropoles.txt')
training_df = pd.read_table(training_data_dir, header=None)
training_triples = list(zip([h for h in training_df[0]],
                            [t for t in training_df[2]],
                            [r for r in training_df[1]]))

imps=[]
imps_all=[]
invs=[]
invs_all=[]
syms=[]
syms_all=[]
equs=[]
equs_all=[]
observed_triples = {}
for triple in training_triples:
    observed_triples[triple[0],triple[1],triple[2]]=1


for triple in training_triples:
        for rule in simplerules:
            if triple[2] == rule[0]:
                 if rule[2]==0:   
                    imp_all = [triple[0],triple[1],rule[0],rule[1]]
                    imps_all.append(imp_all)
                    try:
                        a = observed_triples[triple[0],triple[1],rule[1]]
                    except:
                        imp = [triple[0],triple[1],rule[0],rule[1]]
                        imps.append(imp)
                 if rule[2]==1:   
                    inv_all = [triple[0],triple[1],rule[0],rule[1]]
                    invs_all.append(inv_all)
                    try:
                        a = observed_triples[triple[1],triple[0],rule[1]]
                    except:
                        inv = [triple[0],triple[1],rule[0],rule[1]]
                        invs.append(inv)
                 if rule[2]==2:
                    sym_all = [triple[0],triple[1],rule[0],rule[1]]
                    syms_all.append(sym_all)
                    try:
                        a = observed_triples[triple[1],triple[0],rule[1]]
                    except:
                        sym = [triple[0],triple[1],rule[0],rule[1]]
                        syms.append(sym)
                 if rule[2]==3:
                    equ_all = [triple[0],triple[1],rule[0],rule[1]]
                    equs_all.append(equ_all)
                    try:
                        a = observed_triples[triple[0],triple[1],rule[1]]
                    except:
                        equ = [triple[0],triple[1],rule[0],rule[1]]
                        equs.append(equ)

#                if rule[2]==1:
#                    sym = [triple[0],triple[1],rule[0],rule[1]]
#                    syms.append(sym)
#                if rule[2]==2:
#                    inv = [triple[0],triple[1],rule[0],rule[1]]
#                    invs.append(inv)
#                if rule[2]==3:
#                    double_inv = [triple[0],triple[1],rule[0],rule[1]]
#                    double_invs.append(double_inv)

implication_grounding_data_dir = os.path.join(data_dir, 'groundings_implication.txt')
inverse_grounding_data_dir = os.path.join(data_dir, 'groundings_inverse.txt')
symmetric_grounding_data_dir = os.path.join(data_dir, 'groundings_symmetric.txt')
equivalence_grounding_data_dir = os.path.join(data_dir, 'groundings_equivalence.txt')

np.savetxt(implication_grounding_data_dir,np.array(imps))
np.savetxt(inverse_grounding_data_dir,np.array(invs))
np.savetxt(symmetric_grounding_data_dir,np.array(syms))
np.savetxt(equivalence_grounding_data_dir,np.array(equs))
