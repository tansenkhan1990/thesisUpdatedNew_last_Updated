import numpy as np
import pandas as pd

def build_new_data(path):
    data=pd.read_table(path)
    data = data[(data['Std Confidence'] >= 0.7) & (data['Std Confidence']<=1.0)]
    #rules = data.loc[:,data.columns.isin(['Rule','Std Confidence'])]
    rules = data.loc[:, data.columns.isin(['Rule', 'Std Confidence'])]
    #rules = rules['Rule'].astype(str) + '   ' +rules['Std Confidence'].astype(str)
    rules = rules['Rule'].astype(str) + '   ' + rules['Std Confidence'].astype(str)
    #rules = rules.drop(['Confidence'], axis=1)
    print(rules)
    print(len(data))
    return rules

data_dir = '/home/mirza/PycharmProjects/frame_work/dataset/kinship'
file_name = '/kinship_train_rules_all_thresholded_'
data_new_data =build_new_data(data_dir + '/kinship_all_rules.txt').reset_index()
data_new_data = data_new_data.drop(['index'], axis=1)
data_new_data.to_csv(data_dir+ file_name+'rule.tsv',index=False,sep='\t', header=None)

print('now go to grounding_creation.py')

