import numpy as np
import  pandas as pd

data = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/DBPedia5/result/train' , header=None)
data = data[[0,1,2]]
data.columns = ['head','relation','tail']
unique_relations = data['relation'].unique()

one_to_X_dataset = pd.DataFrame()
X_to_one_dataset = pd.DataFrame()

for relation in unique_relations:
    data_per_relation = data.loc[data['relation'] == relation]
    one_to_X = data_per_relation.groupby('head')['tail'].apply(list).reset_index(name='tail')
    one_to_X['relation'] = relation
    one_to_X_dataset = one_to_X_dataset.append(one_to_X)

for relation in unique_relations:
    data_per_relation = data.loc[data['relation'] == relation]
    X_to_one = data_per_relation.groupby('tail')['head'].apply(list).reset_index(name='head')
    X_to_one['relation'] = relation
    X_to_one_dataset = X_to_one_dataset.append(X_to_one)

X_to_one_dataset = X_to_one_dataset.reset_index(drop=True)
unique_relation_count_tail = [len(np.unique(X_to_one_dataset['head'][i])) for i in range(len(X_to_one_dataset))]
X_to_one_dataset['head_count'] = unique_relation_count_tail

# one_to_many_based_on_tail is actually many to one
one_to_one_based_on_tail = len(X_to_one_dataset.loc[X_to_one_dataset['head_count']==1])
one_to_many_based_on_tail = len(X_to_one_dataset.loc[X_to_one_dataset['head_count']>1])


one_to_X_dataset = one_to_X_dataset.reset_index(drop=True)
unique_relation_count_head = [len(np.unique(one_to_X_dataset['tail'][i])) for i in range(len(one_to_X_dataset))]
one_to_X_dataset['tail_count'] = unique_relation_count_head

#one_to_many_based_on_head  is actually one to many
one_to_one_based_on_head = len(one_to_X_dataset.loc[one_to_X_dataset['tail_count']==1])
one_to_many_based_on_head = len(one_to_X_dataset.loc[one_to_X_dataset['tail_count']>1])


one_to_one_based_on_tail = X_to_one_dataset.loc[X_to_one_dataset['head_count'] == 1]
one_to_one_based_on_head = one_to_X_dataset.loc[one_to_X_dataset['tail_count'] == 1]

one_to_one_first_dataset = pd.DataFrame()
one_to_one_first_dataset['head'] = one_to_one_based_on_tail['head']
one_to_one_first_dataset['tail'] = one_to_one_based_on_tail['tail']
one_to_one_first_dataset['relation'] = one_to_one_based_on_tail['relation']
one_to_one_first_dataset = one_to_one_first_dataset.reset_index(drop=True)
one_to_one_first_dataset['head'] = [one_to_one_first_dataset['head'][i][0] for i in range(len(one_to_one_first_dataset))]


#for head
one_to_one_second_dataset = pd.DataFrame()
one_to_one_second_dataset['head'] = one_to_one_based_on_head['tail']
one_to_one_second_dataset['tail'] = one_to_one_based_on_head['head']
one_to_one_second_dataset['relation'] = one_to_one_based_on_head['relation']
one_to_one_second_dataset = one_to_one_second_dataset.reset_index(drop=True)
one_to_one_second_dataset['head'] = [one_to_one_second_dataset['head'][i][0] for i in range(len(one_to_one_second_dataset))]

merge_for_one_two_one = pd.DataFrame()
merge_for_one_two_one = one_to_one_second_dataset.append(one_to_one_second_dataset)
merge_for_one_two_one = merge_for_one_two_one.drop_duplicates()

# headResult = mainData.groupby('relation')['head'].apply(list).reset_index(name='head')
# tailResult = mainData.groupby('relation')['tail'].apply(list).reset_index(name='tail')
#
# #headResult = mainData.groupby('relation')['head'].apply(list).reset_index(name='relation')
# #tailResult = mainData.groupby('tail')['relation'].apply(list).reset_index(name='tail')
#
#
# #head
# unique_relation_count_head = [len(np.unique(headResult['relation'][i])) for i in range(len(headResult))]
# headResult['unique_relations'] = unique_relation_count_head
#
# one_to_one_based_on_head = len(headResult.loc[headResult['unique_relations']==1])
# one_to_many_based_on_head = len(headResult.loc[headResult['unique_relations']>1])
#
#
# #tail
# unique_relation_count_tail = [len(np.unique(tailResult['relation'][i])) for i in range(len(tailResult))]
# tailResult['unique_relations'] = unique_relation_count_tail
#
# one_to_one_based_on_tail = len(tailResult.loc[tailResult['unique_relations']==1])
# one_to_many_based_on_tail = len(tailResult.loc[tailResult['unique_relations']>1])