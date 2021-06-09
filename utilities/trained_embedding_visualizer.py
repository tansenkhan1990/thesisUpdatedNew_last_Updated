import pandas as pd
import numpy as np
import itertools
from sklearn.cluster import KMeans
import pprint
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from adjustText import adjust_text
from sklearn.manifold import TSNE

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def original_index_to_type(data, value_to_converted):

    converted = []
    data = dict(zip(data[0], data[1]))

    for d in value_to_converted:
        converted.append(data[str(d)])
    return np.array(converted)

def checkIfDuplicates_1(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        print('does not contain duplicates')
    else:
        print('contain duplicates')

def prepare(type_A):
    a = []
    for i in type_A[0]:
        a.append(i)
    return np.array(a)

def entity_to_id_conversion(entities):
    id_converted = []
    ###Should be changed for both types 0 1 for sem
    entity_to_id_dict = dict(zip(entity_to_id[1], entity_to_id[0]))

    for entity in entities:
        id_converted.append(entity_to_id_dict[entity])
    return np.array(id_converted)

def id_to_entity_conversion(entity_id):
    entity_converted = []
    ###Should be changed for both types 0 1 for sem
    id_to_entity_dict = dict(zip(entity_to_id[0], entity_to_id[1]))

    for id in entity_id:
        entity_converted.append(id_to_entity_dict[id])
    return np.array(entity_converted)

#Read trained embeddings
entity_embedding = np.load('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/mapped/trained_model/entity_embedding_.npy')

#min_max_scaler = preprocessing.MinMaxScaler()
#entity_embedding = pd.DataFrame(min_max_scaler.fit_transform(entity_embedding))
relation_embedding = np.load('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/mapped/trained_model/relation_embedding.npy')
#Read the dictionary for original text
entity_to_id = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/result/entities.dict', header=None)
#entity_to_id= entity_to_id[[1, 0]]

#entity_to_id = pd.DataFrame(entity_to_id.T)

relation_to_id = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/result/relations.dict', header=None)
#relation_to_id= relation_to_id[[1, 0]]

entity_embedding = pd.DataFrame(entity_embedding)
#for time
#stats = pd.read_pickle('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/time_category_wise_entity.pkl')
#for locations
stats = pd.read_pickle('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/location_category_wise_entity.pkl')

#Best indicated types
#'/education/educational_institution', '/book/author', '/film/film'

#indicated_types = ['/education/educational_institution', '/book/author', '/film/film', '/location/location']
#indicated_types = ['/education/educational_institution', '/book/author' ,'/film/film', '/tv/tv_actor',  '/tv/tv_program', '/music/instrument', '/travel/travel_destination', '/business/employer']
# indicated_types = ['/film/film'  , '/tv/tv_program', '/music/instrument',
#                    '/organization/endowed_organization'
#                    ,'/people/person', '/music/group_member']
indicated_types = stats['category'].values
ids = []
types_with_ids = []

for i in range(len(indicated_types)):
    type_ = np.array(stats.loc[stats['category'] == indicated_types[i]]['matched_entities'])
    type_prepared = prepare(type_)
    type_ids = entity_to_id_conversion(type_prepared)
    type_ids_type = np.repeat(indicated_types[i], len(type_ids))
    type_combined = np.c_[type_ids, type_ids_type]
    ids.append(type_ids)
    types_with_ids.append(type_combined)

ids = tuple(ids)
ids = (list(np.concatenate(ids)))
#total_ids = set(list(np.concatenate(ids)))
total_ids = unique(ids)
com = np.vstack(np.array(types_with_ids))



#total_ids = (list(np.concatenate(args)))
#total_ids = unique(total_ids)
#com = np.vstack( [type_A_combined , type_B_combined, type_C_combined, type_D_combined, type_E_combined, type_F_combined] )

checkIfDuplicates_1(total_ids)

dim_reduced = entity_embedding.loc[total_ids]
original_index = dim_reduced.index

tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
tsne_result = tsne.fit_transform(dim_reduced)

tsne_df = pd.DataFrame()

tsne_df['tsne-one'] = tsne_result[:,0]
tsne_df['tsne-two'] = tsne_result[:,1]
#tsne_df['tsne-three'] = tsne_result[:,2]
tsne_df['original_index'] = original_index

type = original_index_to_type(pd.DataFrame(com), tsne_df['original_index'])
tsne_df['type'] =type

tsne_df['original_entity'] = id_to_entity_conversion(tsne_df['original_index'].values)

# import plotly.express as px
#
# fig = px.scatter_3d(tsne_df, x='tsne-one', y='tsne-two', z='tsne-three',
#               color='type')
# fig.show()
c_palate = {}
colors = ['tab:blue','tab:orange','tab:green','tab:red',
          'tab:purple','tab:brown', 'tab:pink','tab:gray','tab:olive','c'
,'m']
#colors = list(np.random.choice(range(256), size=11))
for i,v in zip(indicated_types, colors):
    #print(indicated_types[i])
    c_palate[i] = v


plt.figure(figsize=(16,16))
plot = sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue='type',
    data=tsne_df,
    palette=c_palate,
    legend="full",
    alpha=0.8
)
plt.setp(plot.get_legend().get_texts(), fontsize='15')
plt.show()
