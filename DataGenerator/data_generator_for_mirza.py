import pandas as pd
import numpy as np

columns = ['subject', 'predicate', 'object','location', 'time']

yago = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/yagoUpdated.txt', header=None)
yago.columns = columns
yago_sub = yago['subject'].unique()
yago_sub_len = len(yago_sub)

yago_pre = yago['predicate'].unique()
yago_pre_len = len(yago_pre)

yago_obj = yago['object'].unique()
yago_obj_len = len(yago_obj)

all_sub = list(yago['subject'].values)
all_obj = list(yago['object'].values)
unique_entities =  np.unique([*all_sub,*all_obj])

yago_loc= yago['location'].unique()
yago_loc_len = len(yago_loc)


yago_time = yago['time'].unique()
yago_time_len = len(yago_time)

#wiki data
columns = ['subject', 'predicate', 'object', 'time', 'location']
wiki = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/wikidata/wikidata.txt', header=None)

wiki.columns = columns

all_sub = list(yago['subject'].values)
all_obj = list(yago['object'].values)
unique_entities =  np.unique([*all_sub,*all_obj])

wiki_sub = wiki['subject'].unique()
wiki_sub_len = len(wiki_sub)

wiki_pre = wiki['predicate'].unique()
wiki_pre_len = len(wiki_pre)

wiki_obj = wiki['object'].unique()
wiki_obj_len = len(wiki_obj)

wiki_loc= wiki['location'].unique()
wiki_loc_len = len(wiki_loc)


wiki_time = wiki['time'].unique()
wiki_time_len = len(wiki_time)

#dbpedia

columns = ['subject', 'predicate', 'object', 'time', 'location']
dbpedia = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/dbpediadata/dbpedia.txt', header=None)
dbpedia.columns = columns
dbpedia_sub = dbpedia['subject'].unique()
dbpedia_sub_len = len(dbpedia_sub)

dbpedia_pre = dbpedia['predicate'].unique()
dbpedia_pre_len = len(dbpedia_pre)

dbpedia_obj = dbpedia['object'].unique()
dbpedia_obj_len = len(dbpedia_obj)

dbpedia_loc= dbpedia['location'].unique()
dbpedia_loc_len = len(dbpedia_loc)


dbpedia_time = dbpedia['time'].unique()
dbpedia_time_len = len(dbpedia_time)
