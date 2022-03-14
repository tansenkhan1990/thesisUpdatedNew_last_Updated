import pandas as pd
import pickle
import  numpy as np
import seaborn as sns
#for Yago
# entity_embedding = np.load('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/mapped/trained_model/entity_embedding_.npy')
# locations = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/result/locations.dict', header=None,dtype=str)
# fifthopole_df = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/yagoUpdated.txt', header=None)
# fifthopole_df.columns = ['subject', 'predicate', 'object', 'time', 'location']
#dbpedia
# entity_embedding = np.load('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/DBPedia5/mapped/trained_model/entity_embedding_.npy')
# locations = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/result/locations.dict', header=None,dtype=str)
# fifthopole_df = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/dbpediaUpdated.txt', header=None)

#wiki
entity_embedding = np.load('/dataset/Wikidata5/result/mapped/trained_model/old2/entity_embedding_.npy')
locations = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/result/locations.dict', header=None,dtype=str)
fifthopole_df = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/wikiUpdated.txt', header=None)

#for mojtauba's changes
fifthopole_df.columns = ['subject', 'predicate', 'object','location', 'time']

# times = times.loc[(times[1]>=1000) & (times[1]<=2022)].reset_index(drop=True)
new_loc = fifthopole_df.groupby(['location'], ).agg({'subject': ['count']})
new_loc.columns = ['count']
new_loc = new_loc.sort_values(by=['count'],ascending=False)
selected_locations = ['United_Kingdom','Germany','Sapin','Poland']
fifthopole_df_specific_locations = fifthopole_df.loc[fifthopole_df['location'].isin(selected_locations)]
# new_loc =  locations.groupby([1], ).agg({1: ['count']}).reset_index()
# new_loc.columns = ['country','iteration']
# poko = new_loc[new_loc['iteration'] >1]
# print(poko)
#time_min = times[1].min()
#time_max = times[1].max()
selected_locations = fifthopole_df_specific_locations['location'].unique()

df =  pd.DataFrame()
for location in selected_locations:
    location_specific_entities_subject = fifthopole_df_specific_locations.loc[fifthopole_df_specific_locations['location']==location]['subject'].unique()
    location_specific_entities_object = fifthopole_df_specific_locations.loc[fifthopole_df_specific_locations['location']==location]['object'].unique()
    location_specific_entities_subject_object =  np.unique([*location_specific_entities_subject,*location_specific_entities_object])
    df = df.append(pd.DataFrame(np.array([location, location_specific_entities_subject_object])).T)
    #print('--------------------------------------')
    #print(location_specific_entities_subject_object)

df = df.reset_index(drop=True)


df.columns = ['category', 'matched_entities']
df.to_pickle('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/location_category_wise_entity.pkl')
