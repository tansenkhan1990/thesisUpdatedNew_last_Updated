import pandas as pd
import pickle
import  numpy as np
import seaborn as sns
entity_embedding = np.load('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/result/mapped/trained_model/entity_embedding_.npy')
times = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/result/times.dict', header=None,dtype=float)
locations = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/result/locations.dict', header=None,dtype=str)
fifthopole_df = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/wikiUpdated.txt', header=None)
#For Wiki
#fifthopole_df = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/wikiUpdated.txt', header=None)
# fifthopole_df.columns = ['subject', 'predicate', 'object', 'time', 'location']
#for mojtauba's changes
fifthopole_df.columns = ['subject', 'predicate', 'object','location', 'time']
#for yago5
# df_1 = fifthopole_df.loc[(fifthopole_df['time']>=1990) & (fifthopole_df['time']<=1900)].reset_index(drop=True)
# df_2 = fifthopole_df.loc[(fifthopole_df['time']>=1600) & (fifthopole_df['time']<=1700)].reset_index(drop=True)
# df_3 = fifthopole_df.loc[(fifthopole_df['time']>=1999) & (fifthopole_df['time']<=2009)].reset_index(drop=True)
# df_4 = fifthopole_df.loc[(fifthopole_df['time']>=1933) & (fifthopole_df['time']<=1943)].reset_index(drop=True)
# df_5 = fifthopole_df.loc[(fifthopole_df['time']>=1955) & (fifthopole_df['time']<=1965)].reset_index(drop=True)

#for Dbpedia
df_1 = fifthopole_df.loc[(fifthopole_df['time']>=1951) & (fifthopole_df['time']<=1990)].reset_index(drop=True)
df_2 = fifthopole_df.loc[(fifthopole_df['time']>=1990) & (fifthopole_df['time']<=2000)].reset_index(drop=True)
df_3 = fifthopole_df.loc[(fifthopole_df['time']>=2000) & (fifthopole_df['time']<=2010)].reset_index(drop=True)
df_4 = fifthopole_df.loc[(fifthopole_df['time']>=2000) & (fifthopole_df['time']<=2020)].reset_index(drop=True)
# df_4 = fifthopole_df.loc[(fifthopole_df['time']>=1933) & (fifthopole_df['time']<=1943)].reset_index(drop=True)
# df_5 = fifthopole_df.loc[(fifthopole_df['time']>=1955) & (fifthopole_df['time']<=1965)].reset_index(drop=True)

#
# time_dfs = [df_1, df_3, df_4, df_5]
time_dfs = [df_1,df_2,df_3,df_4]
entities_in_time_category = pd.DataFrame()
for df in time_dfs:
    range_start = df['time'].min()
    range_end = df['time'].max()
    subject_within_time_range = list(df.loc[(df['time']>=range_start) & (df['time']<=range_end)]['subject'].unique())
    object_within_time_range = list(df.loc[(df['time']>=range_start) & (df['time']<=range_end)]['object'].unique())
    subject_object_within_time_range = np.unique([*subject_within_time_range,*object_within_time_range])
    time_category = str(range_start) + '-' + str(range_end)
    entities_in_time_category = entities_in_time_category.append(pd.DataFrame(np.array([time_category, subject_object_within_time_range])).T)

entities_in_time_category = entities_in_time_category.reset_index(drop=True)

entities_in_time_category.columns = ['category', 'matched_entities']
# new_time_category = entities_in_time_category['']
#For Yago
# entities_in_time_category.to_pickle('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/time_category_wise_entity.pkl')
# For Wikidata5
#entities_in_time_category.to_pickle('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/time_category_wise_entity.pkl')
#for wiki
entities_in_time_category.to_pickle('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/time_category_wise_entity.pkl')
# # times = times.loc[(times[1]>=1900) & (times[1]<=2022)].reset_index(drop=True)
# times = pd.concat([times1,times2,times3,times4,times5],ignore_index=True)
# times = times.reset_index(drop=True)
# # times = times.loc[(times[1]>=1909) & (times[1]<=2012)].reset_index(drop=True)
# # new_loc =  locations.groupby([1], ).agg({1: ['count']}).reset_index()
# # new_loc.columns = ['country','iteration']
# # poko = new_loc[new_loc['iteration'] >1]
# # print(poko)
# time_min = times[1].min()
# time_max = times[1].max()
#
# #sns.histplot(times[1])
#
# range_start = time_min
# df_time_category = pd.DataFrame()
# while range_start<=time_max:
#     range_end = range_start + 10
#     if range_end>=time_max:
#         break
#     # range_end = range_start + 20
#     df_time_category = df_time_category.append(pd.DataFrame(np.array([range_start, range_end])).T)
#     range_start = range_end + 1
#     #print(i)
# df_time_category = df_time_category.reset_index(drop=True)
# # df = pd.DataFrame()
# # for times in times_category:
# #     small_df = df.loc[(df['times']>=times[0]) | (df['times']<=times[1])]
# #     enetities_list_subject = list(np.unique(small_df['subject'].values))
# #     enetities_list_object = list(np.unique(small_df['subject'].values))
# #     entities_sub_obj = enetities_list_subject + enetities_list_object
# #     temp = np.array([times,entities_sub_obj])
# #     arr = np.vstack(arr,temp)
#
# entities_in_time_category = pd.DataFrame()
# # category_time = '1912.0' + '-' + '1932.0'
# for i in df_time_category.index:
#    #print(df_time_category.loc[i,0], df_time_category.loc[i,1])
#    start_time_category = df_time_category.loc[i,0]
#    end_time_category = df_time_category.loc[i,1]
#    #print(start_time_category)
#    #print(end_time_category)
#    small_df = fifthopole_df.loc[(fifthopole_df['time'] >= start_time_category) & (fifthopole_df['time'] <= end_time_category)]
#    entities_list_subject = list(np.unique(small_df['subject'].values))
#    entities_list_object = list(np.unique(small_df['object'].values))
#    #print(entities_list_subject)
#    #print(entities_list_object)
#    joint_list_sub_obj = np.unique([*entities_list_subject,*entities_list_object])
#    #print(small_df)
#    #for fixed time category
#    ##########
#    print('######time category#####')
#
#    category_time = str(start_time_category) + '-' + str(end_time_category)
#    print(category_time)
#    entities_per_category_time = np.array([category_time, joint_list_sub_obj])
#    entities_in_time_category = entities_in_time_category.append(pd.DataFrame(entities_per_category_time).T)
#    #print(joint_list_sub_obj)
#    print('#############################')
# entities_in_time_category.columns = ['category', 'matched_entities']
# # new_time_category = entities_in_time_category['']
# entities_in_time_category.to_pickle('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/time_category_wise_entity.pkl')
