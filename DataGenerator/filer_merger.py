import numpy as np
import pandas as pd

#for wiki data
# data_1 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/label/books-wd-label', header=None)
# data_2 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/label/crimes-wd-label', header=None)
# data_3 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/label/movies-wd-label', header=None)
# data_4 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/label/music-wd-label', header=None)
# data_5 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/label/paintings-wd-label', header=None)
# data_6 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/label/physicists-awards-wd-label', header=None, sep=' ')

#for dbpedia dataset

data_1 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/battles', header=None)
data_2 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/books', header=None)
data_3 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/buildings', header=None)
data_4 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/movies', header=None)
data_5 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/music', header=None)
data_6 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/space-missions', header=None, sep=' ')


df_ultimate = pd.DataFrame()

df_ultimate = df_ultimate.append(data_1).reset_index(drop=True)
df_ultimate = df_ultimate.append(data_2).reset_index(drop=True)
df_ultimate = df_ultimate.append(data_3).reset_index(drop=True)
df_ultimate = df_ultimate.append(data_4).reset_index(drop=True)
df_ultimate = df_ultimate.append(data_5).reset_index(drop=True)
df_ultimate = df_ultimate.append(data_6).reset_index(drop=True)

df_ultimate = np.array(df_ultimate).astype(str)
df_ultimate = np.unique(df_ultimate, axis=0)
df_ultimate = pd.DataFrame(df_ultimate)

df_ultimate[0] = df_ultimate[0].str.replace('"','')
df_ultimate[1] = df_ultimate[1].str.replace('"','')
df_ultimate[2] = df_ultimate[2].str.replace('"','')
df_ultimate[3] = df_ultimate[3].str.replace('"','')
df_ultimate[4] = df_ultimate[4].str.replace('"','')

#df_ultimate[2] = df_ultimate[2].map(lambda x: x.lstrip('\"\"\"').rstrip(''))
# for wikipedia
#pd.DataFrame(df_ultimate).to_csv( '/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/wikidata.txt',index = False, sep= '\t', header=None)
#for dbpedia 
pd.DataFrame(df_ultimate).to_csv( '/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/dbpedia.txt',index = False, sep= '\t', header=None)