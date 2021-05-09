import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/result/train_original.txt', header=None)
data_dbpedia = df1.groupby([3, 4], ).agg({1: ['count']}).reset_index()
data_dbpedia.columns = ['year', 'country', 'feature']
data_dbpedia['dataset'] = 'dbpedia'
# data_dbpedia = data_dbpedia[data_dbpedia.year > 2000]
#df[4] = df[4].map(str) + '_dbpedia'

df2 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/result/train_original.txt', header=None)
data_wiki = df2.groupby([3, 4], ).agg({1: ['count']}).reset_index()
data_wiki.columns = ['year', 'country', 'feature']
data_wiki['dataset'] = 'wiki'
data_wiki = data_wiki[data_wiki.year > 1900]
data_dbpedia = data_dbpedia[data_dbpedia.year > 1900]
mainDataSet = pd.merge(data_dbpedia, data_wiki, how='outer')

sns.set_theme(style="ticks", color_codes=True)
sns.catplot(x="year", y="feature", hue = 'dataset', data=mainDataSet)
#sns.catplot(x="year", y="type", hue="type", kind="swarm", data=mainDataSet)

#df2[4] = df2[4].map(str) + '_wikipedia'
# data1.columns = ['year', 'country', 'location_count']
# data1['country'] = data1['country'].map(str) + '_dbpedia'
# data1 = data1[data1.year > 2017]
# dbpedia_year = data_dbpedia.year
# dbpedia_feature = data_dbpedia.feature
# # plt.scatter(dbpedia_year, dbpedia_feature , c = 'blue' )
#
#
# wiki_year = data_wiki.year
# wiki_feature = data_wiki.feature
# plt.scatter(wiki_year, wiki_feature , c = '#88c999')
# plt.scatter(dbpedia_year, dbpedia_feature , c = 'blue')
# plt.title('wikidata vs dbpedia')
# plt.xlabel(' years ')
# plt.ylabel(' features ')
# # plt.legend('dbpedia', 'wikidata')
# plt.legend(['blue', '#88c999'], ["dbpedia", "wiki"])
# plt.show()


#
# import matplotlib.pyplot as plt
# db_year = data_dbpedia.year
# db_feature = data_dbpedia.feature
# wiki_year = data_wiki.year
# wiki_feature = data_wiki.feature
# fig=plt.figure()
# ax=fig.add_axes([0,0,1,1])
# ax.scatter(db_year, db_feature, color='r')
# ax.scatter(wiki_year, wiki_feature, color='b')
# ax.set_xlabel('Year')
# ax.set_ylabel('Feature')
# ax.set_title('scatter plot')
# plt.show()