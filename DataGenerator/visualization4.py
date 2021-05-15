import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/result/train_original.txt', header=None)
data_dbpedia = df1.groupby([3, 4], ).agg({1: ['count']}).reset_index()
data_dbpedia.columns = ['year', 'country', 'feature']
data_dbpedia['dataset'] = 'dbpedia'

df2 = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/wikidata/result/train_original.txt', header=None)
data_wiki = df2.groupby([3, 4], ).agg({1: ['count']}).reset_index()
data_wiki.columns = ['year', 'country', 'feature']
data_wiki['dataset'] = 'wiki'
# data_wiki = data_wiki[data_wiki.year > 2000]
# data_dbpedia = data_dbpedia[data_dbpedia.year > 2000]
mainDataSet = pd.merge(data_dbpedia, data_wiki, how='outer')

# adjust coordinates
x1 = data_dbpedia['year']
y1 = data_dbpedia['feature']

x2 = data_wiki['year']
y2 = data_wiki['feature']

# depict illustration
plt.scatter(x1, y1)

plt.scatter(x2, y2)

# apply legend()
plt.legend(['dbpedia', 'wiki'])
plt.xlabel('years')
plt.ylabel('feature')
plt.show()

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import matplotlib.pyplot as plt
# import seaborn as sns

# df1 = pd.read_table('/content/sample_data/train_original.txt', header=None)
# data_dbpedia = df1.groupby([3, 4], ).agg({1: ['count']}).reset_index()
# data_dbpedia.columns = ['year', 'country', 'feature']
# data_dbpedia['dataset'] = 'dbpedia'
# # data_dbpedia = data_dbpedia[data_dbpedia.year > 2000]
# #df[4] = df[4].map(str) + '_dbpedia'

# df2 = pd.read_table('/content/sample_data/wiki/train_original.txt', header=None)
# data_wiki = df2.groupby([3, 4], ).agg({1: ['count']}).reset_index()
# data_wiki.columns = ['year', 'country', 'feature']
# data_wiki['dataset'] = 'wiki'
# data_wiki = data_wiki[data_wiki.year > 2000]
# data_dbpedia = data_dbpedia[data_dbpedia.year > 2000]
# mainDataSet = pd.merge(data_dbpedia, data_wiki, how='outer')

# sns.set_theme(style="ticks", color_codes=True)

# sns.catplot(x="year", y="feature", hue = 'dataset', data=mainDataSet)
