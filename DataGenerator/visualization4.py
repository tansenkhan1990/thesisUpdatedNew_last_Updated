import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/result/train', header=None)

#test
data_yago = df1.groupby([3, 0])

tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=1000)
tsne_result = tsne.fit_transform(dim_reduced)



c_palate = {}
colors = ['tab:blue','tab:orange','tab:green','tab:red',
          'tab:purple','tab:brown', 'tab:pink','tab:gray','tab:olive','c'
,'m']
#colors = list(np.random.choice(range(256), size=11))


plt.figure(figsize=(16,16))
plt.xlabel('tsne-one', fontsize=20)
plt.ylabel('tsne-two', fontsize=20)

plot = sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue='type',
    data=tsne_df,
    palette=c_palate,
    legend="full",
    alpha=0.8
)


plt.setp(plot.get_legend().get_texts(), fontsize='20')
plt.show()


#end test

data_yago.columns = ['year', 'country']
data_yago['dataset'] = 'yago'
data_yago = data_yago[(data_yago.year > 1000) & (data_yago.year<2022)]

data_yago = df1.groupby([4, 3], ).agg({1: ['count']}).reset_index()
data_yago.columns = ['year', 'country', 'feature']
data_yago['dataset'] = 'yago'
data_yago = data_yago[(data_yago.year > 1000) & (data_yago.year<2022)]

df2 = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/result/train', header=None)
data_wiki = df2.groupby([4, 3], ).agg({1: ['count']}).reset_index()
data_wiki.columns = ['year', 'country', 'feature']
data_wiki['dataset'] = 'wikidata'
data_wiki = data_wiki[(data_wiki.year > 1000) & (data_wiki.year<2022)]

df3 = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/DBPedia5/result/train', header=None)
dbpedia = df3.groupby([4, 3], ).agg({1: ['count']}).reset_index()
dbpedia.columns = ['year', 'country', 'feature']
dbpedia['dataset'] = 'dbpedia'
dbpedia = dbpedia[(dbpedia.year > 1000) & (dbpedia.year<2022)]

# data_wiki = data_wiki[data_wiki.year > 2000]
# data_dbpedia = data_dbpedia[data_dbpedia.year > 2000]
# mainDataSet = pd.merge(data_dbpedia, data_wiki, how='outer')
# mainDataSet = mainDataSet[(mainDataSet.year > 1000) & (mainDataSet.year<2022)]
# adjust coordinates



x1 = data_yago['year']
y1 = data_yago['feature']

x2 = data_wiki['year']
y2 = data_wiki['feature']

x3 = dbpedia['year']
y3 = dbpedia['feature']
# depict illustration

plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels

plt.xlabel('xlabel', fontsize=20)
plt.ylabel('ylabel', fontsize=20)

plt.scatter(x2, y2)
plt.scatter(x3, y3)
plt.scatter(x1, y1)
# plt.scatter(x1, y1)


# apply legend()
# plt.legend(fontsize=20)
plt.legend(['YAGO','WikiData','DBpedia'],prop={"size":20} )
plt.xlabel('years')
plt.ylabel('feature( Entities and Relations)')
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
