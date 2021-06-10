import seaborn as sns
import matplotlib.pyplot as plt
import  pandas as pd
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/result/train', header=None)

data_dbpedia = df.groupby([4, 3], ).agg({1: ['count']}).reset_index()
data_dbpedia.columns = ['year', 'country', 'feature']
data_dbpedia = data_dbpedia[(data_dbpedia.year > 1000) & (data_dbpedia.year<2022)]

interval_size=100
sns.lmplot(x="year", y="feature", data=data_dbpedia, fit_reg=False)
#plot = sns.scatter(x=x_axis, y=y_axis, markers=True,  data=data)
plt.xticks(np.arange(0, data_dbpedia['year'].max()+1, 100))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(interval_size))
plt.grid()
plt.show()



