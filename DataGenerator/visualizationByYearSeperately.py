import seaborn as sns
import matplotlib.pyplot as plt
import  pandas as pd
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/Wikidata5/result/train', header=None)

data_dbpedia = df.groupby([4, 3], ).agg({1: ['count']}).reset_index()
data_dbpedia.columns = ['Year', 'country', 'feature (Entities Relations)']
data_dbpedia = data_dbpedia[(data_dbpedia.Year > 1000) & (data_dbpedia.Year<2022)]


interval_size=100
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels

sns.lmplot(x="Year", y="feature (Entities Relations)", data=data_dbpedia, fit_reg=False)
#plot = sns.scatter(x=x_axis, y=y_axis, markers=True,  data=data)
plt.xticks(np.arange(0, data_dbpedia['Year'].max()+1, 100))
plt.xlabel('Year', fontsize=20)
plt.ylabel('feature (Entities and  Relations)', fontsize=20)
plt.xlim([1000, 2022])
#ax.xaxis.set_major_locator(ticker.MultipleLocator(interval_size))
# plt.grid()
plt.show()



