import seaborn as sns
import matplotlib.pyplot as plt
import  pandas as pd
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/result/train_original.txt', header=None)
data_dbpedia = df.groupby([3, 4], ).agg({1: ['count']}).reset_index()
data_dbpedia.columns = ['year', 'country', 'feature']

# basic scatterplot
#fig, ax = plt.subplots()
interval_size=100
sns.lmplot(x="year", y="feature", data=data_dbpedia, fit_reg=False)
#plot = sns.scatter(x=x_axis, y=y_axis, markers=True,  data=data)
plt.xticks(np.arange(0, data_dbpedia['year'].max()+1, 100))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(interval_size))
plt.show()


# control x and y limits
# plt.plot(range(10))
# plt.ylim(0, None)
# # plt.xscale()
# plt.xlim(100, 2022)
#
# plt.show()

