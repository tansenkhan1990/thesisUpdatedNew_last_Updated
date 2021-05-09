import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(style="ticks", color_codes=True)
df = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/result/train_original.txt', header=None)
data = df.groupby([3, 4], ).agg({1: ['count']}).reset_index()
data.columns = ['year', 'country', 'feature']
data = data[(data['year'] > 2000) & (data['year'] < 2015) &
            ((data.country == 'Germany') | (data.country == 'Spain') |
             (data.country == 'Japan') | (data.country == 'United_Kingdom'))]
tips = sns.load_dataset("tips")
sns.catplot(x="year", y="feature",hue="country", kind="swarm", data=data)