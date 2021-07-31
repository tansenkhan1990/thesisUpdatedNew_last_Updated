import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_table('/home/tansen/my_files/thesis_new_files/thesisUpdatedNew/dataset/yago5/result/train', header=None)

data_yago = df
data_yago.columns  = ['subject', 'predicate', 'object','location', 'time']
entities_yago = data_yago.subject.unique()
entities_yago_lenth = len(entities_yago)

relations_yago = data_yago.predicate.unique()
relations_yago_lenth = len(relations_yago)

year_yago = data_yago.predicate.unique()
year_yago_lenth = len(year_yago)

# data = df.groupby([4, 3], ).agg({1: ['count']}).reset_index()
# data.columns = ['year', 'country', 'feature']
# data = data[data.year > 2000]
# #
# # fig = px.scatter(data, x='year', y='feature', color='country',
# #                  width=1200, height=800,
# #                  title="3D Scatter Plot")
# # fig.show()
#
# fig = px.scatter_3d(data, x='year', y='feature', z='country',
#                     color='country',
#                     title="3D Scatter Plot")
# fig.show()