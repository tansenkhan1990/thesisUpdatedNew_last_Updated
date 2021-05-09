import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/result/train_original.txt', header=None)
data = df.groupby([3, 4], ).agg({1: ['count']}).reset_index()
data.columns = ['year', 'country', 'feature']
data = data[data.year > 2000]
#
# fig = px.scatter(data, x='year', y='feature', color='country',
#                  width=1200, height=800,
#                  title="3D Scatter Plot")
# fig.show()

fig = px.scatter_3d(data, x='year', y='feature', z='country',
                    color='country',
                    title="3D Scatter Plot")
fig.show()