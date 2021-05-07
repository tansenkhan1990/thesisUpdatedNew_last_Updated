import pandas as pd
import numpy as np
df = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/result/train_original.txt',
                   header= None)
# df = pd.read_csv('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/result/first_100_columns/first_100_columns_dbpedia.csv', header= None)
# df.set_index(df.index)
# print(df)
df2 = df.groupby([3,4],).agg({1:['count']}).reset_index()

