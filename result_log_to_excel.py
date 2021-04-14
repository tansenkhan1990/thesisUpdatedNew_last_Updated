import os
import re
import numpy as np
import json
import pandas as pd

data_dir = '/home/tansen/my_files/thesisUpdatedNew/dataset/kinship/tansen_results'

df = pd.DataFrame()
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith("txt"):
            file_contents = pd.read_table(os.path.join(root, file), header=None, sep=':')
            if len(file_contents) != 0:
                file_contents.columns = ['metric', 'value']
                temp_df = pd.DataFrame(file_contents['value'].values).T
                df = df.append(temp_df)

df.columns = columns= ['dims', 'epoch', 'gamma', 'lr', 'batch_size',
                            'temp', 'Mean Rank(F)', 'Mean RR(F)' , 'Hit@1(F)', 'Hit@3(F)', 'Hit@5(F)', 'Hit@10(F)']
df.to_excel(os.path.join(data_dir,'summarized.xlsx'), index=False)
