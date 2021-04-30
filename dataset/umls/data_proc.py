import numpy as np
import pandas as pd


data = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/umls/UML.tsv')

des_str = 'sub:' + str(62)

inter_data = data[des_str]
inter_data = np.array(inter_data)
inter_data = [i-1 for i in inter_data if ~np.isnan(i)]
inter_data = np.array(inter_data)
x = inter_data[~np.isnan(inter_data)]

x = pd.DataFrame(x)
x.index = x


all_entity = np.arange(0, 5, 1)

