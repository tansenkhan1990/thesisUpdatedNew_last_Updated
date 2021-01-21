# # import numpy as np
# #
# #
# # f = open('workfile', 'w')
# # for i in range(3):
# #     f.write('\n')
# #     stra = str(i) + '\t' + str(i+2)
# #     s = str(stra)
# #     f.write(s)
# #
# # f.close()
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# data = pd.read_table('/home/mirza/PycharmProjects/LogicENN filtered/fb15k_new/test_1converted_workfile', header=None)
# import os
#
# #plt.plot(data[0], label='total')
# #plt.plot(data[2], label='implication')
# plt.plot(data[3], label='inverse')
# #plt.plot(data[4], label='symmetric')
# #plt.plot(data[5], label='equivalence')
# #plt.plot(data[6], label='complex')
#
# plt.legend()
# plt.show()

# Chengjins graph

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data_1 = pd.read_table('/media/rony/Feel Free/Others/Temp/jan25_rule_learning/rule_learning_framework-master/dataset/FB15k_noise_included/mloss_per_33epochs.txt', header=None)
#data_2 = pd.read_table('/home/mirza/PycharmProjects/LogicENN filtered/rule_loss_noconv', header=None)

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 10))
# plt.plot(data_1[0], label='total_loss')
# plt.plot(data_1[1], label='implication_loss')
# plt.plot(data_1[2], label='inverse_loss')
# plt.plot(data_1[3], label='symmetric_loss')
# plt.plot(data_1[2], label='equivalence')
axes[0].plot(data_1[0], label='total')
#axes[0][1].plot(data_2[1], label='implication (Without injected rules)')
axes[1].plot(data_1[1], label='implication')
#axes[1][1].plot(data_2[2], label='inverse(Without injected rules)')
axes[2].plot(data_1[2], label='inverse')
#axes[2][1].plot(data_2[3], label='symmetric (Without injected rules)')
axes[3].plot(data_1[3], label='symmetric')
#axes[3][1].plot(data_2[4], label='equivalence(Without injected rules)')
axes[4].plot(data_1[4], label='equivalence')
#axes[4][1].plot(data_2[5], label='complex (Without injected rules)')
#plt.plot(data[5], label='complex')

axes[0].legend(loc='upper right',prop={'size': 10})
#axes[0][1].legend(loc='upper right',prop={'size': 10})
# axes[0][0].set_xlim([0,1.5])
# axes[0][1].set_xlim([0,1.5])
axes[0].set_ylim([0,2])
#axes[0][1].set_ylim([0,2])
axes[1].legend(loc='upper right',prop={'size': 10})
#axes[1][1].legend(loc='upper right',prop={'size': 10})
axes[1].set_ylim([0,2])
#axes[1][1].set_ylim([0,2])
axes[2].legend(loc='upper right',prop={'size': 10})
#axes[2][1].legend(loc='upper right',prop={'size': 10})
axes[2].set_ylim([0,4])
#axes[2][1].set_ylim([0,1])
axes[3].legend(loc='upper right',prop={'size': 10})
#axes[3][1].legend(loc='upper right',prop={'size': 10})
axes[3].set_ylim([0,5])
#axes[3][1].set_ylim([0,15])
axes[4].legend(loc='upper right',prop={'size': 10})
#axes[4][1].legend(loc='upper right',prop={'size': 10})
axes[4].set_ylim([0,5])
#axes[4][1].set_ylim([0,0.005])

# Set common labels
fig.text(0.5, 0.04, 'Training Steps', ha='center', va='center')
fig.text(0.06, 0.5, 'Loss', ha='center', va='center', rotation='vertical')
plt.show()
#data = pd.read_table('/home/mirza/PycharmProjects/LogicENN filtered/kinship/hpt_kinship', header=None)
#prob_table = pd.read_table(os.path.join('/home/mirza/PycharmProjects/LogicENN filtered/kinship', 'hpt_kinship'), names=['tph', 'hpt'],
#                               header=None)

import pandas as pd
import numpy as np
data = np.loadtxt('/home/mirza/PycharmProjects/frame_work/dataset/kinship/groundings_symmetric.txt')
data1 = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/kinship/groundings_symmetric.txt', header=None, sep=' ')
data2 = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/kinship/groundings_symmetric_test.txt', header=None)

model = pd.read_pickle('/home/mirza/PycharmProjects/frame_work/dataset/kinship/trained_model/params49.pkl')

import pandas as pd
import numpy as np
import torch

# listA = torch.tensor([[1, 2, 3],[2,7,8],[5,8,6]])
# listB = torch.tensor([[4, 5, 6],[4,7,9],[5,8,0]])
# listC = torch.tensor([[7,8,10],[4,7,9],[6,8,9]])
# listD = torch.tensor([[7,8,10],[4,7,9],[6,8,9]])
# Given lists
import numpy as np

listA = np.array([[1, 2, 3],[2,7,8],[5,8,6]])
listB = np.array([[4, 5, 6],[4,7,9],[5,8,0]])
listC = np.array([[7,8,10],[4,7,9],[6,8,9]])
listD = np.array([[7,8,10],[4,7,9],[6,8,9]])

listA = torch.from_numpy(listA)
listB = torch.from_numpy(listB)
listC = torch.from_numpy(listC)
listD = torch.from_numpy(listD)

print("Given list A: ", listA)
print("Given list B: ",listB)
print("Given list B: ",listC)
print("Given list B: ",listD)
# Use map
#res = list(map(lambda(i, j, k): [i , j, k], zip(listA, listB, listC)))
res = [[i,j,k, l] for i, j, k, l in zip(listA, listB, listC, listD)]
# Result
print("The concatenated lists: ",res)
print(np.array(res).shape)

import torch

pt_3_by_3_eye_ex = torch.eye(200)
