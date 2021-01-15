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

data_1 = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/umls/mapped/loss', header=None)
data_2 = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/umls/mapped/orig_loss', header=None)
data_3 = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/umls/mapped/loss_mod', header=None)

data_4 = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/umls/mapped/distmult_random_loss', header=None)
data_5 = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/umls/mapped/distmult_orig_loss', header=None)
data_6 = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/umls/mapped/distmult_mod_loss', header=None)
#data_2 = pd.read_table('/home/mirza/PycharmProjects/LogicENN filtered/rule_loss_noconv', header=None)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
#plt.plot(data_1[0], label='loss with random negative')
#plt.plot(data_2[0], label='loss with DNS')
#plt.plot(data_3[0], label='loss with ANS')
# plt.plot(data_1[3], label='symmetric_loss')
# plt.plot(data_1[2], label='equivalence')
axes[0].plot(data_1[0], label='loss convergance with random negative (11.12 sec)')
#axes[0][1].plot(data_2[1], label='implication (Without injected rules)')
axes[0].plot(data_2[0], label='loss convergance with DNS (2229.078 sec)')
#axes[1][1].plot(data_2[2], label='inverse(Without injected rules)')
axes[0].plot(data_3[0], label='loss convergance with ADNS (662.28 sec)')


axes[1].plot(data_4[0], label='loss convergance with random negative (10.658 sec)')
#axes[0][1].plot(data_2[1], label='implication (Without injected rules)')
axes[1].plot(data_5[0], label='loss convergance with DNS (2634.677 sec)')
#axes[1][1].plot(data_2[2], label='inverse(Without injected rules)')
axes[1].plot(data_6[0], label='loss convergance with ADNS (571.7045 sec)')
#axes[2][1].plot(data_2[3], label='symmetric (Without injected rules)')
#axes[3].plot(data_1[3], label='symmetric')
#axes[3][1].plot(data_2[4], label='equivalence(Without injected rules)')
#axes[4].plot(data_1[4], label='equivalence')
#axes[4][1].plot(data_2[5], label='complex (Without injected rules)')
#plt.plot(data[5], label='complex')

axes[0].legend(loc='upper right',prop={'size': 15})
#axes[0][1].legend(loc='upper right',prop={'size': 10})
# axes[0][0].set_xlim([0,1.5])
# axes[0][1].set_xlim([0,1.5])
axes[0].set_ylim([0,2])
#axes[0][1].set_ylim([0,2])
axes[0].legend(loc='upper right',prop={'size': 15})
#axes[1][1].legend(loc='upper right',prop={'size': 10})
axes[0].set_ylim([0,2])
#axes[1][1].set_ylim([0,2])
axes[0].legend(loc='upper right',prop={'size': 15})
#axes[2][1].legend(loc='upper right',prop={'size': 10})
axes[0].set_ylim([0,2])

axes[0].set_title('Loss convergance with TransE')
#axes[2][1].set_ylim([0,1])
#axes[3].legend(loc='upper right',prop={'size': 10})
#axes[3][1].legend(loc='upper right',prop={'size': 10})
#axes[3].set_ylim([0,5])
#axes[3][1].set_ylim([0,15])
#axes[4].legend(loc='upper right',prop={'size': 10})
#axes[4][1].legend(loc='upper right',prop={'size': 10})
#axes[4].set_ylim([0,5])
#axes[4][1].set_ylim([0,0.005])

axes[1].legend(loc='upper right',prop={'size': 15})
#axes[0][1].legend(loc='upper right',prop={'size': 10})
# axes[0][0].set_xlim([0,1.5])
# axes[0][1].set_xlim([0,1.5])
axes[1].set_ylim([0,2])
#axes[0][1].set_ylim([0,2])
axes[1].legend(loc='upper right',prop={'size': 15})
#axes[1][1].legend(loc='upper right',prop={'size': 10})
axes[1].set_ylim([0,2])
#axes[1][1].set_ylim([0,2])
axes[1].legend(loc='upper right',prop={'size': 15})
#axes[2][1].legend(loc='upper right',prop={'size': 10})
axes[1].set_ylim([0,2])

axes[1].set_title('Loss convergance with Distmult')




# Set common labels
fig.text(0.5, 0.04, 'Training Steps', ha='center', va='center')
fig.text(0.06, 0.5, 'Loss', ha='center', va='center', rotation='vertical')
plt.savefig('loss conv.png')
plt.show()
#data = pd.read_table('/home/mirza/PycharmProjects/LogicENN filtered/kinship/hpt_kinship', header=None)
#prob_table = pd.read_table(os.path.join('/home/mirza/PycharmProjects/LogicENN filtered/kinship', 'hpt_kinship'), names=['tph', 'hpt'],
#                               header=None)