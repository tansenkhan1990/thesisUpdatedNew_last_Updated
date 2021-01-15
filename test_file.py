import numpy as np
import itertools
from sklearn.utils import shuffle as skshuffle


# id = [1,2,3,4,5]
# value = []
#
# for v,x in zip(id, *value):
#     print(v)
#
# letters = ['a', 'b', 'c']
# numbers = []
#
# for letter, number in zip(letters, numbers):
#     print(f'{letter} -> {number}')
#
# for combination, value in itertools.zip_longest(id, value):
#     print(combination, value)
#


def get_minibatches(X, mb_size, shuffle=True):
    """
    Generate minibatches from given dataset for training.

    Params:
    -------
    X: np.array of M x 3
        Contains the triplets from dataset. The entities and relations are
        translated to its unique indices.

    mb_size: int
        Size of each minibatch.

    shuffle: bool, default True
        Whether to shuffle the dataset before dividing it into minibatches.

    Returns:
    --------
    mb_iter: generator
        Example usage:
        --------------
        mb_iter = get_minibatches(X_train, mb_size)
        for X_mb in mb_iter:
            // do something with X_mb, the minibatch
    """
    X_shuff = X.copy()
    if shuffle:
        X_shuff = skshuffle(X_shuff)

    for i in range(0, X_shuff.shape[0], mb_size):
        yield X_shuff[i:i + mb_size]


data = np.random.randint(5, size=(10, 3))

train_triple = list(get_minibatches(data, 2, shuffle=True))

array = np.array([[2,3,1],[3,1,0]])
indexes = [np.where(np.all(data==i,axis=1)) for i in data]

batch_triples = np.where((data==array[:,None]).all(-1))[1]

import pandas as pd



ind = np.array(np.arange(len(data)))

data = np.c_[data,ind]
C =3
M = data.shape[0]
X_corr = np.zeros([M * C, 3])

for i in range(0, M):
    corrupt_head_prob = np.random.rand(1)
    e_idxs = np.random.choice(M, C)
    #        if corrupt_head_prob>tph[X[i,2]]:
    if corrupt_head_prob > 0.5:
        for j in range(0, C):
            if e_idxs[j] != i:
                X_corr[i + j * M, 0] = data[e_idxs[j], 0]
            else:
                X_corr[i + j * M, 0] = data[e_idxs[j] // 2, 0]
            X_corr[i + j * M, 1] = data[i, 1]
            X_corr[i + j * M, 2] = data[i, 2]
    else:
        for j in range(0, C):
            X_corr[i + j * M, 0] = data[i, 0]
            if e_idxs[j] != i:
                X_corr[i + j * M, 1] = data[e_idxs[j], 1]
            else:
                X_corr[i + j * M, 1] = data[e_idxs[j] // 2, 1]
            X_corr[i + j * M, 2] = data[i, 2]
    #            X_corr[i + j * M, 3] = X[i, 3]
    #            if (h_n, t_n, X[i,2]) not in X:
    #                X_corr[i,:]=(h_n, t_n, X[i,2])
    #            else:
    #            else:
    #                X_corr[i,:]= (X[e_idxs,0], X[i,1], X[i,2])


def repeat(arr, count):
    return np.stack([arr for _ in range(count)], axis=1)


ind = np.array(np.arange(len(data)))
corr =  np.tile(ind,C)

X_corr = np.c_[X_corr,corr]
pos_ind = data[:,-1].squeeze()
corr_new =  np.tile(pos_ind,C)
data = np.c_[X_corr,corr_new]

np.reshape()

import torch

def torch_tile(tensor, dim, n):
    """Tile n times along the dim axis"""
    if dim == 0:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,n,1).view(-1,tensor.shape[1])
    else:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,1,n).view(tensor.shape[0], -1)


z = torch.FloatTensor([[1,2,3,4,5,6,7,8,9]])
z = torch_tile(z,1,3)
z = z.view(3,-1).T

import pandas as pd
import numpy as np
data = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/wn18_noise_included/original_quadropoles.txt', header=None)

data_2 = np.loadtxt('/home/mirza/PycharmProjects/frame_work/dataset/wn18_noise_included/original_quadropoles.txt', dtype=str)

data_1 = np.array([1,2,3])
data_2 = np.array([3,4,5])

data_3 = np.vstack((data_1,data_2))


import pandas as pd
import os

entity_df = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/wn18rr/entities.dict', header=None, dtype=str)

entity_dict = dict(zip(entity_df[0], entity_df[1]))