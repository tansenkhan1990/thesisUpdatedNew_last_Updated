# encoding: utf-8
import os
import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle as skshuffle

class KnowledgeGraph:

    def __init__(self, data_dir, rev_set=0, fifthopole = False):
        self.fifthopole = fifthopole
        self.data_dir = data_dir
        self.entity_dict = {}
        self.entities = []
        self.relation_dict = {}
        self.times_dict = {}
        self.times = []
        self.locations_dict = {}
        self.locations = []
        self.n_entity = 0
        self.n_relation = 0
        self.n_times = 0
        self.n_locations = 0
        self.training_triples = []  # list of triples in the form of (h, t, r)
        #self.validation_triples = []
        self.test_triples = []
        self.all_triples = []
        #self.n_validation_triple = 0
        self.n_test_triple = 0
        self.rev_set = rev_set
        '''load dicts and triples'''

        self.load_dicts_quad() if fifthopole == True else self.load_dicts()
        self.load_quadropoles() if fifthopole == True else self.load_triples()
        '''construct pools after loading'''
#        self.training_triple_pool = set(self.training_triples)
#        self.golden_triple_pool = set(self.training_triples)
    def load_dicts_quad(self):
        entity_dict_file = 'entities.dict'
        relation_dict_file = 'relations.dict'
        times_dict_file = 'times.dict'
        locations_dict_file = 'locations.dict'
        #two columns will be added here
        #the other dictionaries
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join(self.data_dir, entity_dict_file), header=None)
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        print('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join(self.data_dir, relation_dict_file), header=None)
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.n_relation = len(self.relation_dict)
        if self.rev_set>0: self.n_relation *= 2
        print('#relation: {}'.format(self.n_relation)) #What is rev set
        print('-----Loading times dict-----')
        times_df = pd.read_table(os.path.join(self.data_dir, times_dict_file), header=None)
        self.times_dict = dict(zip(times_df[0], times_df[1]))
        self.n_times = len(self.times_dict)
        self.times = list(self.times_dict.values())
        print('#times: {}'.format(self.n_times))
        print('-----Loading locations dict-----')
        locations_df = pd.read_table(os.path.join(self.data_dir, locations_dict_file), header=None)
        self.locations_dict = dict(zip(locations_df[0], locations_df[1]))
        self.n_locations = len(self.locations_dict)
        self.locations = list(self.locations_dict.values())
        print('#locations: {}'.format(self.n_locations))
        print("Dicts are loaded...")

    def load_dicts(self):
        entity_dict_file = 'entities.dict'
        relation_dict_file = 'relations.dict'
        #two columns will be added here
        #the other dictionaries
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join(self.data_dir, entity_dict_file), header=None)
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        print('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join(self.data_dir, relation_dict_file), header=None)
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.n_relation = len(self.relation_dict)
        if self.rev_set>0: self.n_relation *= 2
        print('#relation: {}'.format(self.n_relation)) #What is rev set
        print("Dicts are loaded...")

    def load_quadropoles(self):
        training_file = 'train.txt'
        #validation_file = 'valid.txt'
        test_file = 'test.txt'
        # test_file = 'train_demo.txt'
        print('-----Loading training triples-----')
        #Here we need to adapt as we get the time and location
        training_df = pd.read_table(os.path.join(self.data_dir, training_file), header=None)
        self.training_triples = list(zip([h for h in training_df[0]],
                                         [t for t in training_df[2]],
                                         [r for r in training_df[1]],
                                         [tm for tm in training_df[3]],
                                         [loc for loc in training_df[4]]))
############################################################################################################
#Unused
        if self.rev_set>0:
            rev_training_triples = list(zip([h for h in training_df[2]],
                                            [t for t in training_df[0]],
                                            [r+len(self.relation_dict) for r in training_df[1]]))
            self.training_triples += rev_training_triples
################################################333333###########################################################

        self.n_training_triple = len(self.training_triples)
        print('#training triple: {}'.format(self.n_training_triple))
        print('-----Loading validation triples-----')
        # validation_df = pd.read_table(os.path.join(self.data_dir, validation_file), header=None)
        # self.validation_triples = list(zip([h for h in validation_df[0]],
        #                                    [t for t in validation_df[2]],
        #                                    [r for r in validation_df[1]],
        #                                    [tm for tm in validation_df[3]],
        #                                    [loc for loc in validation_df[4]]
        #                                    ))
        # self.n_validation_triple = len(self.validation_triples)
        # print('#validation triple: {}'.format(self.n_validation_triple))
        print('-----Loading test triples------')
        test_df = pd.read_table(os.path.join(self.data_dir, test_file), header=None)
        self.test_triples = list(zip([h for h in test_df[0]],
                                     [t for t in test_df[2]],
                                     [r for r in test_df[1]],
                                     [tm for tm in test_df[3]],
                                     [loc for loc in test_df[4]]
                                     ))
        self.n_test_triple = len(self.test_triples)
        print('#test triple: {}'.format(self.n_test_triple))
        print("Triples are loaded...")
        self.all_triples = self.training_triples +  self.test_triples

    def load_triples(self):
        training_file = 'train.txt'
        #validation_file = 'valid.txt'
        # test_file = 'test_revised.txt'
        test_file = 'test.txt'
        print('-----Loading training triples and it is not a fifthopple-----')
        #Here we need to adapt as we get the time and location
        training_df = pd.read_table(os.path.join(self.data_dir, training_file), header=None)
        self.training_triples = list(zip([h for h in training_df[0]],
                                         [t for t in training_df[2]],
                                         [r for r in training_df[1]]))
        ###################unused###################3
        if self.rev_set>0:
            rev_training_triples = list(zip([h for h in training_df[2]],
                                            [t for t in training_df[0]],
                                            [r+len(self.relation_dict) for r in training_df[1]]))
            self.training_triples += rev_training_triples


        self.n_training_triple = len(self.training_triples)
        print('#training triple: {}'.format(self.n_training_triple))
        #print('-----Loading validation triples-----')
        # validation_df = pd.read_table(os.path.join(self.data_dir, validation_file), header=None)
        # self.validation_triples = list(zip([h for h in validation_df[0]],
        #                                    [t for t in validation_df[2]],
        #                                    [r for r in validation_df[1]]))
        # self.n_validation_triple = len(self.validation_triples)
        # print('#validation triple: {}'.format(self.n_validation_triple))
        # print('-----Loading test triples------')
        test_df = pd.read_table(os.path.join(self.data_dir, test_file), header=None)
        self.test_triples = list(zip([h for h in test_df[0]],
                                     [t for t in test_df[2]],
                                     [r for r in test_df[1]]))
        self.n_test_triple = len(self.test_triples)
        print('#test triple: {}'.format(self.n_test_triple))
        print("Triples are loaded...")
        self.all_triples = self.training_triples + self.test_triples
#######################unused#####################################
    def next_raw_batch(self, batch_size):
        rand_idx = np.random.permutation(self.n_training_triple)
        start = 0
        while start < self.n_training_triple: #Make sure the triples are not duplicating in training
            end = min(start + batch_size, self.n_training_triple)
            for i in rand_idx[start:end]:
                yield [self.training_triples[i]]
            start = end

    #######################unused#####################################
    def generate_training_batch(self, in_queue, out_queue):
        while True:
            raw_batch = in_queue.get()
            if raw_batch is None:
                return
            else:
                batch_pos = raw_batch
                batch_neg = []
                corrupt_head_prob = np.random.binomial(1, 0.5)
                for head, tail, relation in batch_pos:
                    head_neg = head
                    tail_neg = tail
                    while True:
                        if corrupt_head_prob:
                            head_neg = random.choice(self.entities)
                        else:
                            tail_neg = random.choice(self.entities)
                        if (head_neg, tail_neg, relation) not in self.training_triple_pool:
                            break
                    batch_neg.append((head_neg, tail_neg, relation))
                out_queue.put((batch_pos, batch_neg))

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

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples