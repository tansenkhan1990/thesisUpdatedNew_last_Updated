import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from torch.autograd import Variable

def accuracy(y_pred, y_true):
    """
    Compute accuracy score.

    Params:
    -------
    y_pred: np.array
        Predicted (Bernoulli) probabilities.

    y_true: np.array, binary
        True (Bernoulli) labels.

    thresh: float, default: 0.5
        Classification threshold.

    reverse: bool, default: False
        If it is True, then classify (y <= thresh) to be 1.
    """
    return np.mean(y_pred == y_true)


def mean_rank(rank):
    m_r = 0
    N = len(rank)
    for i in rank:
        m_r = m_r + i / N

    return m_r


def mrr(rank):
    mrr = 0
    N = len(rank)
    for i in rank:
        mrr = mrr + 1 / i / N

    return mrr


def hit_N(rank, N):
    hit = 0
    for i in rank:
        if i <= N:
            hit = hit + 1

    hit = hit / len(rank)

    return hit


def rank_left_LogicENN(self, X, rev_set=0):
    rank = []
    rank_filter = []
    if rev_set == 0:
        for triple in X:
            X_i = np.ones([self.kg.n_entity, 3])
            for i in range(0, self.kg.n_entity):
                X_i[i, 0] = i
                X_i[i, 1] = triple[1]
                X_i[i, 2] = triple[2]
            i_score = self.score(X_i)
            if self.gpu:
                i_score = i_score.view(1, -1).cuda()

            [i_score, rank_all] = torch.topk(i_score, self.kg.n_entity)
            rank_triple = 1
            rank_filter_triple = 1
            for rank_i in rank_all[0]:
                if rank_i.cpu().numpy() == triple[0]:
                    break
                else:
                    rank_triple += 1
                    try:
                        tag_head = self.all_pos[rank_i.cpu().item(), triple[1], triple[2]]
                    except KeyError:
                        rank_filter_triple += 1

            rank.append(rank_triple)
            rank_filter.append(rank_filter_triple)
    else:
        for triple in X:
            X_i = np.ones([self.kg.n_entity, 3])
            for i in range(0, self.kg.n_entity):
                X_i[i, 0] = triple[1]
                X_i[i, 1] = i
                X_i[i, 2] = triple[2] + self.kg.n_relation / 2
            i_score = self.score(X_i)
            if self.gpu:
                i_score = i_score.view(1, -1).cuda()

            [i_score, rank_all] = torch.topk(i_score, self.kg.n_entity)
            rank_triple = 1
            rank_filter_triple = 1
            for rank_i in rank_all[0]:
                if rank_i.cpu().numpy() == triple[0]:
                    break
                else:
                    rank_triple += 1
                    try:
                        tag_head = self.all_pos[triple[1], rank_i.cpu().item(), triple[2] + self.kg.n_relation / 2]
                    except KeyError:
                        rank_filter_triple += 1

            rank.append(rank_triple)
            rank_filter.append(rank_filter_triple)

    return [rank, rank_filter]


def rank_right_LogicENN(self, X):
    rank = []
    rank_filter = []
    for triple in X:
        X_i = np.ones([self.kg.n_entity, 3])
        for i in range(0, self.kg.n_entity):
            X_i[i, 0] = triple[0]
            X_i[i, 1] = i
            X_i[i, 2] = triple[2]
        i_score = self.score(X_i)
        if self.gpu:
            i_score = i_score.view(1, -1).cuda()

        [i_score, rank_all] = torch.topk(i_score, self.kg.n_entity)
        rank_triple = 1
        rank_filter_triple = 1
        for rank_i in rank_all[0]:
            if rank_i.cpu().numpy() == triple[1]:
                break
            else:
                rank_triple += 1
                try:
                    tag_head = self.all_pos[triple[0], rank_i.cpu().item(), triple[2]]
                except KeyError:
                    rank_filter_triple += 1

        rank.append(rank_triple)
        rank_filter.append(rank_filter_triple)

    return [rank, rank_filter]

def right_rank(self, X , Largest=False): #True for distmult complex
    if  self.name == 'distmult' or self.name == 'complEx' :
        Largest = True
    rank = []
    rank_filter = []
    for triple in X:
        X_i = np.ones([self.kg.n_entity, 3])
        for i in range(0, self.kg.n_entity):
            X_i[i, 0] = triple[0]
            X_i[i, 1] = i
            X_i[i, 2] = triple[2]
        if self.regul == True:
            [i_score, reg] = self.forward(X_i)
        else:
            i_score = self.forward(X_i)
        if self.gpu:
            i_score = i_score.view(1, -1).cuda()

        [i_score, rank_all] = torch.topk(i_score, self.kg.n_entity, largest=Largest)
        rank_triple = 1
        rank_filter_triple = 1
        for rank_i in rank_all:
            if rank_i.cpu().numpy() == triple[1]:
                break
            else:
                rank_triple += 1
                try:
                    tag_head = self.all_pos[triple[0], rank_i.cpu().item(), triple[2]]
                except KeyError:
                    rank_filter_triple += 1
        print([rank_triple, rank_filter_triple])
        rank.append(rank_triple)
        rank_filter.append(rank_filter_triple)
    return [rank, rank_filter]

def right_rank_t(self, X , Largest=False): #True for distmult complex
    if  self.name == 'distmult' or self.name == 'ComplEx' or self.name == 'distmult_quad' or self.name == 'complEx_quad':
        Largest = True
    rank = []
    rank_filter = []
    for triple in X:
        X_i = np.ones([self.kg.n_entity, 5], dtype=int)
        for i in range(0, self.kg.n_entity):
            X_i[i, 0] = triple[0]
            X_i[i, 1] = i
            X_i[i, 2] = triple[2]
            X_i[i, 3] = triple[3]
            X_i[i, 4] = triple[4]
        if self.regul == True:
            [i_score, reg] = self.forward_t(X_i)
        else:
            i_score = self.forward_t(X_i)
        if self.gpu:
            i_score = i_score.view(1, -1).cuda()

        [i_score, rank_all] = torch.topk(i_score, self.kg.n_entity, largest=Largest)
        rank_triple = 1
        rank_filter_triple = 1
        for rank_i in rank_all:
            if rank_i.cpu().numpy() == triple[1]:
                break
            else:
                rank_triple += 1
                try:
                    tag_head = self.all_pos[triple[0], rank_i.cpu().item(), triple[2],triple[3], triple[4]]
                except KeyError:
                    rank_filter_triple += 1
        print([rank_triple, rank_filter_triple])
        rank.append(rank_triple)
        rank_filter.append(rank_filter_triple)
    return [rank, rank_filter]

def left_rank(self, X, rev_set=0, Largest=False): #true for distmult, Complex
    if  self.name == 'distmult' or self.name == 'complEx' :
        Largest = True
    rank = []
    rank_filter = []
    #print(X)
    #exit()
    if rev_set == 0:
        for triple in X:
            #print(triple[0])
            #exit()
            #3 should be changed to 5.
            X_i = np.ones([self.kg.n_entity, 3])
            for i in range(0, self.kg.n_entity):
                X_i[i, 0] = i
                X_i[i, 1] = triple[1]
                X_i[i, 2] = triple[2]
                #two lines should be added for time and location and load it to Xi
            #print(X_i)
            #exit()
            if self.regul == True:
                [i_score, reg] = self.forward(X_i)
            else:
                i_score = self.forward(X_i)

            if self.gpu:
                i_score = i_score.view(1, -1).cuda()
            #print(i_score.size())
            #exit()
            # print(len(i_score))
            # print(self.kg.n_entity)
            # exit()
            [i_score, rank_all] = torch.topk(i_score, self.kg.n_entity, largest=Largest)
            #print(rank_all)
            #exit()
            #names should be changed as well
            rank_triple = 1
            rank_filter_triple = 1
            for rank_i in rank_all:
                #print(rank_i)
                #exit()
                if rank_i.cpu().numpy() == triple[0]:
                    break
                else:
                    rank_triple += 1
                    try:
                        tag_head = self.all_pos[rank_i.cpu().item(), triple[1], triple[2]]
                    except KeyError:
                        rank_filter_triple += 1
            print([rank_triple,rank_filter_triple])
            rank.append(rank_triple)
            rank_filter.append(rank_filter_triple)
    else:
        for triple in X:
            X_i = np.ones([self.kg.n_entity, 3])
            for i in range(0, self.kg.n_entity):
                X_i[i, 0] = triple[1]
                X_i[i, 1] = i
                X_i[i, 2] = triple[2] + self.kg.n_relation / 2
            i_score = self.forward(X_i)
            if self.gpu:
                i_score = i_score.view(1, -1).cuda()

            [i_score, rank_all] = torch.topk(i_score, self.kg.n_entity, largest=Largest)
            rank_triple = 1
            rank_filter_triple = 1
            for rank_i in rank_all:
                if rank_i.cpu().numpy() == triple[0]:
                    break
                else:
                    rank_triple += 1
                    try:
                        tag_head = self.all_pos[triple[1], rank_i.cpu().item(), triple[2] + self.kg.n_relation / 2]
                    except KeyError:
                        rank_filter_triple += 1
            #print([rank_triple,rank_filter_triple])
            rank.append(rank_triple)
            rank_filter.append(rank_filter_triple)

    return [rank, rank_filter]

def left_rank_t(self, X, rev_set=0, Largest=False): #true for distmult, Complex
    if  self.name == 'distmult' or self.name == 'complEx' or self.name == 'distmult_quad' or self.name == 'complEx_quad':
        Largest = True
    rank = []
    #print(Largest)
    rank_filter = []
    #print(X)
    #exit()
    if rev_set == 0:
        for triple in X:
            #print(triple[0])
            #exit()
            #3 should be changed to 5.
            X_i = np.ones([self.kg.n_entity, 5], dtype=int)
            for i in range(0, self.kg.n_entity):
                X_i[i, 0] = i
                X_i[i, 1] = triple[1]
                X_i[i, 2] = triple[2]
                X_i[i, 3] = triple[3]
                X_i[i, 4] = triple[4]
                #two lines should be added for time and location and load it to Xi
            #print(X_i)
            #exit()
            if self.regul == True:
                [i_score, reg] = self.forward_t(X_i)
            else:
                i_score = self.forward_t(X_i)
            #print(i_score)
            #exit()
            if self.gpu:
                i_score = i_score.view(1, -1).cuda()
            #print(i_score.size())
            #exit()
            [i_score, rank_all] = torch.topk(i_score, self.kg.n_entity, largest=Largest)
            #print(rank_all)
            #exit()
            #names should be changed as well
            rank_triple = 1
            rank_filter_triple = 1
            for rank_i in rank_all:
                #print(rank_i)
                #exit()
                if rank_i.cpu().numpy() == triple[0]:
                    break
                else:
                    rank_triple += 1
                    try:
                        tag_head = self.all_pos[rank_i.cpu().item(), triple[1], triple[2], triple[3], triple[4]]
                    except KeyError:
                        rank_filter_triple += 1
            print([rank_triple,rank_filter_triple])
            rank.append(rank_triple)
            rank_filter.append(rank_filter_triple)
    else:
        for triple in X:
            X_i = np.ones([self.kg.n_entity,5])
            for i in range(0, self.kg.n_entity):
                X_i[i, 0] = triple[1]
                X_i[i, 1] = i
                X_i[i, 2] = triple[2] + self.kg.n_relation / 2
            i_score = self.forward(X_i)
            if self.gpu:
                i_score = i_score.view(1, -1).cuda()

            [i_score, rank_all] = torch.topk(i_score, self.kg.n_entity, largest=Largest)
            rank_triple = 1
            rank_filter_triple = 1
            for rank_i in rank_all[0]:
                if rank_i.cpu().numpy() == triple[0]:
                    break
                else:
                    rank_triple += 1
                    try:
                        tag_head = self.all_pos[triple[1], rank_i.cpu().item(), triple[2] + self.kg.n_relation / 2]
                    except KeyError:
                        rank_filter_triple += 1
            #print([rank_triple,rank_filter_triple])
            rank.append(rank_triple)
            rank_filter.append(rank_filter_triple)

    return [rank, rank_filter]
