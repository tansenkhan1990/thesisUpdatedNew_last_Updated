from utilities import evaluation_utils,loss_functions
from dataset import KnowledgeGraph
import torch
import numpy as np
from time import time
from sklearn.utils import shuffle as skshuffle
import os
from torch.nn import functional as F

def sample_negatives(X, C):
    """
    Perform negative sampling by corrupting head or tail of each triplets in
    dataset.

    Params:
    -------
    X: int matrix of M x 3, where M is the (mini)batch size
        First column contains index of head entities.
        Second column contains index of relationships.
        Third column contains index of tail entities.

    n_e: int
        Number of entities in dataset.

    Returns:
    --------
    X_corr: int matrix of M x 3, where M is the (mini)batch size
        Similar to input param X, but at each column, either first or third col
        is subtituted with random entity.
    """
    M = X.shape[0]
    X_corr = np.zeros([M * C, 3])

    for i in range(0, M):
        corrupt_head_prob = np.random.rand(1)

        e_idxs = np.random.choice(M, C)

        #        if corrupt_head_prob>tph[X[i,2]]:
        if corrupt_head_prob > 0.5:
            for j in range(0, C):
                if e_idxs[j] != i:
                    X_corr[i + j * M, 0] = X[e_idxs[j], 0]
                else:
                    X_corr[i + j * M, 0] = X[e_idxs[j] // 2, 0]
                X_corr[i + j * M, 1] = X[i, 1]
                X_corr[i + j * M, 2] = X[i, 2]
        #        X_corr[i + j * M, 3] = X[i, 3]
        #            if (h_n, t_n, X[i,2]) not in X:
        #                X_corr[i,:]=(h_n, t_n, X[i,2])
        #            else:
        #                X_corr[i,:]=(X[i,0], X[e_idxs,1], X[i,2])
        else:
            for j in range(0, C):
                X_corr[i + j * M, 0] = X[i, 0]
                if e_idxs[j] != i:
                    X_corr[i + j * M, 1] = X[e_idxs[j], 1]
                else:
                    X_corr[i + j * M, 1] = X[e_idxs[j] // 2, 1]
                X_corr[i + j * M, 2] = X[i, 2]
    #            X_corr[i + j * M, 3] = X[i, 3]
    #            if (h_n, t_n, X[i,2]) not in X:
    #                X_corr[i,:]=(h_n, t_n, X[i,2])
    #            else:
    #            else:
    #                X_corr[i,:]= (X[e_idxs,0], X[i,1], X[i,2])

    return X_corr

def sample_negatives_t(X, C):
    """
    Perform negative sampling by corrupting head or tail of each triplets in
    dataset.

    Params:
    -------
    X: int matrix of M x 3, where M is the (mini)batch size
        First column contains index of head entities.
        Second column contains index of relationships.
        Third column contains index of tail entities.

    n_e: int
        Number of entities in dataset.

    Returns:
    --------
    X_corr: int matrix of M x 3, where M is the (mini)batch size
        Similar to input param X, but at each column, either first or third col
        is subtituted with random entity.
    """
    M = X.shape[0]
    X_corr = np.zeros([M * C,5])

    for i in range(0, M):
        corrupt_head_prob = np.random.rand(1)

        e_idxs = np.random.choice(M, C)

        #        if corrupt_head_prob>tph[X[i,2]]:
        if corrupt_head_prob > 0.5:
            for j in range(0, C):
                if e_idxs[j] != i:
                    X_corr[i + j * M, 0] = X[e_idxs[j], 0]
                else:
                    X_corr[i + j * M, 0] = X[e_idxs[j] // 2, 0]
                X_corr[i + j * M, 1] = X[i, 1]
                X_corr[i + j * M, 2] = X[i, 2]
                X_corr[i + j * M, 3] = X[i, 3]
                X_corr[i + j * M, 4] = X[i, 4]
        #        X_corr[i + j * M, 3] = X[i, 3]
        #            if (h_n, t_n, X[i,2]) not in X:
        #                X_corr[i,:]=(h_n, t_n, X[i,2])
        #            else:
        #                X_corr[i,:]=(X[i,0], X[e_idxs,1], X[i,2])
        else:
            for j in range(0, C):
                X_corr[i + j * M, 0] = X[i, 0]
                if e_idxs[j] != i:
                    X_corr[i + j * M, 1] = X[e_idxs[j], 1]
                else:
                    X_corr[i + j * M, 1] = X[e_idxs[j] // 2, 1]
                X_corr[i + j * M, 2] = X[i, 2]
                X_corr[i + j * M, 3] = X[i, 3]
                X_corr[i + j * M, 4] = X[i, 4]
    #            X_corr[i + j * M, 3] = X[i, 3]
    #            if (h_n, t_n, X[i,2]) not in X:
    #                X_corr[i,:]=(h_n, t_n, X[i,2])
    #            else:
    #            else:
    #                X_corr[i,:]= (X[e_idxs,0], X[i,1], X[i,2])

    return X_corr

def DNS_paper_orig(positive_batch, C, model, all_triples, number_of_entities, prob_table):
    entity_set_length = number_of_entities
    negative_batch = []
    t = positive_batch.copy()
    for triple in t:
        start = time()
        N = np.zeros((C,3), dtype=int)
        # print(triple)
        current_relation = triple[2]

        tph = prob_table['tph'][current_relation]
        hpt = prob_table['hpt'][current_relation]

        head_corruption_probability = tph / (tph + hpt)
        tail_corruption_probability = hpt / (tph + hpt)

        corrupt_ind = np.random.choice([0, 1], p=[head_corruption_probability, tail_corruption_probability])
        candidate = triple[corrupt_ind]
        # print(candidate)
        # exit()
        # print('to be corrupted: ', candidate)
        candidate_emb = model.get_embedding(candidate).view(1, -1)
        # candidate_repeated = np.repeat(candidate,entity_set_length-1)
        all_entity = np.arange(0, entity_set_length, 1)
        all_entity_embedding = model.get_vectorized_embedding(all_entity)
        M = F.cosine_similarity(all_entity_embedding, candidate_emb)
        # all_triples = list(zip([h for h in all_triples[0]],
        #                        [t for t in all_triples[2]],
        #                        [r for r in all_triples[1]]))
        # all_df = pd.DataFrame(all_triples)
        #d = {}
        c = 0
        for i, m in enumerate(M):
            #print('index: ', i, 'value: ', m)
            if (torch.tensor(i) != candidate):
                # print(candidate)
                # print(triple)
                #print(c)
                if c == C:
                    #print('here')
                    break
                #print(c)
                clone_triple = triple.copy()
                clone_triple[corrupt_ind] = i
                # print(clone_triple)
                # exit()
                #if ((all_df.loc[(all_df[0] == clone_triple[0]) & (all_df[1] == clone_triple[1]) & (
                #        all_df[2] == clone_triple[2])]).empty):  # for tail
                try:
                    #check whether it exists in the original training_triple set. Here all pos is the training triple set
                    check = model.all_pos[clone_triple[0], clone_triple[1], clone_triple[2]]
                    #print('found')
                    continue
                except:
                    # print(m)
                    p_accept = torch.max(torch.tensor(0.0).cuda(), m.cuda())

                    # May be I create a dictionary here
                    # d = {entity_index, probability}
                    p_accept = p_accept.detach().cpu().numpy()
                    #d[i] = p_accept
                    p_reject = 1 - p_accept
                    # print(p_accept)
                    # print(p_reject)
                    take =  np.random.choice(['take','not_take'], 1, p=[p_accept,p_reject])

                    if ((take == 'take') and (c<C)):
                        #print(N)
                        N[c] = clone_triple
                        c+=1
                    else:
                        #print('Not take')
                        continue

            else:

                continue
        end = time()
        #print('time :{:.2f}s'.format(end - start))
        #negative_batch = np.append(negative_batch,N, axis=0)
        for i in range(len(N)):
            negative_batch.append(np.array(N[i]))
        #negative_batch.append(np.array(N[0]))
    return np.array(negative_batch)

def DNS_paper_modified(positive_batch,C, model, all_triples, number_of_entities,prob_table):
    entity_set_length = number_of_entities
    negative_batch = []
    t = positive_batch.copy()
    for triple in t:
        #N = []
        #print(triple)
        start = time()
        current_relation = triple[2]
        tph = prob_table['tph'][current_relation]
        hpt = prob_table['hpt'][current_relation]
        head_corruption_probability = tph / (tph + hpt)
        tail_corruption_probability = hpt / (tph + hpt)
        corrupt_ind = np.random.choice([0, 1], p=[head_corruption_probability, tail_corruption_probability])
        candidate = triple[corrupt_ind]
        #print(candidate)
        #exit()
        #print('to be corrupted: ', candidate)
        candidate_emb = model.get_embedding(candidate).view(1,-1)
        #candidate_repeated = np.repeat(candidate,entity_set_length-1)
        all_entity = np.arange(0, entity_set_length, 1)
        all_entity_candidate_removed = np.delete(all_entity, list(all_entity).index(candidate))

        # for  i in all_entity_candidate_removed:
        #     print(type(i))
        #     exit()

        all_entity_embedding = model.get_vectorized_embedding(all_entity_candidate_removed)
        #print(all_entity_embedding.size())
        #print(candidate_emb.size())
        #exit()
        ##################################################################
        #track either head or tail was corrupted



        ###################################################################

        M = F.cosine_similarity(all_entity_embedding,candidate_emb)

        M[M<0]=0

        #print(type(all_entity_candidate_removed))
        #exit()
        #print(all_entity_candidate_removed.size)
        #print(M.detach().cpu().numpy() / sum(M.detach().cpu().numpy()))
        #print(all_entity_candidate_removed.shape)
        #exit()
        try:
            N = np.random.choice(all_entity_candidate_removed, C, p=M.detach().cpu().numpy() / sum(M.detach().cpu().numpy()), replace=False)
        except:
            #when we have fewer fitness than required
            N = np.random.choice(all_entity_candidate_removed, C)
        #print(N)
        #exit()
        #N = torch.multinomial(torch.from_numpy(all_entity_candidate_removed), C, p = M / sum(M,replacement=False))  # Size 4: 3, 2, 1, 2
        # #TODO

        def change_again(all_entity_candidate_removed, C, M, existing_candidate, candidate_from_cosine_array):
            #print(list(all_entity_candidate_removed).index(existing_candidate))
            all_entity_candidate_removed = np.delete(all_entity_candidate_removed, list(all_entity_candidate_removed).index(existing_candidate))
            #print(all_entity_candidate_removed.shape)
            #exit()
            #print(list(M).index(candidate_from_cosine_array))
            M = M[M!=candidate_from_cosine_array]
            N = np.random.choice(all_entity_candidate_removed, C,
                                 p=M.detach().cpu().numpy() / sum(M.detach().cpu().numpy()), replace=False)
            return  N, all_entity_candidate_removed, M


        if (corrupt_ind == 1):
            for i in range(0,C):
                try:
                    while(True):
                        check = model.all_pos[triple[0], N[i], triple[2]]
                        #print('exist',[triple[0], N[i], triple[2]])
                        N, all_entity_candidate_removed, M = change_again(all_entity_candidate_removed, C, M, N[i], M[list(all_entity_candidate_removed).index(N[i])])
                        #exit()
                        try:
                            check = model.all_pos[triple[0], N[i], triple[2]]
                        except:
                            break
                        #print('after changing', [triple[0], N[i], triple[2]])
                except:
                    continue
        else:
            for i in range(0, C):
                try:
                    while(True):
                        check = model.all_pos[N[i], triple[1], triple[2]]
                        #print('exist', [N[i], triple[1], triple[2]])
                        N, all_entity_candidate_removed, M = change_again(all_entity_candidate_removed, C, M, N[i], M[list(all_entity_candidate_removed).index(N[i])])
                        #print('after chenging', [N[i], triple[1], triple[2]])
                        try:
                            check = model.all_pos[N[i], triple[1], triple[2]]
                        except:
                            break
                except:
                    continue

        temp_triple = np.copy(triple)
        temp_matrix = np.tile(temp_triple,(C,1))

        for i in range(len(temp_matrix)):
            temp_matrix[i][corrupt_ind] = N[i]
        #print(i)
        #temp_triple[corrupt_ind]=N[0]
        end = time()
        #print('time :{:.2f}s'.format(end - start))
        for i in range(len(temp_matrix)):
            negative_batch.append(temp_matrix[i])
        #print(negative_triple)
        #exit()
    #print(negative_batch)
    #print(np.array(positive_batch).shape)
    return np.array(negative_batch)


def unique_negative_sampling(positive_batch,C, model, all_triples, number_of_entities,prob_table, unique_neg_sampling):
    entity_set_length = number_of_entities
    negative_batch = []
    t = positive_batch.copy()
    for triple in t:
        #print(triple)
        #N = []
        #print(triple)
        start = time()
        current_relation = triple[2]
        tph = prob_table['tph'][current_relation]
        hpt = prob_table['hpt'][current_relation]
        head_corruption_probability = tph / (tph + hpt)
        tail_corruption_probability = hpt / (tph + hpt)



        corrupt_ind = np.random.choice([0, 1], p=[head_corruption_probability, tail_corruption_probability])
        candidate = triple[corrupt_ind]


        #print(candidate)
        #exit()
        #print('to be corrupted: ', candidate)
        candidate_emb = model.get_embedding(candidate).view(1,-1)
        #candidate_repeated = np.repeat(candidate,entity_set_length-1)
        all_entity = np.arange(0, entity_set_length, 1)
        #all_entity = unique_neg_sampling

        des_str = ''

        if corrupt_ind == 0:
            des_str = 'sub:' + str(triple[2])
        else:
            des_str = 'obj:' + str(triple[2])

        inter_data = unique_neg_sampling[des_str]
        inter_data = np.array(inter_data)
        inter_data = [i for i in inter_data if ~np.isnan(i)]

        #print(inter_data)
        #inter_data = inter_data
        all_unique_entity = np.setxor1d(all_entity, inter_data).astype(int)


        #print(type(all_unique_entity))
        #exit()
        #print(all_entity_embedding)

        #print(candidate)
        #exit()
        #all_entity_candidate_removed = np.delete(all_entity, list(all_entity).index(candidate))
        all_entity_embedding = model.get_vectorized_embedding(all_unique_entity)
        #print((all_entity_embedding.detach().cpu().numpy()).shape)
        #print(type(all_entity_embedding))

        #exit()
        #print(all_entity_embedding.size())
        #print(candidate_emb.size())
        #exit()

        ##################################################################

        ###################################################################

        M = F.cosine_similarity(all_entity_embedding,candidate_emb)
        #print(type(M))
        #exit()
        M[M<0]=0
        #print(M)

        #exit()
        #print(type(all_entity_candidate_removed))
        #exit()
        #print(all_entity_candidate_removed.size)
        #print(M.detach().cpu().numpy() / sum(M.detach().cpu().numpy()))
        try:
            #print('in try')
            #print(len(all_unique_entity))
            N = np.random.choice(all_unique_entity , C, p=M.detach().cpu().numpy() / sum(M.detach().cpu().numpy()), replace=False)
        except:
            #print(len(all_unique_entity))
            #when we have fewer fitness than required
            try:
                #((len(all_unique_entity)) < C)
                #print('here')
                N = np.random.choice(all_entity , C)
            except:
                N = np.random.choice(all_unique_entity , C)

        #print(N)
        #exit()
        #N = torch.multinomial(torch.from_numpy(all_entity_candidate_removed), C, p = M / sum(M,replacement=False))  # Size 4: 3, 2, 1, 2
        # #TODO

        def change_again(all_unique_entity, C, M, existing_candidate, candidate_from_cosine_array):
            #print(list(all_entity_candidate_removed).index(existing_candidate))
            all_unique_entity = np.delete(all_unique_entity, list(all_unique_entity).index(existing_candidate))
            #print(all_entity_candidate_removed.shape)
            #exit()
            #print(list(M).index(candidate_from_cosine_array))
            M = M[M!=candidate_from_cosine_array]
            N = np.random.choice(all_unique_entity, C,
                                 p=M.detach().cpu().numpy() / sum(M.detach().cpu().numpy()), replace=False)
            return  N, all_unique_entity, M


        if (corrupt_ind == 1):
            for i in range(0,C):
                try:
                    while(True):
                        check = model.all_pos[triple[0], N[i], triple[2]]
                        #print('exist',[triple[0], N[i], triple[2]])
                        N, all_unique_entity, M = change_again(all_unique_entity, C, M, N[i], M[list(all_unique_entity).index(N[i])])
                        #exit()
                        try:
                            check = model.all_pos[triple[0], N[i], triple[2]]
                        except:
                            break
                        #print('after changing', [triple[0], N[i], triple[2]])
                except:
                    continue
        else:
            for i in range(0, C):
                try:
                    while(True):
                        check = model.all_pos[N[i], triple[1], triple[2]]
                        #print('exist', [N[i], triple[1], triple[2]])
                        N, all_unique_entity, M = change_again(all_unique_entity, C, M, N[i], M[list(all_unique_entity).index(N[i])])
                        #print('after chenging', [N[i], triple[1], triple[2]])
                        try:
                            check = model.all_pos[N[i], triple[1], triple[2]]
                        except:
                            break
                except:
                    continue

        temp_triple = np.copy(triple)
        temp_matrix = np.tile(temp_triple,(C,1))

        for i in range(len(temp_matrix)):
            temp_matrix[i][corrupt_ind] = N[i]
        #print(i)
        #temp_triple[corrupt_ind]=N[0]
        end = time()
        #print('time :{:.2f}s'.format(end - start))
        for i in range(len(temp_matrix)):
            negative_batch.append(temp_matrix[i])
        #print(negative_triple)
        #exit()
    #print(negative_batch)
    #print(np.array(positive_batch).shape)
    return np.array(negative_batch)


