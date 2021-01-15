from model_utilities.model_initialization import model
from utilities.negative_sampling import sample_negatives, sample_negatives_t,DNS_paper_modified, DNS_paper_orig, unique_negative_sampling
from utilities.evaluation_utils import *
from utilities.data_utilities import KnowledgeGraph
from utilities.data_utilities import get_minibatches
from utilities.loss_functions import *
from utilities.new_evaluation_prototype import *
#from dataset import KnowledgeGraph
import torch
import numpy as np
from time import time
from sklearn.utils import shuffle as skshuffle
import itertools
import os
import pandas as pd
from utilities.dataloader import TrainDataset
from utilities.dataloader import BidirectionalOneShotIterator
from utilities.model_saving_utilities import save_model



def train(name, model,data_dir='dataset/yago3_10/mapped',dim=200, batch_size=2750,
          lr=0.1,min_epoch=500, max_epoch=10, gamma=1, temp=0, negsample_num=8,rev_set=0,
          Optimal = False, lam=0.01, L = "L1",
          regul=False, train_with_groundings = False, neg_sampling = 'random' ,
          lam1 = 0, lam2 = 0, lam3=0, lam4 = 0, test_mode=False,
          saving =False, fifthopole = False):
    L = L
    neg_sampling_mode = neg_sampling
    print('#Training Started!')
    randseed = 9999
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    embedding_creation_start = time()
    fifthopole = fifthopole
    kg = KnowledgeGraph(data_dir=data_dir,rev_set=rev_set,fifthopole=fifthopole)
    embedding_creation_end = time()
    print('kg creation time taken ', str(embedding_creation_end-embedding_creation_start))
    #exit()
    train_pos = np.array(kg.training_triples)

    #ind = np.array(np.arange(len(train_pos)))
    #train_pos = np.c_[train_pos, ind]
    test_pos = np.array(kg.test_triples)
    n_triples = len(train_pos)
    #print(n_triples)

    n_relation = len(kg.relation_dict)
    n_entity = len(kg.entity_dict)
    n_times = len(kg.times_dict)
    n_locations = len(kg.locations_dict)

################this is called for model initialize model#########################
    model = model(name,kg=kg, embedding_dim=dim, batch_size=batch_size, learning_rate=lr, L=L, gamma=gamma, n_triples = n_triples, n_relation = n_relation, n_entity = n_entity,
                  n_times = n_times, n_locations = n_locations ,gpu=False, regul=regul, negative_adversarial_sampling = True,
                  temp = temp,train_with_groundings=train_with_groundings, fifthopole=fifthopole)

    solver = torch.optim.Adagrad(model.parameters(), model.learning_rate)
    #solver = torch.optim.Adam(model.parameters(), model.learning_rate)

###################################################################################################
    #Unused
    grounding1 = None
    grounding2 = None
    grounding3 = None
    grounding4 = None

    grounding1_size = 0
    grounding2_size = 0
    grounding3_size = 0
    grounding4_size = 0

    grounding_1 = []
    grounding_2 = []
    grounding_3 = []
    grounding_4 = []

    rule1_loss = 0
    rule2_loss = 0
    rule3_loss = 0
    rule4_loss = 0
    #Groundings are unncecessery here.
    rule_loss = 0
    grounding1_empty_flag, grounding2_empty_flag, grounding3_empty_flag, grounding4_empty_flag = True, True, True, True
    if model.train_with_groundings == True:

        grounding2 = np.loadtxt(os.path.join(data_dir, 'groundings_inverse.txt'))
        grounding1 = np.loadtxt(os.path.join(data_dir, 'groundings_implication.txt'))
        grounding4 = np.loadtxt(os.path.join(data_dir, 'groundings_equivalence.txt'))
        grounding3 = np.loadtxt(os.path.join(data_dir, 'groundings_symmetric.txt'))

        grounding4_size = len(grounding4) // (kg.n_training_triple // batch_size)
        grounding3_size = len(grounding3) // (kg.n_training_triple // batch_size)
        grounding2_size = len(grounding2) // (kg.n_training_triple // batch_size)
        grounding1_size = len(grounding1) // (kg.n_training_triple // batch_size)



        if len(grounding1)!=0:
            grounding1_empty_flag = False
        if len(grounding2)!=0:
            grounding2_empty_flag = False
        if len(grounding3)!=0:
            grounding3_empty_flag = False
        if len(grounding4)!=0:
            grounding4_empty_flag = False
#########################################################################################################
    #Unused
    all_triples = np.loadtxt(os.path.join(data_dir, 'train.txt'))

    all_triples = pd.DataFrame(all_triples)
#########################################################################################################
#unused
    unique_neg = None

    if neg_sampling_mode == 'unique':
        unique_neg = pd.read_table(os.path.join(data_dir,'table.tsv'))
    if neg_sampling_mode != 'random':
        prob_table = pd.read_table(os.path.join(data_dir, 'hpt_umls.txt'), names=['tph', 'hpt'],
                                  header=None)
################################################################################################################
    losses = []
    mrr_std = 0
    C = negsample_num
    dir_ext = ''
##################################################################################################################
#Unused
    if train_with_groundings == False:
        dir_ext = 'without_grounding'
    else:
        dir_ext = 'with_grounding'
################################################################################################################
    base_dir = data_dir
    res_dir_name = neg_sampling_mode + name + dir_ext + '_refined'
    data_dir = os.path.join(data_dir,'tansen_results')
    path = os.path.join(data_dir, res_dir_name)
    if os.path.isdir(path)==False:
        os.makedirs(path)

    epoch_start = time()
##############################################################################
    #For testing only use test mode = True
    if (test_mode == True):
        trained_model = torch.load(os.path.join(base_dir, 'trained_model/distmult_quad'))
        model.load_state_dict(trained_model['model_state_dict'])
        print('####loading trained model###')
        [rank, rank_filter] = left_rank_t(model, test_pos)
        # print('printing right rank')
        [rank_right, rank_filter_right] = right_rank_t(model, test_pos)
        rank = rank + rank_right
        rank_filter = rank_filter + rank_filter_right
        m_rank = mean_rank(rank)
        mean_rr = mrr(rank)
        hit_1 = hit_N(rank, 1)
        hit_3 = hit_N(rank, 3)
        hit_5 = hit_N(rank, 5)
        hit_10 = hit_N(rank, 10)
        m_rank_filter = mean_rank(rank_filter)
        mrr_filter = mrr(rank_filter)
        hit_1_filter = hit_N(rank_filter, 1)
        hit_3_filter = hit_N(rank_filter, 3)
        hit_5_filter = hit_N(rank_filter, 5)
        hit_10_filter = hit_N(rank_filter, 10)
        print('mrr filtered', mean_rr)
        print('hit@10 filtered', hit_10_filter)

        # exit()
#Training Start Here!
    for epoch in range(max_epoch):
        print('Epoch-{}'.format(epoch + 1))
        print('————————————————')
        it = 0
        train_triple = list(get_minibatches(train_pos, batch_size, shuffle=True))
##############################################################################################################
        #Unused
        if model.train_with_groundings == True:
            if grounding2_empty_flag == False:
                grounding_2 = (get_minibatches(grounding2, grounding2_size, shuffle=True))
            if grounding1_empty_flag == False:
                grounding_1 = (get_minibatches(grounding1, grounding1_size, shuffle=True))
            if grounding3_empty_flag == False:
                grounding_3 = (get_minibatches(grounding3, grounding3_size, shuffle=True))
            if grounding4_empty_flag == False:
                grounding_4 = (get_minibatches(grounding4, grounding4_size, shuffle=True))

#################################################################################################################
        for iter_triple, rule1, rule2, rule3, rule4  in itertools.zip_longest(train_triple, grounding_1, grounding_2, grounding_3, grounding_4  ):

            if iter_triple.shape[0] < batch_size:
                break
            start = time()

            iter_neg = None
            if neg_sampling == 'random' and fifthopole==True:

                iter_neg = sample_negatives_t(iter_triple, C)
            elif neg_sampling == 'random' and fifthopole==False:
                iter_neg = sample_negatives(iter_triple, C)
            ####################################################################################################################
            # Unused
            elif neg_sampling == 'dns_orig':
                iter_neg = DNS_paper_orig(iter_triple, C, model, all_triples, model.kg.n_entity, prob_table)
            elif neg_sampling == 'dns_mod':
                iter_neg = DNS_paper_modified(iter_triple, C, model, all_triples, model.kg.n_entity, prob_table)
            elif neg_sampling == 'unique':
                iter_neg = unique_negative_sampling(iter_triple, C, model, all_triples, model.kg.n_entity, prob_table, unique_neg)
######################################################

            pos_reg = None
            neg_reg = None
            forward_time_start = time()
            if  fifthopole == True:
                if (model.regul == True):
                    [pos_score, pos_reg] = model.forward_t(iter_triple)
                    [neg_score, neg_reg] = model.forward_t(iter_neg)
                else:
                    pos_score = model.forward_t(iter_triple)
                    neg_score= model.forward_t(iter_neg)
            else:
                if (model.regul == True):
                    [pos_score, pos_reg] = model.forward(iter_triple)
                    [neg_score, neg_reg] = model.forward(iter_neg)
                else:
                    pos_score = model.forward(iter_triple)
                    neg_score= model.forward(iter_neg)

            if (model.regul == True):
                loss = log_loss_adv_with_regularization(model, pos_score, neg_score, pos_reg, neg_reg, lam ,temp)
                #loss = binary_cross(model, pos_score, neg_score, pos_reg, neg_reg, lam ,temp)
            else:
                #loss = log_loss_adv(model ,pos_score, neg_score, temp=temp)
                #loss = rank_loss(model, pos_score ,neg_score, gamma ,temp)
                #loss = binary_cross(model,pos_score, neg_score, pos_reg, neg_reg, temp) #for complex use binary crossentropy loss
                loss = adversarial_loss(model,pos_score,neg_score, gamma , C, temp)
#############################################################################################
            #unused
            if model.train_with_groundings == True:

                if (model.name == 'LogicENN'):
                    if grounding1_empty_flag == False:
                        rule1_loss = torch.sum(rule_loss_implication_logicENN(model,groundings=rule1,lam=0.01))
                    if grounding2_empty_flag == False:
                        rule2_loss = torch.sum(rule_loss_inverse_logicENN(model,groundings=rule2, lam=0.01))

                    # rule2_loss = torch.sum(rule_model.rule1_loss(groundings=inv_to_imp, lam=lam1))
                    if grounding3_empty_flag == False:
                        rule3_loss = torch.sum(rule_loss_symmetric_logicENN(model,groundings=rule3,lam=lam3))
                    # rule3_loss = torch.sum(rule_model.rule4_loss(groundings=sym_ground, lam=lam4))
                    if grounding4_empty_flag == False:
                        rule4_loss = torch.sum(rule_loss_equivalence_logicENN(model,groundings=rule4,lam=0.01))
                    # rule5_loss = torch.sum(rule_model.rule5_loss(groundings=rule5,lam=lam5,a=alpha))
                else:
                    if grounding1_empty_flag == False:
                        rule1_loss = torch.sum(rule_loss_implication(model,groundings=rule1, lam=lam1))
                    if grounding2_empty_flag == False:
                        rule2_loss = torch.sum(rule_loss_inverse(model,groundings=rule2, lam=lam2))

                    if grounding3_empty_flag == False:
                        # rule2_loss = torch.sum(rule_model.rule1_loss(groundings=inv_to_imp, lam=lam1))
                        rule3_loss = torch.sum(rule_loss_symmetric(model,groundings=rule3, lam=lam3))
                        # rule3_loss = torch.sum(rule_model.rule4_loss(groundings=sym_ground, lam=lam4))
                    if grounding4_empty_flag == False:
                        rule4_loss = torch.sum(rule_loss_equivalence(model,groundings=rule4, lam=lam4))

                rule_loss = (rule1_loss + rule2_loss + rule3_loss + rule4_loss) / (
                            grounding1_size + grounding2_size + grounding3_size + grounding4_size)


###########################################################################################
            #loss = (loss_neg+loss_pos)/2
            total_loss = loss + lam * rule_loss
            losses.append(total_loss.item())

            solver.zero_grad()
            total_loss.backward()
            solver.step()
            if model.name == 'transE':
                model.normalize_embeddings()  # for transE you have to normalize
            elif model.name == 'transH':
                model.normalize_embeddings_tH()
            elif model.name == 'LogicENN':
                model.normalize_embeddings()
            end = time()

            if it % 33 == 0:
                #print('Iter-{}; loss: {:.4f}; rule loss {:.4f}; time per batch:{:.2f}s'.format(it, total_loss.item(), rule2_loss ,end - start))
                # print('Iter-{}; loss: {:.4f};rule1_loss: {:.4f};rule2_loss: {:.4f}; rule3_loss: {:.4f} ; rule4_loss: {:.4f};time per batch:{:.2f}s'.format(it,
                #                                                                                                   total_loss.item(),
                #                                                                                                   rule1_loss.item()/grounding1_size,
                #                                                                                                   rule2_loss.item()/grounding2_size,
                #                                                                                                   rule3_loss.item()/grounding3_size,
                #                                                                                                   rule4_loss.item()/grounding4_size,
                #                                                                                                   end - start))
                print('Iter-{}; loss: {:.4f};  time per batch:{:.2f}s'.format(it, total_loss.item(),
                                                                                              end - start))
            it += 1
        #print(model.emb_E.weight.data)
        #print(model.emb_R.weight.data)
        #exit()
        #if ((epoch+1)//min_epoch>epoch//min_epoch and epoch < max_epoch):
        #epoch_time_end = time()
        #print('per epoch taken ', str(epoch_time_end-epoch_time_start))
        if epoch+1 == max_epoch:
##################################################################################
#Unused
            if model.name == "LogicENN":
                epoch_end = time()
                if train_with_groundings == True:
                    saved_model_name_ext = 'with_grounding'
                else:
                    saved_model_name_ext = 'without_grounding'
                if saving == True:
                    saved_model_name = 'trained_model_' + model.name + '_' + saved_model_name_ext
                    model_save_path = os.path.join(base_dir, 'trained_model')
                    save_model(model, optimizer=solver, save_path=model_save_path, model_name=saved_model_name)
                print('training time ', str(epoch_end - epoch_start))
                # torch.save(model.state_dict(), os.path.join(path, 'params{:.0f}.pkl'.format(epoch)))
                f = open(os.path.join(path, 'result{:.0f}.txt'.format(epoch)), 'w')
                print('printing left rank')
                [rank, rank_filter] = rank_left_LogicENN(model, test_pos)
                print('printing right rank')
                [rank_right, rank_filter_right] = rank_right_LogicENN(model, test_pos)
                rank = rank + rank_right
                rank_filter = rank_filter + rank_filter_right
                m_rank = mean_rank(rank)
                mean_rr = mrr(rank)
                hit_1 = hit_N(rank, 1)
                hit_3 = hit_N(rank, 3)
                hit_5 = hit_N(rank, 5)
                hit_10 = hit_N(rank, 10)
                m_rank_filter = mean_rank(rank_filter)
                mrr_filter = mrr(rank_filter)
                hit_1_filter = hit_N(rank_filter, 1)
                hit_3_filter = hit_N(rank_filter, 3)
                hit_5_filter = hit_N(rank_filter, 5)
                hit_10_filter = hit_N(rank_filter, 10)
                # print('mean rank: ', m_rank)
                # print('mrr: ', mean_rr)
                # print('hit@10 :' ,hit_10)
                saved_model_name_ext = ''

                f.write('Mean Rank: {:.0f}, {:.0f}\n'.format(m_rank, m_rank_filter))
                f.write('Mean RR: {:.4f}, {:.4f}\n'.format(mean_rr, mrr_filter))
                f.write('Hit@1: {:.4f}, {:.4f}\n'.format(hit_1, hit_1_filter))
                f.write('Hit@3: {:.4f}, {:.4f}\n'.format(hit_3, hit_3_filter))
                f.write('Hit@5: {:.4f}, {:.4f}\n'.format(hit_5, hit_5_filter))
                f.write('Hit@10: {:.4f}, {:.4f}\n'.format(hit_10, hit_10_filter))
#####################################################################################
            epoch_end = time()
            print('training time ', str(epoch_end-epoch_start))
            #torch.save(model.state_dict(), os.path.join(path, 'params{:.0f}.pkl'.format(epoch)))
            f = open(os.path.join(path, 'result{:.0f}.txt'.format(epoch)), 'w')
            #print('printing left rank')
            if fifthopole == True:
                [rank, rank_filter] = left_rank_t(model,test_pos)
                #print('printing right rank')
                [rank_right, rank_filter_right] = right_rank_t(model,test_pos)
            else:
                [rank, rank_filter] = left_rank(model, test_pos)
                # print('printing right rank')
                [rank_right, rank_filter_right] = right_rank(model, test_pos)
            rank = rank + rank_right
            rank_filter = rank_filter + rank_filter_right
            m_rank = mean_rank(rank)
            mean_rr = mrr(rank)
            hit_1 = hit_N(rank, 1)
            hit_3 = hit_N(rank, 3)
            hit_5 = hit_N(rank, 5)
            hit_10 = hit_N(rank, 10)
            m_rank_filter = mean_rank(rank_filter)
            mrr_filter = mrr(rank_filter)
            hit_1_filter = hit_N(rank_filter, 1)
            hit_3_filter = hit_N(rank_filter, 3)
            hit_5_filter = hit_N(rank_filter, 5)
            hit_10_filter = hit_N(rank_filter, 10)
            #print('mean rank: ', m_rank)
            #print('mrr: ', mean_rr)
            #print('hit@10 :' ,hit_10)
            saved_model_name_ext = ''
            if train_with_groundings == True:
                saved_model_name_ext = 'with_grounding'
            else:
                saved_model_name_ext = 'without_grounding'
            if saving == True:
                saved_model_name = 'trained_model_' + model.name + '_' + saved_model_name_ext
                model_save_path = os.path.join(base_dir, 'trained_model')
                save_model(model, optimizer=solver, save_path=model_save_path, model_name = saved_model_name)
            f.write('Mean Rank: {:.0f}, {:.0f}\n'.format(m_rank, m_rank_filter))
            f.write('Mean RR: {:.4f}, {:.4f}\n'.format(mean_rr, mrr_filter))
            f.write('Hit@1: {:.4f}, {:.4f}\n'.format(hit_1, hit_1_filter))
            f.write('Hit@3: {:.4f}, {:.4f}\n'.format(hit_3, hit_3_filter))
            f.write('Hit@5: {:.4f}, {:.4f}\n'.format(hit_5, hit_5_filter))
            f.write('Hit@10: {:.4f}, {:.4f}\n'.format(hit_10, hit_10_filter))
            # model = model(name, kg=kg, embedding_dim=dim, batch_size=batch_size, learning_rate=lr, L='L1', gamma=gamma,
            #               n_triples=n_triples, n_relation=n_relation, n_entity=n_entity, gpu=True, regul=regul,
            #               negative_adversarial_sampling=True, temp=temp, train_with_groundings=train_with_groundings)
            #exit()
            #model = save_model
            #model = model.load_state_dict(torch.load('/home/mirza/PycharmProjects/frame_work/dataset/kinship/trained_model/params49.pkl'))
            # trained_model = torch.load(os.path.join(base_dir, 'trained_model/trained_model'))
            # model.load_state_dict(trained_model['model_state_dict'])
            # print('####loading trained model###')
            # [rank, rank_filter] = left_rank(model, test_pos)
            # # print('printing right rank')
            # [rank_right, rank_filter_right] = right_rank(model, test_pos)
            # rank = rank + rank_right
            # rank_filter = rank_filter + rank_filter_right
            # m_rank = mean_rank(rank)
            # mean_rr = mrr(rank)
            # hit_1 = hit_N(rank, 1)
            # hit_3 = hit_N(rank, 3)
            # hit_5 = hit_N(rank, 5)
            # hit_10 = hit_N(rank, 10)
            # m_rank_filter = mean_rank(rank_filter)
            # mrr_filter = mrr(rank_filter)
            # hit_1_filter = hit_N(rank_filter, 1)
            # hit_3_filter = hit_N(rank_filter, 3)
            # hit_5_filter = hit_N(rank_filter, 5)
            # hit_10_filter = hit_N(rank_filter, 10)
            # print('mrr', mean_rr)
            # print('hit@10', hit_10_filter)
            # for loss in losses:
            #     f.write(str(loss))
            #     f.write('\n')
            # f.close()
            # losses = []

    #return [rank, rank_filter]


def main():
    #lr = 0.1
    lr = 0.5
    #################################################################
    #unused in my case
    lam = 0.1
    lam2 = 3
    lam3 = 0.001
    lam4 = 0.1
    #dim = 200
    ###################################################################
    negsample_num = 10
    dim = 200
    gamma = 1000
    #gamma = 10
    temp = 0
    # for lr in [0.5]:
    #     for lam in [0.01]: #0.05 for others
    #         for lam1 in [0.0]:
    #             for lam2 in [3]: #3,7 prev, #0.1 for fb15k
    #                 for lam3 in [0.001]: #0.01 for fb15k
    #                     for lam4 in [0.1]: #0.1 prev
    #                         for dim in [200]:
    #                             for negsample_num in [10]:
    #                                 for gamma in [24]: #gamma 10 for umls
    #                                     for temp in [0]:


# ##########################complex_quad##############################
    print("********************complex_quad*******************************8")
    train( name='complEx_quad', model = model,dim=dim, lr=lr, negsample_num=negsample_num,
        gamma=gamma, temp=temp, lam  = lam, lam2=lam2, lam3 =lam3, lam4=lam4,
        regul=True, max_epoch= 200,
        test_mode=False, saving=False, fifthopole = True, batch_size=2750, data_dir='dataset/yago3_10/mapped', L = 'L2')

    ####################### for triple complEx##############################
    # print("********************running*******************************8")
    # train(name='complEx', model=model, dim=dim, lr=lr, negsample_num=negsample_num,
    #       gamma=gamma, temp=temp, lam=lam, lam2=lam2, lam3=lam3, lam4=lam4,
    #       regul=True, max_epoch=200,
    #       test_mode=False, saving=False, fifthopole=False, batch_size=2750, data_dir='dataset/yago3_10/mapped_two')


   ##############################dismult_quad#########################

    # print("********************dismult_quad*******************************8")
    # train( name='distmult_quad', model = model,dim=dim, lr=lr, negsample_num=negsample_num,
    #     gamma=gamma, temp=temp, lam  = lam, lam2=lam2, lam3 =lam3, lam4=lam4,
    #     regul=False, max_epoch= 200,
    #     test_mode=False, saving=True, fifthopole = True, batch_size=2750, data_dir='dataset/yago3_10/mapped', L = 'L2')


    # print("********************dismult*******************************8")
    # train( name='distmult', model = model,dim=dim, lr=lr, negsample_num=negsample_num,
    #     gamma=gamma, temp=temp, lam  = lam, lam2=lam2, lam3 =lam3, lam4=lam4,
    #     regul=False, max_epoch= 200,
    #     test_mode=False, saving=True, fifthopole = False, batch_size=2750, data_dir='dataset/yago3_10/mapped', L = 'L2')


    # print("********************transH_element_quad*******************************8")
    # train( name='transH_element_quad', model = model,dim=dim, lr=lr, negsample_num=negsample_num,
    #     gamma=gamma, temp=temp, lam  = lam, lam2=lam2, lam3 =lam3, lam4=lam4,
    #     regul=False, max_epoch= 200,
    #     test_mode=False, saving=True, fifthopole = True, batch_size=2750, data_dir='dataset/yago3_10/mapped', L = 'L2')

    # print("********************running*******************************8")
    # train( name='transH_element', model = model,dim=dim, lr=lr, negsample_num=negsample_num,
    #     gamma=gamma, temp=temp, lam  = lam, lam2=lam2, lam3 =lam3, lam4=lam4,
    #     regul=False, max_epoch= 200,
    #     test_mode=False, saving=True, fifthopole = False, batch_size=2750, data_dir='dataset/yago3_10/mapped', L = 'L2')
    #

    ########################### for QuadThrople ######################################

    # print("********************running*******************************8")
    # train( name='transE_quad', model = model,dim=dim, lr=lr, negsample_num=negsample_num,
    #     gamma=gamma, temp=temp, lam  = lam, lam2=lam2, lam3 =lam3, lam4=lam4,
    #     regul=False, max_epoch= 200,
    #     test_mode=False, saving=True, fifthopole = True, batch_size=2750, data_dir='dataset/yago3_10/mapped')

######################## for triple transH##############################

    # print("********************running*******************************8")
    # train(name='transH', model=model, dim=dim, lr=lr, negsample_num=negsample_num,
    #       gamma=gamma, temp=temp, lam=lam, lam2=lam2, lam3=lam3, lam4=lam4,
    #       regul=False, max_epoch=200, L =  'L1',
    #       test_mode=False, saving=True, fifthopole=False, batch_size=2750, data_dir='dataset/yago3_10/mapped')

    ######################## for triple transE##############################
    # print("********************running*******************************8")
    # train(name='transE', model=model, dim=dim, lr=lr, negsample_num=negsample_num,
    #       gamma=gamma, temp=temp, lam=lam, lam2=lam2, lam3=lam3, lam4=lam4,
    #       regul=False, max_epoch=200,
    #       test_mode=False, saving=True, fifthopole=False, batch_size=2750, data_dir='dataset/yago3_10/mapped_two')



if __name__ == '__main__':
    main()