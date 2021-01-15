from model_utilities.model_initialization import model
from utilities.negative_sampling import sample_negatives
from utilities.evaluation_utils import *
from utilities.new_evalulation import *
from utilities.data_utilities import KnowledgeGraph
from utilities.data_utilities import get_minibatches
from utilities.loss_functions import *
import torch
from torch.utils.data import DataLoader
import numpy as np
from time import time
import itertools
import os
import argparse
import pandas as pd
from utilities.negative_sampling import sample_negatives, DNS_paper_modified, DNS_paper_orig


def train(name, model,data_dir,dim=100, batch_size=500, lr=0.1, max_epoch=100, gamma=15, temp=1, negsample_num=10,rev_set=0, lam=0.01,
          regul=False, train_with_groundings = False, neg_sampling = 'random' ,  lam1 = 0, lam2 = 0, lam3=0, lam4 = 0):
    neg_sampling_mode = neg_sampling
    print('#Training Started!')
    randseed = 9999
    np.random.seed(randseed)
    torch.manual_seed(randseed)

    kg = KnowledgeGraph(data_dir=data_dir, rev_set=rev_set)
    train_pos = np.array(kg.training_triples)
    val_pos = np.array(kg.validation_triples)

    args.nentity = kg.n_entity
    args.nrelation = kg.n_relation

    test_pos = np.array(kg.test_triples)
    n_triples = len(train_pos)
    gpu = False
    if args.cuda == True:
        gpu = True
    #batch = kg.n_training_triple//n_batch
    model = model(name,kg=kg, embedding_dim=dim, batch_size=batch_size, learning_rate=lr, L='L1', gamma=gamma, n_triples = n_triples,gpu=True, regul=regul, negative_adversarial_sampling = True, temp = temp,train_with_groundings=train_with_groundings)
    #solver = torch.optim.SGD(model.parameters(), model.learning_rate, momentum=0.9)
    solver = torch.optim.Adagrad(model.parameters(), model.learning_rate)

    #   load rule groundings in datasets
    #   grounding1 are groundings of implication rules
    #   grounding2 are groundings of inverse rules
    #   grounding3 are groundings of symmetric rules
    #   grounding4 are groundings of equalivency rules
    #   grounding5 are groundings of transitivity rules

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

    rule_loss = 0

    if model.train_with_groundings == True:
        grounding2 = np.loadtxt(os.path.join(data_dir, 'groundings_inverse.txt'))
        grounding1 = np.loadtxt(os.path.join(data_dir, 'groundings_implication.txt'))
        grounding4 = np.loadtxt(os.path.join(data_dir, 'groundings_equivalence.txt'))
        #grounding5 = np.loadtxt(os.path.join(data_dir, 'groundings_compos.txt'))
        grounding3 = np.loadtxt(os.path.join(data_dir, 'groundings_symmetric.txt'))

        grounding4_size = len(grounding4) // (kg.n_training_triple // batch_size)
        grounding3_size = len(grounding3) // (kg.n_training_triple // batch_size)
        grounding2_size = len(grounding2) // (kg.n_training_triple // batch_size)
        grounding1_size = len(grounding1) // (kg.n_training_triple // batch_size)
    #grounding5_size = len(grounding5) // (kg.n_training_triple // n_batch)

    #val_pos = np.array(kg.validation_triples)

    if neg_sampling!='random':
        all_triples = np.loadtxt(os.path.join(data_dir, 'original_quadropoles.txt'))
        # all_triples = np.load(os.path.join(data_dir, 'train.npy'))
        all_triples = pd.DataFrame(all_triples)
        prob_table = pd.read_table(os.path.join(data_dir, 'hpt_nations.txt'), names=['tph', 'hpt'],
                                  header=None)

    losses = []
    mrr_std = 0
    C = negsample_num
    # path = os.path.join(data_dir,'fb15k237_old_rotatE_500_epoch/embedding_dim{:.0f}/batch{:.0f}/lr{:.4f}/negsample{:.0f}/gamma{:.1f}'
    #                              '/temp{:.3f}/rev_set{:.0f}'.format(dim, n_batch, lr, negsample_num,gamma,temp,int(rev_set)))
    # os.makedirs(path)
    res_dir_name = neg_sampling_mode + name + '200'
    path = os.path.join(data_dir, res_dir_name)
    os.makedirs(path)

    for epoch in range(max_epoch):
        print('Epoch-{}'.format(epoch + 1))
        print('————————————————')
        it = 0
        train_triple = list(get_minibatches(train_pos, batch_size, shuffle=True))

        if model.train_with_groundings == True:
            grounding_2 = (get_minibatches(grounding2, grounding2_size, shuffle=True))
            grounding_1 = (get_minibatches(grounding1, grounding1_size, shuffle=True))
            grounding_3 = (get_minibatches(grounding3, grounding3_size, shuffle=True))
            grounding_4 = (get_minibatches(grounding4, grounding4_size, shuffle=True))
        #grounding_5 = (get_minibatches(grounding5, grounding5_size, shuffle=True))
        #for iter_triple in train_triple:
        #for iter_triple in train_triple:
        for iter_triple, rule1, rule2, rule3 ,rule4 in itertools.zip_longest(train_triple, grounding_1, grounding_2, grounding_3 , grounding_4 ):
            #print(iter_triple.shape)
            #exit()
            #print('here')
            #print(train_pos.shape)
            #batch_indexes = np.where((train_pos== iter_triple[:, None]).all(-1))[1]
            #print(batch_indexes)
            #exit()
            if iter_triple.shape[0] < batch_size:
                break
            start = time()
            #TODO
            #create negative sampling mode modular
            #print(iter_triple.shape)
            iter_neg = None
            if neg_sampling == 'random':
                iter_neg = sample_negatives(iter_triple, C)
            elif neg_sampling == 'dns_orig':
                iter_neg = DNS_paper_orig(iter_triple, C, model, all_triples, model.kg.n_entity, prob_table)
            elif neg_sampling == 'dns_mod':
                iter_neg = DNS_paper_modified(iter_triple, C, model, all_triples, model.kg.n_entity, prob_table)
            #pos_batch_ind = iter_triple[:, -1].squeeze()
            #pos_batch_ind = iter_triple[:, -1].squeeze()
            #neg_batch_ind = np.tile(pos_batch_ind,C)
            #print(pos_batch_ind)
            #print(neg_batch_ind)
            #print(iter_neg.shape)
            #iter_neg = DNS_paper_orig(iter_triple, C, model, all_triples, model.kg.n_entity, prob_table)
            #print(iter_triple[:,:-1].shape)
            #print(iter_neg[:,:].shape)
            #exit()
            pos_reg = None
            neg_reg = None
            if (model.regul == True):
                #[pos_score, pos_reg] = model.forward(iter_triple[:,:-1])
                [pos_score, pos_reg] = model.forward(iter_triple)
                [neg_score, neg_reg] = model.forward(iter_neg)
            else:
                pos_score = model.forward(iter_triple)
                #[pos_score, pos_reg] = model.forward(iter_triple)
                neg_score= model.forward(iter_neg)
            #print(pos_score.size())
            #print(neg_score.size())
            #exit()
            #loss_pos = model.log_rank_loss(pos_score, C=1, gamma=gamma, neg=-1, temp=0)
            #loss_neg = model.log_rank_loss(neg_score, C=C, gamma=gamma, neg=1, temp=temp)

            if (model.regul == True):
                #loss = log_loss_adv_with_regularization(model, pos_score, neg_score, pos_reg, neg_reg, lam ,temp)
                loss = binary_cross(model, pos_score, neg_score, pos_reg, neg_reg, lam ,temp)
            else:
                #loss = log_loss_adv(model ,pos_score, neg_score, temp=temp)
                #loss = rank_loss(model, pos_score ,neg_score, gamma ,temp)
                #loss = binary_cross(model,pos_score, neg_score, pos_reg, neg_reg, temp) #for complex use binary crossentropy loss
                loss = adversarial_loss(model,pos_score,neg_score, gamma , C, temp)
                #loss, positive_sample_loss, negative_sample_loss = Adaptive_Margin_Loss_P2(model, pos_score, neg_score, pos_batch_ind ,neg_batch_ind, C)
                #print(loss)
                #exit()

            if model.train_with_groundings == True:
                if (model.name == 'LogicENN'):
                    rule1_loss = torch.sum(rule_loss_implication_logicENN(model,groundings=rule1,lam=0.01))
                    rule2_loss = torch.sum(rule_loss_inverse_logicENN(model,groundings=rule2, lam=0.01))
                    # rule2_loss = torch.sum(rule_model.rule1_loss(groundings=inv_to_imp, lam=lam1))
                    rule3_loss = torch.sum(rule_loss_symmetric_logicENN(groundings=rule3,lam=lam3))
                    # rule3_loss = torch.sum(rule_model.rule4_loss(groundings=sym_ground, lam=lam4))
                    rule4_loss = torch.sum(rule_loss_equivalence_logicENN(model,groundings=rule4,lam=0.01))
                    # rule5_loss = torch.sum(rule_model.rule5_loss(groundings=rule5,lam=lam5,a=alpha))
                else:

                    rule1_loss = torch.sum(rule_loss_implication(model,groundings=rule1, lam=lam1))
                    rule2_loss = torch.sum(rule_loss_inverse(model,groundings=rule2, lam=lam2))
                    # rule2_loss = torch.sum(rule_model.rule1_loss(groundings=inv_to_imp, lam=lam1))
                    rule3_loss = torch.sum(rule_loss_symmetric(model,groundings=rule3, lam=lam3))
                    # rule3_loss = torch.sum(rule_model.rule4_loss(groundings=sym_ground, lam=lam4))
                    rule4_loss = torch.sum(rule_loss_equivalence(model,groundings=rule4, lam=lam4))

                rule_loss = (rule1_loss + rule2_loss + rule3_loss + rule4_loss) / (
                            grounding1_size + grounding2_size + grounding3_size + grounding4_size)

            #print(rule4_loss)
            #rule5_loss = torch.sum(rule5_loss(model, groundings=rule5, lam=0))
            #print(rule5_loss)



            #loss = (loss_neg+loss_pos)/2
            total_loss = loss + lam * rule_loss
            losses.append(total_loss.item())

            solver.zero_grad()
            total_loss.backward()
            solver.step()
            if model.name == 'transE':
                model.normalize_embeddings() #for transE you have to normalize
            end = time()

            if it % 33 == 0:
                #print('Iter-{}; loss: {:.4f}; rule loss {:.4f}; time per batch:{:.2f}s'.format(it, total_loss.item(), rule_loss ,end - start))
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

        if (epoch + 1) % eval_log_steps == 0:
            #evaluation with test dataloader...
            print('Evaluating on validation Dataset...')
            metrics = test_step(model,  val_pos, kg.all_triples, args)
            log_metrics('Test', epoch, metrics)

        #if ((epoch+1)//min_epoch>epoch//min_epoch and epoch < max_epoch):
        if epoch+1 == max_epoch:

            print('Evaluating on Test Dataset...')
            metrics = test_step(model, test_pos, kg.all_triples, args)
            log_metrics('Test', epoch, metrics)

            torch.save(model.state_dict(), os.path.join(path, 'params{:.0f}.pkl'.format(epoch)))
            f = open(os.path.join(path, 'result{:.0f}.txt'.format(epoch)), 'w')
            print('printing left rank')
            [rank, rank_filter] = left_rank(model,test_pos)
            print('printing right rank')
            [rank_right, rank_filter_right] = right_rank(model,test_pos)
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
            f.write('Mean Rank: {:.0f}, {:.0f}\n'.format(m_rank, m_rank_filter))
            f.write('Mean RR: {:.4f}, {:.4f}\n'.format(mean_rr, mrr_filter))
            f.write('Hit@1: {:.4f}, {:.4f}\n'.format(hit_1, hit_1_filter))
            f.write('Hit@3: {:.4f}, {:.4f}\n'.format(hit_3, hit_3_filter))
            f.write('Hit@5: {:.4f}, {:.4f}\n'.format(hit_5, hit_5_filter))
            f.write('Hit@10: {:.4f}, {:.4f}\n'.format(hit_10, hit_10_filter))
            for loss in losses:
                f.write(str(loss))
                f.write('\n')
            f.close()
            losses = []

    #return [rank, rank_filter]


def main():


    train(name=model_name, model=model, data_dir=dataset, dim=dim, batch_size=batch_size, lr=lr, max_epoch=epochs,
          negsample_num=neg_sample, regul=regularization, train_with_groundings=False)
    '''for lr in [lr]:
        for dim in [dim]:
            for b_size in [batch_size]:
                for negsample_num in [10]:
                    for gamma in [24]:
                        for temp in [0]:
                            print('running')'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "-m", help="name of the model")
    parser.add_argument("--dataset", help="name of the dataset")
    parser.add_argument("--lr", "-l", help="learning rate")
    parser.add_argument("--dimension", "-d", help="dimension of embeddings")
    parser.add_argument("--batch_size", "-b", help="batch size")
    parser.add_argument("--neg_sample", "-n", help="negative sampling size")
    parser.add_argument("--epochs", "-e", help="maximum epochs to be done")
    parser.add_argument('--regul', action='store_true', help='Use L1 Regularization or not')
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--eval_log_steps', default=50, type=int)

    #TODO: for the sake of running the new evaluation technique first... normally below argument is useless
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    args = parser.parse_args()

    model_name = args.model_name
    dataset = "dataset/" + str(args.dataset)
    lr = float(args.lr)
    dim = int(args.dimension)
    batch_size = int(args.batch_size)
    neg_sample = int(args.neg_sample)
    epochs = int(args.epochs)
    eval_log_steps = args.eval_log_steps
    regularization = False
    if args.regul:
        regularization = True

    print("Given hyperparameters")
    print("Model name: ", model_name)
    print("Dataset: ", dataset)
    print("Learning rate: ", lr)
    print("Embedding dimension: ", dim)
    print("Batch size: ", batch_size)
    print("Negative Sampling size: ", neg_sample)
    print("Max epochs: ", epochs)
    print("Regularization: ", regularization)

    main()