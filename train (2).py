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

def train(name, model,data_dir,dim=100, batch_size=2750, lr=0.1,min_epoch=500, max_epoch=200, gamma=1, temp=0, negsample_num=10,rev_set=0,Optimal = False, lam=0.01,
          regul=False, train_with_groundings = False):
    print('#Training Started!')
    randseed = 9999
    np.random.seed(randseed)
    torch.manual_seed(randseed)

    kg = KnowledgeGraph(data_dir=data_dir,rev_set=rev_set)
    train_pos = np.array(kg.training_triples)
    val_pos = np.array(kg.validation_triples)

    args.nentity = kg.n_entity
    args.nrelation = kg.n_relation

    test_pos = np.array(kg.test_triples)
    n_triples = len(train_pos)
    gpu = False
    if args.cuda == True:
        gpu = True
    model = model(name,kg=kg, embedding_dim=dim, batch_size=batch_size, learning_rate=lr, L='L1', gamma=gamma, n_triples = n_triples,gpu=gpu, regul=regul, negative_adversarial_sampling = True, temp = temp,train_with_groundings=train_with_groundings)
    #solver = torch.optim.Adam(model.parameters(), model.learning_rate)
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
        # grounding3 = np.loadtxt(os.path.join(data_dir, 'groundings_syms.txt'))
        grounding4_size = len(grounding4) // (kg.n_training_triple // batch_size)
        grounding2_size = len(grounding2) // (kg.n_training_triple // batch_size)
        grounding1_size = len(grounding1) // (kg.n_training_triple // batch_size)
    #grounding5_size = len(grounding5) // (kg.n_training_triple // n_batch)

    #    val_pos = np.array(kg.validation_triples)
    #all_triples = np.loadtxt(os.path.join(data_dir, 'original_quadropoles.txt'))
    # all_triples = np.load(os.path.join(data_dir, 'train.npy'))
    #all_triples = pd.DataFrame(all_triples)
    #prob_table = pd.read_table(os.path.join(data_dir, 'hpt_nations.txt'), names=['tph', 'hpt'],
    #                           header=None)

    grounding1 = np.loadtxt(os.path.join(data_dir, 'groundings_implication.txt'))
    grounding2 = np.loadtxt(os.path.join(data_dir, 'groundings_inverse.txt'))
    grounding3 = np.loadtxt(os.path.join(data_dir, 'groundings_symmetric.txt'))
    grounding4 = np.loadtxt(os.path.join(data_dir, 'groundings_equivalence.txt'))

    if model.name == 'ruge':


        all_groundings_map = {}
        #groundings are saved as such format: e1,e2,r1,r2,confidence

        grounding1 = np.loadtxt(os.path.join(data_dir, 'groundings_implication.txt'))
        for grounding in grounding1:
            #for IMPLICATION, premise=e1,e2,r1, conclusion=e1,e2,r2
            all_groundings_map[grounding[0], grounding[1], grounding[2]] = ([grounding[0],grounding[1],grounding[3] ] ,grounding[4])

        grounding2 = np.loadtxt(os.path.join(data_dir, 'groundings_inverse.txt'))
        for grounding in grounding2:
            #for INVERSE, premise=e1,e2,r1, conclusion=e2,e1,r2
            all_groundings_map[grounding[0], grounding[1], grounding[2]] = ([grounding[1],grounding[0],grounding[3] ] ,grounding[4])

        grounding3 = np.loadtxt(os.path.join(data_dir, 'groundings_symmetric.txt'))
        for grounding in grounding3:
            #for SYMMETRIC, premise=e1,e2,r1, conclusion=e2,e1,r1
            all_groundings_map[grounding[0], grounding[1], grounding[2]] = ([grounding[1],grounding[0],grounding[2] ] ,grounding[4])

        #TODO: UNDERSTAND EQUIVALENCE RULE AND IMPLEMENT LATER ON...
        grounding4 = np.loadtxt(os.path.join(data_dir, 'groundings_equivalence.txt'))


        if grounding1.size == 0:
            grounding1 = np.array([], dtype=np.float).reshape(0,5)
        if grounding2.size == 0:
            grounding2 = np.array([], dtype=np.float).reshape(0, 5)
        if grounding3.size == 0:
            grounding3 = np.array([], dtype=np.float).reshape(0,5)
        if grounding4.size == 0:
            grounding4 = np.array([], dtype=np.float).reshape(0, 5)
        all_groundings = np.concatenate((grounding1, grounding2, grounding3, grounding4), axis=0).astype(np.float64)


    losses = []
    mrr_std = 0
    C = negsample_num
    # path = os.path.join(data_dir,'fb15k237_old_rotatE_500_epoch/embedding_dim{:.0f}/batch{:.0f}/lr{:.4f}/negsample{:.0f}/gamma{:.1f}'
    #                              '/temp{:.3f}/rev_set{:.0f}'.format(dim, n_batch, lr, negsample_num,gamma,temp,int(rev_set)))
    # os.makedirs(path)

    # CHANGE
    # path = os.path.join(data_dir, 'rotatE_fb15k_with_noise rem 8')
    filename = model.name +"_"+ data_dir
    path = os.path.join(data_dir, filename)
    # CHANGE

    #TODO uncomment below to create a directory.. comment it during debugging
    #os.makedirs(path)

    #make a grounding map to their confidence scores to check their existence when it is necessary and also easily fetch confidence
    #given a triple



    for epoch in range(max_epoch):
        print('Epoch-{}'.format(epoch + 1))
        print('--------------')
        it = 0
        train_triple = list(get_minibatches(train_pos, batch_size, shuffle=True))

        if model.train_with_groundings == True:
            grounding_2 = (get_minibatches(grounding2, grounding2_size, shuffle=True))
            grounding_1 = (get_minibatches(grounding1, grounding1_size, shuffle=True))
            grounding_4 = (get_minibatches(grounding4, grounding4_size, shuffle=True))
        #grounding_5 = (get_minibatches(grounding5, grounding5_size, shuffle=True))

        for iter_triple, rule1, rule2, rule4 in itertools.zip_longest(train_triple, grounding_1, grounding_2, grounding_4 ):
            #only take the groundings whose premises are present in the current batch
            if model.name == 'ruge':
                filtered_groundings_premises_conf = []
                filtered_groundings_conclusions = []
                for triple in iter_triple:
                    try:
                        filtered_groundings_premises_conf.append(np.append(triple,all_groundings_map[triple[0], triple[1], triple[2]][1])) #triple(premise), confidence
                        filtered_groundings_conclusions.append(all_groundings_map[triple[0],triple[1],triple[2]][0]) #get the conclusion triple
                    except:
                        continue


            if iter_triple.shape[0] < batch_size:
                break
            start = time()
            #TODO
            #create negative sampling mode modular
            #print(iter_triple.shape)
            iter_neg = sample_negatives(iter_triple, C)
            #pos_batch_ind = iter_triple[:, -1].squeeze()
            #neg_batch_ind = np.tile(pos_batch_ind,C)
            #print(pos_batch_ind)
            #print(neg_batch_ind)
            #print(iter_neg.shape)
            #iter_neg = DNS_paper_modified(iter_triple, C, model, all_triples, model.kg.n_entity, prob_table)
            #print(iter_triple[:,:-1].shape)
            #print(iter_neg[:,:].shape)
            #exit()
            pos_reg = None
            neg_reg = None

            #SCORINGS
            if (model.regul == True):
                [pos_score, pos_reg] = model.forward(iter_triple)
                [neg_score, neg_reg] = model.forward(iter_neg)
            else:
                pos_score = model.forward(iter_triple)
                neg_score= model.forward(iter_neg)

            if model.name == 'ruge':
                #calculating unlabeled scores...
                filtered_groundings_premises_conf = np.asarray(filtered_groundings_premises_conf)
                filtered_groundings_conclusions = np.asarray(filtered_groundings_conclusions)
                #get the unlabeled scores via forwarding conclusion (unlabeled) triples.
                if filtered_groundings_conclusions.size > 0:
                    has_grounding = True

                    unlabeled_scores = model.forward(filtered_groundings_conclusions)
                    #calculating the soft_labels...
                    premise_scores = model.forward(filtered_groundings_premises_conf[:,:-1])
                    if model.regul == True:
                        unlabeled_scores = unlabeled_scores[0]
                        premise_scores = premise_scores[0]
                    confidences = torch.tensor(filtered_groundings_premises_conf[:, -1])
                    if gpu:
                        confidences = confidences.cuda()

                    #TODO set the below 0.5 to C and make it a variable across the code
                    soft_labels =  0.01 * confidences * premise_scores + unlabeled_scores
                    soft_labels = torch.clamp(soft_labels, min=0, max=1)
                else:
                    #print("No filtered conclusions...")
                    has_grounding = False
                    unlabeled_scores = 0
                    soft_labels = 0

                #for later calculating loss later on...
                y_1s = torch.ones(pos_score.shape)
                y_0s = torch.zeros(neg_score.shape)
                labeled_targets = torch.cat(tensors=(y_1s, y_0s))
                if gpu:
                    labeled_targets = labeled_targets.cuda()
                #then give sx and concl_score to loss functions
                #unlabeled_premise = model.forward()
                #unlabeled_conclusion = model



            #LOSS CALCULATION BELOW SPECIFY WHICH LOSS YOU WANT TO USE...
            if (model.regul == True):
                #loss = log_loss_adv_with_regularization(model, pos_score, neg_score, pos_reg, neg_reg, lam ,temp)
                #loss = binary_cross(model, pos_score, neg_score, pos_reg, neg_reg, lam ,temp)
                loss, labeled_loss, unlabeled_loss = ruge_total_loss(pos_scores=pos_score,
                                                                     neg_scores=neg_score,
                                                                     labeled_targets=labeled_targets,
                                                                     unlabeled_scores=unlabeled_scores,
                                                                     soft_labels=soft_labels,
                                                                     has_grounding=has_grounding,
                                                                     regularization=(lam, pos_reg, neg_reg))
            else:
                #loss = log_loss_adv(model ,pos_score, neg_score, temp=temp)
                #loss = rank_loss(model, pos_score ,neg_score, gamma ,temp)
                #loss = binary_cross(model, pos_score, neg_score, pos_reg, neg_reg,lam=lam,temp=temp)  # for complex use binary crossentropy loss
                #loss = adversarial_loss(model, pos_score, neg_score, gamma, C, temp)
                #loss = log_loss_adv_with_regularization(model, pos_score, neg_score, pos_reg, neg_reg, lam, temp)
                #loss = cross_entropy(pos_scores=pos_score, neg_scores=neg_score)

                loss, labeled_loss, unlabeled_loss = ruge_total_loss(pos_scores=pos_score,
                                               neg_scores=neg_score,
                                               labeled_targets=labeled_targets,
                                               unlabeled_scores=unlabeled_scores,
                                               soft_labels=soft_labels,
                                               has_grounding=has_grounding)
                #loss = adversarial_loss(model,pos_score,neg_score, gamma , C, temp)
                #loss = binary_cross(self=model,y_pos=pos_score, y_neg=neg_score,reg_pos=0, reg_neg=0,lam=0,temp=0)
                #loss, positive_sample_loss, negative_sample_loss = Adaptive_Margin_Loss_P2(model, pos_score, neg_score, pos_batch_ind ,neg_batch_ind, C)
                #exit()'''

            if model.train_with_groundings == True:
                if (model.name == 'LogicENN'):
                    rule1_loss = torch.sum(rule_loss_implication_logicENN(model,groundings=rule1,lam=0.01))
                    rule2_loss = torch.sum(rule_loss_inverse_logicENN(model,groundings=rule2, lam=0.01))
                    # rule2_loss = torch.sum(rule_model.rule1_loss(groundings=inv_to_imp, lam=lam1))
                    #rule3_loss = torch.sum(rule_loss_symmetric_logicENN(groundings=rule3,lam=lam3))
                    # rule3_loss = torch.sum(rule_model.rule4_loss(groundings=sym_ground, lam=lam4))
                    rule4_loss = torch.sum(rule_loss_equivalence_logicENN(model,groundings=rule4,lam=0.01))
                    # rule5_loss = torch.sum(rule_model.rule5_loss(groundings=rule5,lam=lam5,a=alpha))
                else:
                    rule1_loss = torch.sum(rule_loss_implication(model,groundings=rule1, lam=0))
                    #print(rule1_loss)
                    rule2_loss = torch.sum(rule_loss_inverse(model,groundings=rule2, lam=0))
                    #print(rule2_loss)
                    rule4_loss = torch.sum(rule_loss_equivalence(model, groundings=rule4, lam=0))

                rule_loss = (rule1_loss + rule2_loss + rule4_loss) / (
                            grounding1_size + grounding2_size + grounding4_size)

            #print(rule4_loss)
            #rule5_loss = torch.sum(rule5_loss(model, groundings=rule5, lam=0))
            #print(rule5_loss)


            #loss = (loss_neg+loss_pos)/2
            total_loss = loss + lam * rule_loss
            losses.append(total_loss.item())


            solver.zero_grad()
            total_loss.backward()
            solver.step()
            #model.normalize_embeddings() #for transE you have to normalize
            end = time()

            if it % 50 == 0:
                #print('Iter-{}; loss: {:.4f}; rule loss {:.4f}; time per batch:{:.2f}s'.format(it, total_loss.item(), rule_loss ,end - start))
                try:
                    print('Iter-{}; loss: {:.4f}; labeled_loss: {:.4f}; unlabeled_loss: {:.4f};  time per batch:{:.2f}s'.format(it,
                                                                                                                            total_loss.item(),
                                                                                                                            labeled_loss.item(),
                                                                                                                            unlabeled_loss.item(),
                                                                                                                            end - start))
                except:
                    print(
                        'Iter-{}; loss: {:.4f}; labeled_loss: {:.4f}; unlabeled_loss: {:.4f};  time per batch:{:.2f}s'.format(
                            it,
                            total_loss.item(),
                            labeled_loss.item(),
                            unlabeled_loss,
                            end - start))

            it += 1

        if (epoch + 1) % eval_log_steps == 0:
            #evaluation with test dataloader...
            print('Evaluating on Test Dataset...')
            metrics = test_step(model,  val_pos, kg.all_triples, args)
            log_metrics('Test', epoch, metrics)



        #do an evaluation after each epoch
        '''if (epoch+1) % 98 == 0:
            print("Left-rank evaluation is started...")
            [rank, rank_filter] = left_rank(model, val_pos, Largest=True)
            print("Right-rank evaluation is started...")
            [rank_right, rank_filter_right] = right_rank(model, val_pos, Largest=True)
            rank = rank + rank_right
            rank_filter = rank_filter + rank_filter_right
            m_rank = mean_rank(rank)
            mean_rr = mrr(rank)
            print("Hits_N are being calculated...")
            hit_1 = hit_N(rank, 1)
            hit_3 = hit_N(rank, 3)
            hit_5 = hit_N(rank, 5)
            hit_10 = hit_N(rank, 10)
            m_rank_filter = mean_rank(rank_filter)
            mrr_filter = mrr(rank_filter)
            print("Hits_N for filtered rank are being calculated...")
            hit_1_filter = hit_N(rank_filter, 1)
            hit_3_filter = hit_N(rank_filter, 3)
            hit_5_filter = hit_N(rank_filter, 5)
            hit_10_filter = hit_N(rank_filter, 10)
            print("Epoch: ", epoch, " validation results.")
            print("m_rank: ", m_rank, " -- m_rank_filter: ",m_rank_filter)
            print("mean_rr: ", mean_rr, " -- mrr_filter: ",mrr_filter)
            print("hit_1: ", hit_1, " -- hit_1_filter: ",hit_1_filter)
            print("hit_3: ", hit_3, " -- hit_3_filter: ",hit_3_filter)
            print("hit_5: ", hit_5, " -- hit_5_filter: ",hit_5_filter)
            print("hit_10: ", hit_10, " -- hit_10_filter: ",hit_10_filter)
            '''

        #EVALUATION - SAVING THE MODEL
        #TODO: commented out for colab training...


        if epoch+1 == max_epoch:
            torch.save(model.state_dict(), os.path.join(path, 'params{:.0f}.pkl'.format(epoch)))
            f = open(os.path.join(path, 'result{:.0f}.txt'.format(epoch)), 'w')
            [rank, rank_filter] = left_rank(model,val_pos, Largest=True)
            [rank_right, rank_filter_right] = right_rank(model,val_pos, Largest=True)
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