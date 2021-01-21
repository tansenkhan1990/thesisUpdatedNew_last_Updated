import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from torch.autograd import Variable
#from model_utilities.model_initialization import *
from einops import repeat

class model(nn.Module):
    def __init__(self, name, kg, embedding_dim, batch_size, learning_rate, L, gamma, n_triples ,n_relation, n_entity,
                 n_times , n_locations,gpu=True, regul = False, negative_adversarial_sampling = False ,
                 temp = 0 ,train_with_groundings = False, fifthopole = False):
        super(model, self).__init__()
        self.name = name
        self.fifthopole = fifthopole
        self.gpu = gpu
        self.kg = kg
        # if name == 'transH_element':
        #     self.embedding_dim = embedding_dim/3
        # else:
        #     self.embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.regul = regul
        self.train_with_groundings = train_with_groundings
        self.L = L
        self.idx = 1
        self.n_relation = n_relation
        self.n_entity = n_entity
        self.n_times = n_times
        self.n_locations = n_locations
        # Nets
        #For each implimenation model you have to define their score function and their initialization
        self.init_embedding()
        # self.emb_E_real = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        # self.emb_E_img = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        # self.emb_R_phase = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)

        # store positive triples in a Hash table
        self.all_pos = {}
        if fifthopole == True:
            for i in kg.training_triples:
                self.all_pos[i[0], i[1], i[2], i[3], i[4]] = 1
            # for i in kg.validation_triples:
            #     self.all_pos[i[0], i[1], i[2]] = 1
            for i in kg.test_triples:
                self.all_pos[i[0], i[1], i[2], i[3], i[4]] = 1
        else:
            for i in kg.training_triples:
                self.all_pos[i[0], i[1], i[2]] = 1
            # for i in kg.validation_triples:
            #     self.all_pos[i[0], i[1], i[2]] = 1
            for i in kg.test_triples:
                self.all_pos[i[0], i[1], i[2]] = 1
        # # Initialization
        # r = 6 / np.sqrt(self.embedding_dim)
        # self.emb_E_real.weight.data.uniform_(-r, r)
        # self.emb_E_img.weight.data.uniform_(-r, r)
        # self.emb_R_phase.weight.data.uniform_(-r, r)
        # self.error.weight.data.uniform_(0, r)
        self.xi = nn.Parameter(torch.zeros(n_triples, 1))
        nn.init.uniform_(
            tensor=self.xi,
            a=-0.1,
            b=0.1
        )
        self.alpha = nn.Parameter(
            torch.Tensor([10.100000100]),
            requires_grad=False
        )
        self.xi_neg = nn.Parameter(torch.zeros(n_triples, 1))
        nn.init.uniform_(
            tensor=self.xi_neg,
            a=-0.1,
            b=0.1
        )
        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.temp = temp
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        if self.gpu:
            self.cuda()

    def calculate_score(self, head, tail, relation, times = None, locations = None):
        #we also need to adapt
        #print(self.emb_E.weight.data.uniform_(1, 1))
        if self.name == 'distmult_quad' or self.name == 'transE_quad' or self.name == 'transH_element_quad' \
                or self.name == 'complEx_quad' or self.name == 'transR_quad':
            if self.gpu:
                h_i = Variable(torch.from_numpy(head).cuda())
                t_i = Variable(torch.from_numpy(tail).cuda())
                r_i = Variable(torch.from_numpy(relation).cuda())
                tm_i = Variable(torch.from_numpy(times).cuda())
                loc_i = Variable(torch.from_numpy(locations).cuda())

            else:
                h_i = Variable(torch.from_numpy(head))
                t_i = Variable(torch.from_numpy(tail))
                r_i = Variable(torch.from_numpy(relation))
                tm_i = Variable(torch.from_numpy(times))
                loc_i = Variable(torch.from_numpy(locations))
        else:
            if self.gpu:
                h_i = Variable(torch.from_numpy(head).cuda())
                t_i = Variable(torch.from_numpy(tail).cuda())
                r_i = Variable(torch.from_numpy(relation).cuda())

            else:
                h_i = Variable(torch.from_numpy(head))
                t_i = Variable(torch.from_numpy(tail))
                r_i = Variable(torch.from_numpy(relation))
        #print(h_i)
        #exit()
        if self.name == 'RotatE':

            pi = 3.14159265358979323846
            h_real = self.emb_E_real(h_i).view(-1, self.embedding_dim)
            t_real = self.emb_E_real(t_i).view(-1, self.embedding_dim)
            r_real = torch.cos(self.emb_R_phase(r_i).view(-1, self.embedding_dim)/(6 / np.sqrt(self.embedding_dim)/pi))

            h_img = self.emb_E_img(h_i).view(-1, self.embedding_dim)
            t_img = self.emb_E_img(t_i).view(-1, self.embedding_dim)
            r_img = torch.sin(self.emb_R_phase(r_i).view(-1, self.embedding_dim)/(6 / np.sqrt(self.embedding_dim)/pi))

            if self.L == 'L1':
                out_real = torch.sum(torch.abs(h_real*r_real-h_img*r_img-t_real),1)
                out_img  = torch.sum(torch.abs(h_real*r_img+h_img*r_real-t_img),1)
                out = out_real + out_img
            else:
                out_real = torch.sum((h_real*r_real-h_img*r_img-t_real) ** 2, 1)
                out_img = torch.sum((h_real*r_img+h_img*r_real-t_img) ** 2, 1)
                out = torch.sqrt(out_img + out_real)
            #print(h_real.size())
            #exit()
            return out

        elif self.name == 'transComplEx':

            h_real = self.emb_E_real(h_i).view(-1, self.embedding_dim)
            t_real = self.emb_E_real(t_i).view(-1, self.embedding_dim)
            r_real = self.emb_R_real(r_i).view(-1, self.embedding_dim)

            h_img = self.emb_E_img(h_i).view(-1, self.embedding_dim)
            t_img = self.emb_E_img(t_i).view(-1, self.embedding_dim)
            r_img = self.emb_R_img(r_i).view(-1, self.embedding_dim)

            if self.L == 'L1':
                out_real = torch.sum(torch.abs(h_real + r_real - t_real), 1)
                out_img = torch.sum(torch.abs(h_img + r_img + t_img), 1)
                out = out_real + out_img
            else:
                out_real = torch.sum((h_real + r_real - t_real) ** 2, 1)
                out_img = torch.sum((h_img + r_img + t_real) ** 2, 1)
                out = torch.sqrt(out_img + out_real)
            return out

        elif self.name == 'distmult':

            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)
            # score h*t*r*t*l
            #print(h.size())
            #print(t.size())
            #print(r.size())
            #exit()
            out = torch.sum(h *t* r, dim=1)
            #exit()
            if self.regul==True:
                #time regularization and location regularization
                regular = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
                return out, regular

            return out

        elif self.name == 'distmult_quad':

            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)
            tm = self.emb_tm(tm_i).view(-1, self.embedding_dim)
            loc = self.emb_loc(loc_i).view(-1, self.embedding_dim)
            #print(tm)
            # score h*t*r*t*l
            out = torch.sum(h * t * r * tm * loc, dim=1)

            if self.regul==True:
                #time regularization and location regularization
                regular = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2) + torch.mean(tm ** 2) \
                          + torch.mean(loc ** 2))/6
                return out, regular

            return out

        elif self.name == 'complEx':
            '''
            implemented based on 
            https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/ComplEx.py
            '''
            # head = self.emb_E(h_i).view(-1, self.embedding_dim)
            # tail = self.emb_E(t_i).view(-1, self.embedding_dim)
            # relation = self.emb_R(r_i).view(-1, self.embedding_dim)
            # # print(head.size())
            # # print(tail.size())
            # # print(relation.size())
            # re_head, im_head = torch.chunk(head, 2, dim=1)
            # re_relation, im_relation = torch.chunk(relation, 2, dim=1)
            # re_tail, im_tail = torch.chunk(tail, 2, dim=1)
            # # print(re_head.size())
            # re_score = re_head * re_relation - im_head * im_relation
            # im_score = re_head * im_relation + im_head * re_relation
            #
            # score = re_score * re_tail + im_score * im_tail
            #
            # #score = re_score @ re_tail + im_score @ im_tail
            #
            # #score = score @ to_score_real
            #
            #
            # out = torch.sum(score, dim=1)
            #
            # if self.regul == True:
            #    regular = torch.mean(re_head ** 2) + torch.mean(im_head ** 2) + torch.mean(re_tail ** 2) + torch.mean(
            #         im_tail ** 2) + torch.mean(re_relation ** 2) + torch.mean(im_relation ** 2)
            #    return out, regular
            #
            # return out
            head_real = self.emb_E_real(h_i).view(-1, self.embedding_dim)
            head_im = self.emb_E_im(h_i).view(-1, self.embedding_dim)

            tail_real = self.emb_E_real(t_i).view(-1, self.embedding_dim)
            tail_im = self.emb_E_im(t_i).view(-1, self.embedding_dim)

            relation_real = self.emb_R_real(r_i).view(-1, self.embedding_dim)
            relation_im = self.emb_R_im(r_i).view(-1, self.embedding_dim)

            if self.regul == True:
                out = torch.sum(
                    head_real * tail_real * relation_real
                    + head_im * tail_im * relation_real
                    + head_real * tail_im * relation_im
                    - head_im * tail_real * relation_im,
                    dim=1)
                regul = (torch.mean(head_real ** 2) \
                        + torch.mean(head_im ** 2) \
                        + torch.mean(tail_real ** 2) \
                        + torch.mean(tail_im ** 2) \
                        + torch.mean(relation_real ** 2) \
                        + torch.mean(relation_im ** 2))/6
                return out, regul

            out = torch.sum(
                head_real * tail_real * relation_real
                + head_im * tail_im * relation_real
                + head_real * tail_im * relation_im
                - head_im * tail_real * relation_im,
                dim=1)

            return out

        elif self.name == 'complEx_quad':
            '''
            implemented based on 
            https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/ComplEx.py
            '''
            # head = self.emb_E(h_i).view(-1, self.embedding_dim)
            # tail = self.emb_E(t_i).view(-1, self.embedding_dim)
            # relation = self.emb_R(r_i).view(-1, self.embedding_dim)
            # # print(head.size())
            # # print(tail.size())
            # # print(relation.size())
            # re_head, im_head = torch.chunk(head, 2, dim=1)
            # re_relation, im_relation = torch.chunk(relation, 2, dim=1)
            # re_tail, im_tail = torch.chunk(tail, 2, dim=1)
            # # print(re_head.size())
            # re_score = re_head * re_relation - im_head * im_relation
            # im_score = re_head * im_relation + im_head * re_relation
            #
            # score = re_score * re_tail + im_score * im_tail
            #
            # #score = re_score @ re_tail + im_score @ im_tail
            #
            # #score = score @ to_score_real
            #
            #
            # out = torch.sum(score, dim=1)
            #
            # if self.regul == True:
            #    regular = torch.mean(re_head ** 2) + torch.mean(im_head ** 2) + torch.mean(re_tail ** 2) + torch.mean(
            #         im_tail ** 2) + torch.mean(re_relation ** 2) + torch.mean(im_relation ** 2)
            #    return out, regular
            #
            # return out

            head_real = self.emb_E_real(h_i).view(-1, self.embedding_dim)
            head_im = self.emb_E_im(h_i).view(-1, self.embedding_dim)

            tail_real = self.emb_E_real(t_i).view(-1, self.embedding_dim)
            tail_im = self.emb_E_im(t_i).view(-1, self.embedding_dim)

            relation_real = self.emb_R_real(r_i).view(-1, self.embedding_dim)
            relation_im = self.emb_R_im(r_i).view(-1, self.embedding_dim)

            tim_real = self.emb_tim_real(tm_i).view(-1, self.embedding_dim)
            tim_im = self.emb_tim_im(tm_i).view(-1, self.embedding_dim)

            loc_real = self.emb_loc_real(loc_i).view(-1, self.embedding_dim)
            loc_im = self.emb_loc_im(loc_i).view(-1, self.embedding_dim)

            head_real_z = head_real * tim_real - head_im * tim_im
            head_im_z = head_real * tim_im + head_im * tim_real

            tail_real_z = tail_real * loc_real - tail_im * loc_im
            tail_im_z = tail_real * loc_im + tail_im * loc_real

            if self.regul == True:
                out = torch.sum(
                    head_real_z * tail_real_z * relation_real
                    + head_im_z * tail_im_z * relation_real
                    + head_real_z * tail_im_z * relation_im
                    - head_im_z * tail_real_z * relation_im,
                    dim=1)
                # over original not for z
                regul = (torch.mean(head_real ** 2) \
                         + torch.mean(head_im ** 2) \
                         + torch.mean(tail_real ** 2) \
                         + torch.mean(tail_im ** 2) \
                         + torch.mean(relation_real ** 2) \
                         + torch.mean(relation_im ** 2) \
                         + torch.mean(loc_real ** 2) \
                         + torch.mean(loc_im ** 2) \
                         + torch.mean(tim_real ** 2) \
                         + torch.mean(tim_im ** 2)) / 10
                return out, regul

            out = torch.sum(
                head_real_z * tail_real_z * relation_real
                + head_im_z * tail_im_z * relation_real
                + head_real_z * tail_im_z * relation_im
                - head_im_z * tail_real_z * relation_im,
                dim=1)

            return out


        elif self.name == 'transE':

            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)

            #h * wr +r -t
            # print(h.size())
            # exit()
            if self.L == 'L1':
                out = torch.sum(torch.abs(h + r - t), 1)

            else:
                out = torch.sqrt(torch.sum((h + r - t) ** 2, 1))
            return out

        elif self.name == 'transH':

            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)
            wr = self.emb_Wr(r_i).view(-1, self.embedding_dim)
            #print(h.size())
            #exit()
            #print(h.size())
            #print(t.size())
            #print(r.size())
            #print(wr.size())
            #x = torch.sum(h * wr * wr, dim=1)
            #print(x.size())
            #exit()

            h_proj = h- (wr*h*wr)
            t_proj = t -(wr*t*wr)
            #dr = (h- (wr*h*wr)) - (t -(wr*t*wr))
            #print(h_proj.size())
            #print(t_proj.size())
            #exit()
            #h * wr +r -t
            #|h_P + r - t_P
            if self.L == 'L1':
                #out = torch.sum(torch.abs(h + r - t), 1)
                out = torch.sum(torch.abs(h_proj + r - t_proj) ,dim=1)
            else:
                out = torch.norm((h_proj + r - t_proj), p=2, dim=1)
                print(out.size())
                #out = torch.norm((h_proj + dr - t_proj), p=2, dim=1)
                #exit()
                #out = torch.sum(out)
                #print(out)
                #exit()
                #print(out)
                #out = torch.sqrt(out)
                #print(out)
            # print(out.size())
            # exit()
            return out
        elif self.name == 'transR':

            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)
            pr = self.proj(r_i).view(-1, self.embedding_dim, self.embedding_dim)

            #h_pr = torch.matmul(h.view(-1,1,self.embedding_dim),pr)
            #t_pr = torch.matmul(t.view(-1,1, self.embedding_dim),pr)

            h_pr = torch.matmul(h,pr)
            t_pr = torch.matmul(t,pr)

            #exit()
            #exit()
            if self.L == 'L1':
                #out = torch.sum(torch.abs(h + r - t), 1)
                out = torch.sum(torch.abs(h_pr + r - t_pr) ,dim=1)
            else:
                out = torch.norm((h_pr + r - t_pr), p=2, dim=1)

            #print(out)
            #exit()
            return out

        elif self.name == 'transR_quad':

            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)
            tm = self.emb_tm(tm_i).view(-1, self.embedding_dim)
            loc = self.emb_loc(loc_i).view(-1, self.embedding_dim)


            #loc_time = torch.tensor([self.identity * (i*j) for i,j in zip(tm, loc)])
            #repeated_identity = repeat(self.identity, 'i j -> (tile i) j', tile=self.batch_size)
            #print(repeated_identity.size())
            #print(tm.t().size())
            #print(loc.size())
            #exit()
            loc_tim = torch.matmul(loc.t() , tm)

            #print(loc_tim.size())
            #print(self.identity.size())
            #exit()
            #print(self.identity.size())
            #print(loc_tim.size())
            #exit()
            loc_tim = self.identity + loc_tim # [dXd metrix + dXd vector]
            #print(loc_time.size())
            #print(loc_time.size())
            #print(repeated_identity.size())
            #exit()
            h_t = torch.matmul(h, loc_tim)
            t_t = torch.matmul(t, loc_tim)
            #print(h_t.size())
            #print(t_t.size())
            #exit()
            if self.L == 'L1':
                out = torch.sum(torch.abs(h_t + r - t_t), 1)
                #e={1,2,3.....d}

            else:
                out = torch.sqrt(torch.sum((h_t + r - t_t) ** 2, 1))

            return out

        elif self.name == 'transH_element':
            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)
            wr = self.emb_Wr(r_i).view(-1, self.embedding_dim)
            # print("Main H is: ")
            # print(h.shape)
            # exit()
            new_dim = self.embedding_dim//3
            h = h.view(h.size()[0], new_dim,3)
            r = r.view(r.size()[0], new_dim,3)
            t = t.view(t.size()[0], new_dim,3)
            wr = wr.view(wr.size()[0], new_dim,3)
            # h = h.view(h.size()[0], 200,3)
            # t = t.view(t.size()[0], 200 ,3)
            # r = r.view(r.size()[0], 200 ,3)
            # wr = wr.view(wr.size()[0], 200 , 3)

            # print(h)
            # print(r.size())
            # print(t.size())
            # print(wr.size())
            # exit()

            hwr = torch.einsum('ijk,ijz->ij', h, wr)
            twr = torch.einsum('ijk,ijz->ij', t, wr)
            hrh = torch.einsum('ij,ijz->ijz', hwr,h)
            trt = torch.einsum('ij,ijz->ijz', twr, t)
            h_proj = h - hrh
            t_proj = t - trt
            # print(h_proj.size())
            # exit()

            if self.L == 'L1':
                # out = torch.sum(torch.abs(h + r - t), 1)
                out = torch.sum(torch.abs(h_proj + r - t_proj), dim=2)
                out = torch.sum(out,dim = 1)
            else:
                out = torch.norm((h_proj + r - t_proj), p=2, dim=2)
                out = torch.sum(out, dim = 1)

            # print(out.size())
            # exit()
            return out

            # print(t_proj.size())
            # exit()


            #  print(h)
            # print(h.size())
            # print("t")
            # print(t.size())
            # print("wr")
            # print(wr.size())
            # exit()
            #print(h.size())
            # h1, h2, h3 = torch.chunk(h, 3, dim=1)
            # t1, t2, t3 = torch.chunk(t, 3, dim=1)
            # r1, r2, r3 = torch.chunk(r, 3, dim=1)
            # wr1, wr2, wr3 = torch.chunk(wr, 3, dim=1)
            # print("after chunk")
            # print(h1)
            # print(h2)
            # print(h3)
            # print(h1.size())
            # exit()
            out_list = []
            out_ = 0
            for i in range(self.embedding_dim//3):
                # print(h1.size())
                # print(h1[i].size())
                # print(h2[0][i])
                # print(h3[0][i])
                # exit()
                h_ = torch.tensor([[h1[0][i], h2[0][i], h3[0][i]]], requires_grad=True)
                #print("inside loop")
                #print(h_.size())
                #print(h_)
                #exit()
                r_ = torch.tensor([[r1[0][i], r2[0][i], r3[0][i]]], requires_grad=True)
                t_ = torch.tensor([[t1[0][i], t2[0][i], t3[0][i]]], requires_grad=True)
                wr_ = torch.tensor([[wr1[0][i], wr2[0][i], wr3[0][i]]], requires_grad=True)
                h_proj = h_ - (wr_ * h_ * wr_)
                t_proj = t_ - (wr_ * t_ * wr_)
                if self.L == 'L1':
                    # out = torch.sum(torch.abs(h + r - t), 1)
                    out_ =  torch.sum(torch.abs(h_proj + r_ - t_proj), dim=1)
                else:
                    out_ =  torch.norm((h_proj + r_ - t_proj), p=2, dim=1)
                    #print(out_.size())
                    #exit()
                #sum+=out
                out_list.append(out_)
            #print(h1)
            #print(h2)
            #print(h3)

            #exit()
            #print('#################')
            #print(h2)
            #print('#################')
            #print(h3)
            #print('#################')
            # print(h.size())
            # print(h1.size())
            # print(h2.size())
            # print(h3.size())
            # print("tail dimension")
            # print(t.size())
            # print(t1.size())
            # print(t2.size())
            # print(t3.size())        # print("relation dimension")
            # print(r.size())
            # print(r1.size())
            # print(r2.size())
            # print(r3.size())
            # exit()
            #/*-789+4566123          #h1_d = [i for i in zip(h1, h2, h3)]
            #print(h1.size())
            #print(h1[1].size())
            #exit()
            # h1 = h1.cpu().detach().cpu().numpy()
            # h2 = h2.cpu().detach().cpu().numpy()
            # h3 = h3.cpu().detach().cpu().numpy()
            #
            # r1 = r1.cpu().detach().cpu().numpy()
            # r2 = r2.cpu().detach().cpu().numpy()
            # r3 = r3.cpu().detach().cpu().numpy()
            #
            # t1 = t1.cpu().detach().cpu().numpy()
            # t2 = t2.cpu().detach().cpu().numpy()
            # t3 = t3.cpu().detach().cpu().numpy()
            #
            # wr1 = wr1.cpu().detach().cpu().numpy()
            # wr2 = wr2.cpu().detach().cpu().numpy()
            # wr3 = wr3.cpu().detach().cpu().numpy()
            #
            # h = [[i,j,k] for i, j, k in zip(h1, h2, h3)]
            # t = [[i, j, k] for i, j, k in zip(t1, t2, t3)]
            # r = [[i, j, k] for i, j, k in zip(r1, r2, r3)]
            # wr = [[i, j, k] for i, j, k in zip(wr1, wr2, wr3)]
            # # print(h)
            # # print('#################')
            # # print(np.array(h).shape)
            # # print('#################')
            # # exit()
            # h = torch.from_numpy(np.array(h))
            # t = torch.from_numpy(np.array(t))
            # r = torch.from_numpy(np.array(r))
            # wr = torch.from_numpy(np.array(wr))
            #
            # h.requires_grad = True
            # t.requires_grad = True
            # r.requires_grad = True
            # wr.requires_grad = True
            #
            # h_proj = h- (wr*h*wr)
            # t_proj = t-(wr*t*wr)
            # # h1_proj = h1- (wr1*h1*wr1)
            # # h2_proj = h2 - (wr2 * h2 * wr2)
            # # h3_proj = h3 - (wr3 * h3 * wr3)
            # # t1_proj = t1 - (wr1 * t1 * wr1)
            # # t2_proj = t2 - (wr2 * t2 * wr2)
            # # t3_proj = t3 - (wr3 * t3 * wr3)
            # # t_proj = t -(wr*t*wr)
            # if self.L == 'L1':
            #     # out = torch.sum(torch.abs((h- (wr*h*wr)) + r - (t-(wr*t*wr))), 1)
            #     #out = out = torch.sum(torch.abs(h_proj + r - t_proj), dim=1)
            #     out = torch.sum(torch.abs(h_proj + r - t_proj), dim=1)
            # else:
            #     out = torch.norm((h_proj + r - t_proj), p=2, dim=1)
            #     # out = torch.norm((h_proj + dr - t_proj), p=2, dim=1)
            #     # exit()
            #     # out = torch.sum(out)
            #     # print(out)
            #     # exit()
            #     # print(out)
            #     # out = torch.sqrt(out)
            #     # print(out)
            out_list = torch.tensor([out_list], requires_grad=True)
            # print(out_list.size())
            # print(out_list)
            # exit()
            # return out_
        elif self.name == 'transH_element_quad':

            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)
            #wr = self.emb_Wr(r_i).view(-1, self.embedding_dim)
            tm = self.emb_tm(tm_i).view(-1, self.embedding_dim)
            loc = self.emb_loc(loc_i).view(-1, self.embedding_dim)
            # print("Main H is: ")
            # print(h.shape)
            # exit()
            new_dim = self.embedding_dim//3
            h = h.view(h.size()[0], new_dim,3)
            r = r.view(r.size()[0], new_dim,3)
            t = t.view(t.size()[0], new_dim,3)
            #wr = wr.view(wr.size()[0], new_dim,3)
            tm = tm.view(tm.size()[0], new_dim, 3)
            loc = loc.view(loc.size()[0], new_dim, 3)
            #print(h.size())
            wloctime = torch.cross(tm, loc,dim = 2)
            print(tm.size())
            print(loc.size())
            exit()
            # print(wloctime.size())
            #exit()
            # h = h.view(h.size()[0], 200,3)
            # t = t.view(t.size()[0], 200 ,3)
            # r = r.view(r.size()[0], 200 ,3)
            # wr = wr.view(wr.size()[0], 200 , 3)

            # print(h)
            # print(r.size())
            # print(t.size())
            # print(wr.size())
            # exit()

            hwr = torch.einsum('ijk,ijz->ij', h, wloctime)
            hwr2 = torch.einsum('ij,ijk ->ijk', hwr, h)
            twr = torch.einsum('ijk,ijz->ij', t, wloctime)
            twr2 = torch.einsum('ij,ijk ->ijk', twr, t)
            # print(h.size())
            # print(hwr2.size())
            # print(twr2.size())
            # exit()
            #hrh = torch.einsum('ij,ijz->ijz', hwr,h)
            #trt = torch.einsum('ij,ijz->ijz', twr, t)
            h_proj = h - hwr2
            t_proj = t - twr2

            if self.L == 'L1':
                # out = torch.sum(torch.abs(h + r - t), 1)
                out = torch.sum(torch.abs(h_proj + r - t_proj), dim=2)
                out = torch.sum(out,dim = 1)
            else:
                out = torch.norm((h_proj + r - t_proj), p=2, dim=2)
                out = torch.sum(out, dim = 1)

            return out
            # print(out.size())
            # exit()
            # print(h_proj.size())
            # exit()

        elif self.name == 'transE_quad':

            # h = self.emb_E(h_i).view(-1, self.embedding_dim)
            # t = self.emb_E(t_i).view(-1, self.embedding_dim)
            # r = self.emb_R(r_i).view(-1, self.embedding_dim)

            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)
            tm = self.emb_tm(tm_i).view(-1, self.embedding_dim)
            loc = self.emb_loc(loc_i).view(-1, self.embedding_dim)
            if self.L == 'L1':
                out = torch.sum(torch.abs(h + r + tm + loc - t), 1)
                #e={1,2,3.....d}

            else:
                out = torch.sqrt(torch.sum((h + r + tm + loc - t) ** 2, 1))

            return out

        elif self.name == 'LogicENN':
            h_e = self.emb_E(h_i).view(-1, self.embedding_dim)
            t_e = self.emb_E(t_i).view(-1, self.embedding_dim)
            r_e = self.emb_R(r_i).view(-1, self.embedding_dim, 1)
            h_e = torch.cat((h_e, t_e), 1)
            # out=F.relu(self.fc1(h_e))
            out = self.relu(self.fc1(h_e))
            out = self.relu(self.fc2(out))
            out = self.relu(self.fc3(out))
            out = torch.bmm(torch.transpose(r_e, 1, 2), out.view(-1, self.embedding_dim, 1))

            out = out.view(-1, 1)
            return out


    def init_embedding(self):

        if self.name == 'RotatE':
            self.emb_E_real = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_E_img = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_R_phase = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            self.error = torch.nn.Embedding(self.kg.n_training_triple, 1)
            r = 6 / np.sqrt(self.embedding_dim)
            self.emb_E_real.weight.data.uniform_(-r, r)
            self.emb_E_img.weight.data.uniform_(-r, r)
            self.emb_R_phase.weight.data.uniform_(-r, r)
            self.error.weight.data.uniform_(0, r)

        elif self.name == 'transComplEx':
            self.emb_E_real = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_E_img = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_R_real = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            self.emb_R_img = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            self.error = torch.nn.Embedding(self.kg.n_training_triple, 1)
            r = 6 / np.sqrt(self.embedding_dim)
            self.emb_E_real.weight.data.uniform_(-r, r)
            self.emb_E_img.weight.data.uniform_(-r, r)
            self.emb_R_real.weight.data.uniform_(-r, r)
            self.emb_R_img.weight.data.uniform_(-r, r)
            self.error.weight.data.uniform_(0, r)

        elif self.name == 'distmult':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            xavier_normal_(self.emb_E.weight.data)
            xavier_normal_(self.emb_R.weight.data)

        elif self.name == 'distmult_quad':


            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            self.emb_tm = torch.nn.Embedding(self.kg.n_times, self.embedding_dim, padding_idx=0)
            self.emb_loc = torch.nn.Embedding(self.kg.n_locations, self.embedding_dim, padding_idx=0)
            xavier_normal_(self.emb_E.weight.data)
            xavier_normal_(self.emb_R.weight.data)
            xavier_normal_(self.emb_tm.weight.data)
            xavier_normal_(self.emb_loc.weight.data)

        elif self.name == 'complEx':
            # self.emb_E = nn.Embedding(self.kg.n_entity, self.embedding_dim)
            # self.emb_R = nn.Embedding(self.kg.n_relation, self.embedding_dim)
            # self.embeddings = [self.emb_E, self.emb_R]
            # xavier_normal_(self.emb_E.weight.data)
            # xavier_normal_(self.emb_R.weight.data)
            #self.emb_E_real = nn.Embedding(self.kg.n_entity,self.embedding_dim, 3)
            self.emb_E_real = nn.Embedding(self.kg.n_entity, self.embedding_dim)
            self.emb_R_real = nn.Embedding(self.kg.n_entity, self.embedding_dim)
            self.emb_E_im = nn.Embedding(self.kg.n_entity, self.embedding_dim)
            self.emb_R_im = nn.Embedding(self.kg.n_entity, self.embedding_dim)

            nn.init.xavier_uniform_(self.emb_E_real.weight.data)
            nn.init.xavier_uniform_(self.emb_R_real.weight.data)
            nn.init.xavier_uniform_(self.emb_E_im.weight.data)
            nn.init.xavier_uniform_(self.emb_R_im.weight.data)

        elif self.name == 'complEx_quad':
            self.emb_E_real = nn.Embedding(self.kg.n_entity, self.embedding_dim)
            self.emb_R_real = nn.Embedding(self.kg.n_entity, self.embedding_dim)
            self.emb_E_im = nn.Embedding(self.kg.n_entity, self.embedding_dim)
            self.emb_R_im = nn.Embedding(self.kg.n_entity, self.embedding_dim)

            self.emb_tim_real = nn.Embedding(self.kg.n_times, self.embedding_dim)
            self.emb_tim_im = nn.Embedding(self.kg.n_times, self.embedding_dim)

            self.emb_loc_real = nn.Embedding(self.kg.n_locations, self.embedding_dim)
            self.emb_loc_im = nn.Embedding(self.kg.n_locations, self.embedding_dim)

            nn.init.xavier_uniform_(self.emb_E_real.weight.data)
            nn.init.xavier_uniform_(self.emb_R_real.weight.data)
            nn.init.xavier_uniform_(self.emb_E_im.weight.data)
            nn.init.xavier_uniform_(self.emb_R_im.weight.data)

            nn.init.xavier_uniform_(self.emb_tim_real.weight.data)
            nn.init.xavier_uniform_(self.emb_tim_im.weight.data)

            nn.init.xavier_uniform_(self.emb_loc_real.weight.data)
            nn.init.xavier_uniform_(self.emb_loc_im.weight.data)

        elif self.name == 'transE':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim , padding_idx=0)
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)

            # Initialization
            r = 6 / np.sqrt(self.embedding_dim)
            #r = 1
            self.emb_E.weight.data.uniform_(-r, r)
            self.emb_R.weight.data.uniform_(-r, r)
            #print(self.emb_E.weight.data)
            #print(self.emb_R.weight.data)
            #exit()
            #entity_embeddings = torch.Tensor.cpu(self.emb_E).detach().numpy()
            #relation_embeddings = torch.Tensor.cpu(self.emb_R).detach().numpy()
            #print(entity_embeddings)
            #print(relation_embeddings)
            self.normalize_embeddings()

        elif self.name == 'transR':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim , padding_idx=0)
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            self.proj = torch.nn.Embedding(self.n_relation, self.embedding_dim * self.embedding_dim)
            nn.init.xavier_uniform_(self.emb_E.weight.data)
            nn.init.xavier_uniform_(self.emb_R.weight.data)
            nn.init.xavier_uniform_(self.proj.weight.data)
            self.normalize_embeddings()

        elif self.name == 'transR_quad':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            self.emb_tm = torch.nn.Embedding(self.kg.n_times, self.embedding_dim, padding_idx=0)
            self.emb_loc = torch.nn.Embedding(self.kg.n_locations, self.embedding_dim, padding_idx=0)
            self.identity = torch.eye(self.embedding_dim)
            xavier_normal_(self.emb_E.weight.data)
            xavier_normal_(self.emb_R.weight.data)
            xavier_normal_(self.emb_tm.weight.data)
            xavier_normal_(self.emb_loc.weight.data)


        elif self.name == 'transH':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            self.emb_Wr = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            #self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            # Initialization
            r = 6 / np.sqrt(self.embedding_dim)
            self.emb_E.weight.data.uniform_(-r, r)
            self.emb_R.weight.data.uniform_(-r, r)
            self.emb_Wr.weight.data.uniform_(-r, r)
            #print(self.emb_E.weight.data)
            #print(self.emb_R.weight.data)
            #exit()
            self.normalize_embeddings_tH()
        elif self.name == 'transH_element':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            self.emb_Wr = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            # self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            # Initialization
            r = 6 / np.sqrt(self.embedding_dim)
            self.emb_E.weight.data.uniform_(-r, r)
            self.emb_R.weight.data.uniform_(-r, r)
            self.emb_Wr.weight.data.uniform_(-r, r)

        elif self.name == 'transH_element_quad':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            self.emb_Wr = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            self.emb_tm = torch.nn.Embedding(self.kg.n_times, self.embedding_dim, padding_idx=0)
            self.emb_loc = torch.nn.Embedding(self.kg.n_locations, self.embedding_dim, padding_idx=0)
            # self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            # Initialization
            r = 6 / np.sqrt(self.embedding_dim)
            self.emb_E.weight.data.uniform_(-r, r)
            self.emb_R.weight.data.uniform_(-r, r)
            self.emb_Wr.weight.data.uniform_(-r, r)
            self.emb_tm.weight.data.uniform_(-r, r)
            self.emb_loc.weight.data.uniform_(-r, r)

        elif self.name == 'transE_quad':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            #self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, 3)
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            #self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim,3)
            self.emb_tm = torch.nn.Embedding(self.kg.n_times, self.embedding_dim, padding_idx=0)
            #self.emb_tm = torch.nn.Embedding(self.kg.n_times, self.embedding_dim,3)
            self.emb_loc = torch.nn.Embedding(self.kg.n_locations, self.embedding_dim, padding_idx=0)
            #self.emb_loc = torch.nn.Embedding(self.kg.n_locations, self.embedding_dim,3)
            #print(self.emb_E)
            #exit()
            # Initialization
            r = 6 / np.sqrt(self.embedding_dim)
            #r = 1
            self.emb_E.weight.data.uniform_(-r, r)
            self.emb_R.weight.data.uniform_(-r, r)
            self.emb_tm.weight.data.uniform_(-r, r)
            self.emb_loc.weight.data.uniform_(-r, r)
            #print(self.emb_E.weight.data)
            #print(self.emb_R.weight.data)
            #exit()
            #entity_embeddings = torch.Tensor.cpu(self.emb_E).detach().numpy()
            #relation_embeddings = torch.Tensor.cpu(self.emb_R).detach().numpy()
            #print(entity_embeddings)
            #print(relation_embeddings)
            #exit()
            self.normalize_embeddings_t()

        elif self.name == 'LogicENN':
            # self.all_pos = {}
            # for i in self.kg.training_triples:
            #     self.all_pos[i[0], i[1], i[2]] = 1
            # for i in self.kg.validation_triples:
            #     self.all_pos[i[0], i[1], i[2]] = 1
            # for i in self.kg.test_triples:
            #     self.all_pos[i[0], i[1], i[2]] = 1
            # # for i in self.groundings:
            # #     try:
            # #         self.all_pos[i[1], i[0], i[3]] = 1
            # #     except:
            # #         continue

            self.emb_E = nn.Embedding(self.kg.n_entity, self.embedding_dim)
            self.emb_R = nn.Embedding(self.kg.n_relation, self.embedding_dim)
            self.embeddings = [self.emb_E, self.emb_R]

            bound = 6 / np.sqrt(self.embedding_dim)
            for e in self.embeddings:
                e.weight.data.uniform_(-bound, bound)

            self.normalize_embeddings()
            self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)

            self.fc1 = nn.Linear(self.embedding_dim * 2, self.embedding_dim * 5)
            self.fc2 = nn.Linear(self.embedding_dim * 5, self.embedding_dim * 10)
            self.fc3 = nn.Linear(self.embedding_dim * 10, self.embedding_dim * 1)
                # setup multipliers


    def initialize(self, tensor, in_features, out_features):
        if 'Quat' not in self.name:
            nn.init.uniform_(
                tensor=tensor,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        else:  # use quaternion initialization
            fan_in = in_features
            fan_out = out_features

            s = 1. / np.sqrt(2 * in_features)

            rng = torch.random.manual_seed(42)

            # Generating randoms and purely imaginary quaternions :
            kernel_shape = (in_features, out_features)
            nweigths = in_features * out_features

            wi = torch.FloatTensor(nweigths).uniform_()
            wj = torch.FloatTensor(nweigths).uniform_()
            wk = torch.FloatTensor(nweigths).uniform_()

            # Purely imaginary quaternions unitary
            norms = torch.sqrt(wi ** 2 + wj ** 2 + wk ** 2) + 0.0001
            wi /= norms
            wj /= norms
            wk /= norms

            wi = wi.reshape(kernel_shape)
            wj = wj.reshape(kernel_shape)
            wk = wk.reshape(kernel_shape)

            modulus = torch.zeros(kernel_shape).uniform_(-s, s)
            phase = torch.zeros(kernel_shape).uniform_(-np.pi, np.pi)

            weight_r = modulus * torch.cos(phase)
            weight_i = modulus * wi * torch.sin(phase)
            weight_j = modulus * wj * torch.sin(phase)
            weight_k = modulus * wk * torch.sin(phase)

            tensor.data = torch.cat((weight_r, weight_i, weight_j, weight_k), dim=1)

    def forward(self, X):
        #here we need to adapt
        head, tail, relation = X[:, 0].astype(np.int64), X[:, 1].astype(np.int64), X[:, 2].astype(np.int64)
        #print(head)
        if self.regul == True:
            out, regul = self.calculate_score(head, tail, relation)
            return out,regul
        else:
            out = self.calculate_score(head, tail, relation)
        return out

    def forward_t(self, X):
        #here we need to adapt
        head, tail, relation, times, locations = X[:, 0].astype(np.int64), X[:, 1].astype(np.int64), X[:, 2].astype(np.int64), \
                                                 X[:, 3].astype(np.int64), X[:, 4].astype(np.int64)

        #exit()
        if self.regul == True:
            out, regul = self.calculate_score(head, tail, relation, times, locations)
            return out,regul
        else:
            out = self.calculate_score(head, tail, relation, times, locations)
        return out
    def normalize_embeddings_tH(self):
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        #self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_Wr.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def normalize_embeddings(self):
        if self.fifthopole == True:
            self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
            self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)
            self.emb_tm.weight.data.renorm_(p=2, dim=0, maxnorm=1)
            self.emb_loc.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        else:
            self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
            self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def normalize_embeddings_t(self):
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def normalize_embeddings_tasnim_apu(self):
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        #self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def get_embeddings(self):
        entity_embedding = self.embeddings[0].weight.data.numpy()
        relation_embedding = self.embeddings[1].weight.data.numpy()
        return entity_embedding, relation_embedding

    def get_embedding(self, index):
        index = index.astype(np.int64)
        embedding = self.emb_E(torch.tensor(index).cuda())
        return  embedding

    def get_vectorized_embedding(self, vector):
        if self.gpu:
            vector_emb = Variable(torch.from_numpy(vector).cuda())
        else:
            vector_emb = Variable(torch.from_numpy(vector))
        embedding_vector = self.emb_E(vector_emb).view(-1, self.embedding_dim)
        return embedding_vector
