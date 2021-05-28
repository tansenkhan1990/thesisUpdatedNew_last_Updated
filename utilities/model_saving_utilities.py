from model_utilities.model_initialization import model
from utilities.negative_sampling import sample_negatives, DNS_paper_modified, DNS_paper_orig, unique_negative_sampling
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




def save_model(model, optimizer, save_path, model_name):
    quad_models = ['complEx_quad', 'transR_quad', 'transE_quad', 'transH_element_quad', 'distmult_quad']

    # if model.name != 'RotatE' and model.name != 'transComplEx':
    #     entity_embedding = model.emb_E.weight.data.cpu().numpy()
    #     relation_embedding = model.emb_R.weight.data.cpu().numpy()
    #     '''
    #     Save the parameters of the model and the optimizer,
    #     as well as some other variables such as step and learning_rate
    #     '''
    #     model_name = model.name
    #     save_model_name = 'entity_embedding_' + model.name
    #     torch.save({
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict()},
    #         os.path.join(save_path, model_name)
    #     )
    #
    #     #entity_embedding = model.entity_embedding.detach().cpu().numpy()
    #     np.save(
    #         os.path.join(save_path, save_model_name),
    #         entity_embedding
    #     )
    #
    #     #relation_embedding = model.relation_embedding.detach().cpu().numpy()
    #     np.save(
    #         os.path.join(save_path, save_model_name),
    #         relation_embedding
    #     )
    if model.name == 'RotatE':
        entity_embedding_real = model.emb_E_real.weight.data.cpu().numpy()
        entity_embedding_im = model.emb_E_img.weight.data.cpu().numpy()
        relation_embedding = model.emb_R_phase.weight.data.cpu().numpy()
        '''
        Save the parameters of the model and the optimizer,
        as well as some other variables such as step and learning_rate
        '''

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(save_path, model_name)
        )

        # entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'entity_embedding_real_RotatE'),
            entity_embedding_real
        )
        np.save(
            os.path.join(save_path, 'entity_embedding_im_RotatE'),
            entity_embedding_im
        )

        # relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'relation_embedding_RotatE'),
            relation_embedding
        )
    elif model.name == 'transComplEx':
        entity_embedding = model.emb_E_real.weight.data.cpu().numpy()
        relation_embedding = model.emb_R_real.weight.data.cpu().numpy()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(save_path, model_name)
        )

        # entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'entity_embedding_transComplEx'),
            entity_embedding
        )

        # relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'relation_embedding_transComplEx'),
            relation_embedding
        )

    elif model.name == 'ComplEx':
        entity_embedding_real = model.emb_E_real.weight.data.cpu().numpy()
        relation_embedding_real = model.emb_R_real.weight.data.cpu().numpy()
        entity_embedding_im = model.emb_E_im.weight.data.cpu().numpy()
        relation_embedding_im = model.emb_R_im.weight.data.cpu().numpy()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(save_path, model_name)
        )

        # entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'entity_embedding_real_complex'),
            entity_embedding_real
        )

        # relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'relation_embedding_real_complex'),
            relation_embedding_real
        )
        np.save(
            os.path.join(save_path, 'entity_embedding_im_complex'),
            entity_embedding_im
        )

        # relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'relation_embedding_im_complex'),
            relation_embedding_im
        )

    elif model.name in quad_models:
        if model.name == 'transE_quad' or model.name == 'transR_quad' or model.name == 'distmult_quad' \
                or model.name == 'transH_element_quad':
            entity_embedding = model.emb_E.weight.data.cpu().numpy()
            relation_embedding = model.emb_R.weight.data.cpu().numpy()
            time_embedding = model.emb_tm.weight.data.cpu().numpy()
            location_embedding = model.emb_loc.weight.data.cpu().numpy()
            model_name = model.name
            save_model_name = '_embedding_' + model.name
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(save_path, model_name)
            )
            np.save(
                os.path.join(save_path, 'entity_embedding_' + save_model_name),
                entity_embedding
            )
            np.save(
                os.path.join(save_path, 'relation_embedding_'+ save_model_name),
                relation_embedding
            )
            np.save(
                os.path.join(save_path, 'time_embedding_' +  save_model_name),
                time_embedding
            )
            np.save(
                os.path.join(save_path, 'location_embedding_' + save_model_name),
                location_embedding
            )
            print('done saving')
        elif model_name == 'complEx_quad':
            entity_embedding_real = model.emb_E_real.weight.data.cpu().numpy()
            relation_embedding_real = model.emb_R_real.weight.data.cpu().numpy()
            entity_embedding_im = model.emb_E_im.weight.data.cpu().numpy()
            relation_embedding_im = model.emb_R_im.weight.data.cpu().numpy()
            time_embedding_real = model.emb_tim_real.weight.data.cpu().numpy()
            time_embedding_im = model.emb_tim_im.weight.data.cpu().numpy()
            location_embedding_real = model.emb_loc_real.weight.data.cpu().numpy()
            location_embedding_im = model.emb_loc_im.weight.data.cpu().numpy()
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(save_path, model_name)
            )

            # entity_embedding = model.entity_embedding.detach().cpu().numpy()
            np.save(
                os.path.join(save_path, 'entity_embedding_real_complex'),
                entity_embedding_real
            )

            # relation_embedding = model.relation_embedding.detach().cpu().numpy()
            np.save(
                os.path.join(save_path, 'relation_embedding_real_complex'),
                relation_embedding_real
            )
            np.save(
                os.path.join(save_path, 'entity_embedding_im_complex'),
                entity_embedding_im
            )

            # relation_embedding = model.relation_embedding.detach().cpu().numpy()
            np.save(
                os.path.join(save_path, 'relation_embedding_im_complex'),
                relation_embedding_im
            )
            np.save(
                os.path.join(save_path, 'time_embedding_real_complex'),
                time_embedding_real
            )
            np.save(
                os.path.join(save_path, 'time_embedding_im_complex'),
                time_embedding_im
            )
            np.save(
                os.path.join(save_path, 'location_embedding_real_complex'),
                location_embedding_real
            )
            np.save(
                os.path.join(save_path, 'location_embedding_im_complex'),
                location_embedding_im
            )

