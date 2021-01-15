from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader
from utilities.dataloader import TestDataset_2


def test_step(model, test_triples, all_true_triples, args):
    '''
    Evaluate the model on test or valid datasets
    '''

    model.eval()

    # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
    # Prepare dataloader for evaluation
    test_dataloader_head = DataLoader(
        TestDataset_2(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'head-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TestDataset_2.collate_fn
    )

    test_dataloader_tail = DataLoader(
        TestDataset_2(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'tail-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TestDataset_2.collate_fn
    )

    test_dataset_list = [test_dataloader_head, test_dataloader_tail]

    logs = []
    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                if args.cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                batch_size = positive_sample.size(0)
                out = combine_pos_neg(pos=positive_sample, neg=negative_sample, mode=mode)
                if model.regul:
                    score,regul = model.forward(out.cpu().numpy())[0].view(batch_size, -1)
                else:
                    score = model.forward(out.cpu().numpy()).view(batch_size, -1)
                score += filter_bias

                # Explicitly sort all the entities to ensure that there is no test exposure bias
                if model.name == 'distmult' or model.name == 'complEx':
                    argsort = torch.argsort(score, dim=1, descending=True)
                else:
                    argsort = torch.argsort(score, dim=1, descending=False)
                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 1]
                else:
                    raise ValueError('mode %s not supported' % mode)

                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    # ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0 / ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@5': 1.0 if ranking <= 5 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

    print(metrics)
    return metrics


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))



def combine_pos_neg(pos, neg, mode):
  entity_size = neg.shape[1]
  batch_size = pos.shape[0]
  if mode=='head-batch':
    pos = pos[:,1:].repeat(1,entity_size).view(-1,2)
    neg = neg[0].repeat(1,batch_size).view(entity_size*batch_size, -1)
    out = torch.cat((neg,pos), dim=1)
  else:
    pos_e1 = pos[:,:1].repeat(1,entity_size).view(-1,1)
    pos_rel = pos[:,pos.shape[1]-1:].repeat(1,entity_size).view(-1,1)
    neg = neg[0].repeat(1,batch_size).view(entity_size*batch_size, -1)
    out = torch.cat((pos_e1,neg,pos_rel), dim=1)

  return out