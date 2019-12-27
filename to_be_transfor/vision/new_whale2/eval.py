import numpy as np
import torch
from loss import FocalLossQb, bce_loss

def map_per_image(label, predictions):
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0

def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list: ['qb', 'qb', 'zac', 'jam', ... , 'zac']
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list: [['qb', 'zac', 'ben', 'jer', 'gam'], ['qb', 'zac', 'ben', 'jer', 'gam'], ..., ['qb', 'zac', 'ben', 'jer', 'gam']]
             A list of predicted elements (order does matter, 5 predictions allowed per image)
    """
    return np.mean([map_per_image(l, list(p)) for l,p in zip(labels, predictions)])


def do_valid(net, valid_loader, device):
    loss1_list = []
    loss2_list = []
    label_list = []
    prob_list = []
    with torch.no_grad():
        for input, truth_, in valid_loader:
            input = input.to(device)
            truth_ = truth_.to(device)
            logit = net(input)
            loss1 = FocalLossQb(gamma=2)(logit, truth_)
            loss2 = bce_loss(logit, truth_)
#             ipdb.set_trace()
            _, top5_idx = logit.sigmoid().topk(5)
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())
            label_list.extend(truth_.tolist())
            prob_list.extend(top5_idx.tolist())
    loss1 = np.mean(loss1_list)
    loss2 = np.mean(loss2_list)
    map_5 = map_per_set(label_list, prob_list)
    return loss1, loss2, loss1+loss2, map_5, label_list[:5], prob_list[:5]