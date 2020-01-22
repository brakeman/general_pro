import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import ipdb
from torch import nn


class DiceLoss2D(nn.Module):
    
    def __init__(self, cls_num, device, weights=True):
        super(DiceLoss2D, self).__init__()
        self.device = device
        self.weights = weights
        self.cls_num = cls_num
            
    def compute_class_weights(self, histogram):
        classWeights = np.ones(self.cls_num, dtype=np.float32)
        normHist = histogram / np.sum(histogram)
        for i in range(self.cls_num):
            classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
        return classWeights
    
    def forward(self, output, target):
        """
        output : NxCxHxW Variable
        target :  NxHxW LongTensor
        weights : C FloatTensor
        """
        eps = 0.0001
        output = output.exp()
        encoded_target = output.detach() * 0
        encoded_target.scatter_(1, target.unsqueeze(1).long(), 1)
        if self.weights is None:
            weights = 1
        else:
            frequency = torch.tensor([torch.sum(target == i).item() for i in range(self.cls_num)], dtype=torch.float32).numpy()
            weights = self.compute_class_weights(frequency)
        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = torch.tensor(weights).to(self.device) * (1 - (numerator / denominator))
        return loss_per_channel.sum() / output.size(0)

# if __name__=='__main__':
#     pred=torch.rand((3,6,5,5))
#     device='cuda'
#     y=torch.from_numpy(np.random.randint(0,6,(3,5,5)))
#     loss2 = DiceLoss2D(6, weights=True)(pred.to(device), y.to(device))
#     print('loss', loss2)
    
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss2D_2(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    """

    def __init__(self, device, num_class, cls_weights=True, gamma=2, smooth=0.2):
        super(FocalLoss2D_2, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.smooth = smooth
        self.cls_weights = cls_weights
        self.device = device
        
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def compute_class_weights(self, histogram):
        classWeights = np.ones(self.num_class, dtype=np.float32)
        normHist = histogram / np.sum(histogram)
        for i in range(self.num_class):
            classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
        return classWeights
    
    def forward(self, logit, target):
        
        # 1. logit: N, C; 元素级别logit
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        
        # 2.one_hot_target; 元素级别label one hot; [bs*H*W, C];
        target = target.view(-1, 1)
        epsilon = 1e-10
        one_hot_target = torch.FloatTensor(target.size(0), self.num_class).zero_().to(self.device)
        one_hot_target = one_hot_target.scatter_(1, target.long().to(self.device), 1)
        
        # 3.focal loss + label smooth; 把one_hot_target中的1的 smooth 部分平均分给其它类；
        if self.smooth:
            one_hot_target = torch.clamp(
                one_hot_target, self.smooth/(self.num_class-1), 1.0 - self.smooth)

        # 4. class weights: 元素级别 weights; [bs*h*W,1]
        if self.cls_weights:
            frequency = torch.tensor([torch.sum(target == i).item() for i in range(self.num_class)], dtype=torch.float32).numpy()
            classWeights = self.compute_class_weights(frequency)
            weights = torch.from_numpy(classWeights).float()
            weights=weights[target.view(-1).long()]#这行代码非常重要

        # 5. focal loss 公式；
        pt = (one_hot_target * logit.softmax(dim=1)).sum(1) + epsilon
        logpt = pt.log()
        loss = -1 * weights.to(self.device) * torch.pow((1 - pt), self.gamma) * logpt
        return loss.mean()

    
    
class FocalLoss2d(nn.Module):
    
    def __init__(self, cls_num, device, cls_weights=True, gamma=2):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.cls_weights = cls_weights
        self.device = device
        self.cls_num = cls_num
        
    def compute_class_weights(self, histogram):
        classWeights = np.ones(self.cls_num, dtype=np.float32)
        normHist = histogram / np.sum(histogram)
        for i in range(self.cls_num):
            classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
        return classWeights

    def forward(self, input, target):
        '''
        :param input: [bs, cls, h, w]
        :param target: [bs, h, w]
        :return: scalar
        '''
        n, c, h, w = input.size()
        target = target.long()
        inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.contiguous().view(-1)
        N = inputs.size(0)
        C = inputs.size(1)
        
        if self.cls_weights:
            frequency = torch.tensor([torch.sum(target == i).item() for i in range(self.cls_num)], dtype=torch.float32).numpy()
            classWeights = self.compute_class_weights(frequency)
            weights = torch.from_numpy(classWeights).float()
            weights=weights[target.view(-1)]#这行代码非常重要

        P = F.softmax(inputs, dim=1) #shape [num_samples,num_classes]
        class_mask = torch.zeros([N, C]).to(self.device)
        ids = target.view(-1, 1).long()
        class_mask.scatter_(1, ids, 1.)#shape [num_samples,num_classes]  one-hot encoding
        probs = (P * class_mask).sum(1).view(-1, 1)#shape [num_samples,]
        log_p = probs.log()
        if self.cls_weights:
            batch_loss = - weights.to(self.device) * (torch.pow((1 - probs), self.gamma)) * log_p
        else:
            batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        return batch_loss.mean()
    
# if __name__=='__main__':
#     device = 'cuda'
#     pred=torch.rand((2,6,5,5))
#     y=torch.from_numpy(np.random.randint(0,6,(2,5,5)))
#     loss3=FocalLoss2d(device, cls_weights=True)(pred.to(device),y.to(device))
#     print('loss3',loss3)
    
    

# --------------------------- MULTICLASS LOSSES ---------------------------
def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
#         ipdb.set_trace()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def multi_class_entropy(logits, labels, ignore=None):
    """
    Cross entropy loss;
    logits: [bs,C,H,W], C is num_class
    labels: [bs,H,W], each entry with int value in [0, C-1]
    """
    return F.cross_entropy(logits, labels.long(), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

if __name__ == '__main__':
    img = torch.rand(10, 3, 120,120)
    mask =torch.randint(0, 3, (10, 120, 120))
    loss1 = multi_class_entropy(img, mask)
    loss2 = lovasz_softmax(img, mask, per_image=False)
    print(loss1, loss2)
