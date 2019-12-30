import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# class FocalLossQb(nn.Module):
#     def __init__(self, gamma=2, sigmoid=True, OHEM_percent=None):
#         super(FocalLossQb, self).__init__()
#         self.gamma = gamma
#         self.sigmoid = sigmoid
#         self.OHEM_percent = OHEM_percent
        
#     def forward(self, input, target):
#         # input: bs, cls;   target: bs, 1
#         if len(target.shape)==1:
#             target = target.view(-1, 1)
#         elif len(target.shape)==2:
#             if target.shape[1] != 1:
#                 raise Exception('target shape not allowed')
#         else:
#             raise Exception('target shape not allowed')
#         ## 对于其它cls, 其损失，每个样本 num_cls-1 个scalar: loss = - ((pt)**gamma) *(log(1-pt))
#         ## 目标cls 用了个trick, 计算过程 log(1) = 0, 故不影响；
#         bs, cls = input.size()
#         one_hot_target = torch.FloatTensor(bs, cls).zero_()
#         one_hot_target.scatter_(1, target.long(), 1)
#         pt = 1 - torch.sigmoid(input)*(1-one_hot_target)
#         loss_other  = -1 * ((1-pt)**self.gamma) * pt.log()       
#         if self.OHEM_percent:
#             cls = int(cls * OHEM_percent)
#             loss_other, index= loss.topk(cls, dim=1, largest=True, sorted=True)
#         ## 对于目标cls, 其损失，每个样本一个scalar: loss = - ((1-pt)**gamma) *(logpt)
#         logpt = torch.sigmoid(input).log()
#         logpt = logpt.gather(1, target.long()).view(-1)
#         loss_target = -1 * ((1-logpt.exp())**self.gamma) * logpt
#         loss = loss_target.sum() + loss_other.sum()
#         return loss/bs/cls
    
    

# def focal_loss_tao(input, target):
#     gamma = 2
#     assert target.size() == input.size()
#     max_val = (-input).clamp(min=0)
#     loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
#     invprobs = F.logsigmoid(-input * (target * 2 - 1))
#     loss = (invprobs * gamma).exp() * loss
#     return loss.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn import functional as F
import math
# device = 'cpu'
device = 'cuda'


class ArcFace(torch.nn.Module):
    #norm-->dot-->arccos-->+margin-->logit
    def __init__(self, num_classes, emb_size, easy_margin, margin_m=0.5, banjing_s=64):
        super(ArcFace, self).__init__()

        self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes, emb_size)).to(device)
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.m = margin_m
        self.s = banjing_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m) # cos(pi-m)=-cos(m), 即设计了一个阈值（取代原来的90度阈值）
                                             # 当m=0.5, thresh = -cos(0.5), 这个阈值比原来宽松；
        self.mm = math.sin(math.pi - self.m) * self.m # sin(pi-m)*m=sin(m)*m; 没看懂;

    def forward(self, input, label):
        x = F.normalize(input) # bs, F
        W = F.normalize(self.weight) # cls, F
        cosine = F.linear(x, W) # bs, cls 代表每个样本，在每个cls上 与gt的cos(角度)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        cos_theta_m = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m) = cos(theta)*cos(m)-sin(theta)*sin(m)
        if self.easy_margin:
            cos_theta_m = torch.where(cosine > 0, cos_theta_m, cosine) # 如果角度在(0, 90), 则用cos(theta+m), 如果角度在(90,180), 则用原始cos(theta)
        else:
            # 0<theta+m<pi  等价于  -m<theta<pi-m 等价于 cos(theta)<cos(pi-m)
            cos_theta_m = torch.where(cosine > self.th, cos_theta_m, cosine - self.mm) # 
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cosine) # 如果预测正确用cos(theta+m)，否则用原始cos(theta)
        output *= self.s  
        arcloss = softmax_loss(output, label)
        return output, arcloss
    
    
def softmax_loss(results, labels):
    '''
    results: [bs, cls] 不需要softmax;
    labels:[bs,]
    '''
    labels = labels.view(-1)
    loss = F.cross_entropy(results, labels, reduce=True)
    return loss
    

class FocalLossQb(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLossQb, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # input: bs, cls;   target: bs, 1

        ## 对于其它cls, 其损失，每个样本 num_cls-1 个scalar: loss = - ((pt)**gamma) *(log(1-pt))
        ## 目标cls 用了个trick, 计算过程 log(1) = 0, 故不影响；
        bs, cls = input.size()
        device = input.device.type
        one_hot_target = torch.empty((bs, cls), device=device).zero_()
        one_hot_target.scatter_(1, target.view(-1,1), 1) # 把value=1 按照index=target.view(), 根据轴=1，填充到输入里；
        pt = 1 - torch.sigmoid(input)*(1-one_hot_target)
        loss_other  = -1 * ((1-pt)**self.gamma) * pt.log()

        ## 对于目标cls, 其损失，每个样本一个scalar: loss = - ((1-pt)**gamma) *(logpt)
        logpt = torch.sigmoid(input).log()
        logpt = logpt.gather(1, target.view(-1,1)).view(-1)
        loss_target = -1 * ((1-logpt.exp())**self.gamma)*logpt
        loss = loss_target.sum() + loss_other.sum()
        return loss/bs/cls # 粗略平均，没有考虑目标cls, 此处由于cls=5005, 故不影响

def bce_loss(input, target):
    bs, cls = input.size()
    device = input.device.type
    one_hot_target = torch.empty((bs, cls), device=device).zero_()
    one_hot_target.scatter_(1, target.view(-1,1), 1)
    loss = F.binary_cross_entropy_with_logits(input, one_hot_target)
    return loss


if __name__ == '__main__':
    device='cpu'
    batch_size = 4
    nb_digits = 10
    x = torch.rand(4,10)*random.randint(1,10)
    x[0][0]=1
    # print(x)

    y = torch.LongTensor(batch_size,1).random_() % nb_digits
    x = x.to(device)
    y = y.to(device)
    output0 = FocalLossQb(gamma=2)(x,y)
    output1 = bce_loss(x,y)
    output2 = ArcFace(num_classes=nb_digits, emb_size=x.shape[1], easy_margin=True)(x,y) 
    
    print(output1.item())
    print(output0.item())
    print(output2[1].item())

if __name__ == '__main__':
    batch_size = 4
    nb_digits = 10
    x = torch.rand(4,10)*random.randint(1,10)
    x[0][0]=1
    print(x)
    y = torch.LongTensor(batch_size,1).random_() % nb_digits
    y_onehot = torch.FloatTensor(batch_size, nb_digits).zero_()
    y_onehot.scatter_(1, y, 1)
    output0 = FocalLossQb(gamma=2)(x,y)
    output1 = focal_loss_tao(x, y_onehot)
    print(output1)
    print(output0)