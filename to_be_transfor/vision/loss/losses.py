import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class FocalLossQb(nn.Module):
    def __init__(self, gamma=2, sigmoid=True):
        super(FocalLossQb, self).__init__()
        self.gamma = gamma
        self.sigmoid = sigmoid
        
    def forward(self, input, target):
        # input: bs, cls;   target: bs, 1

        ## 对于其它cls, 其损失，每个样本 num_cls-1 个scalar: loss = - ((pt)**gamma) *(log(1-pt))
        ## 目标cls 用了个trick, 计算过程 log(1) = 0, 故不影响；
        bs, cls = input.size()
        one_hot_target = torch.FloatTensor(bs, cls).zero_()
        one_hot_target.scatter_(1, target.long(), 1)
        pt = 1 - torch.sigmoid(input)*(1-one_hot_target)
        loss_other  = -1 * ((1-pt)**self.gamma) * pt.log()       
        
        ## 对于目标cls, 其损失，每个样本一个scalar: loss = - ((1-pt)**gamma) *(logpt)
        logpt = torch.sigmoid(input).log()
        logpt = logpt.gather(1, target.long()).view(-1)
        loss_target = -1 * ((1-logpt.exp())**self.gamma) * logpt
        loss = loss_target.sum() + loss_other.sum()
        return loss/bs/cls
    
    

def focal_loss_tao(input, target):
    gamma = 2
    assert target.size() == input.size()
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    invprobs = F.logsigmoid(-input * (target * 2 - 1))
    loss = (invprobs * gamma).exp() * loss
    return loss.mean()


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