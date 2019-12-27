import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# import ipdb


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

    print(output1.item())
    print(output0.item())