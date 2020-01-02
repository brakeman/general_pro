# model resnet101 
import sys
sys.path.append('../')
from torch.nn import functional as F
import json
import torch
from torch import nn
import torchvision.models as tvm
from data import WhaleData
from loss import *

###########################################################################################3
class Net(nn.Module):
    # 看看to_be_transer_readme, 经典的resnet101一共四层，算上头尾一共6层；

    def __init__(self, model_name, num_class, arcFace, device):
        super(Net,self).__init__()     
        assert model_name in ['50', '101']
        if model_name == '50':
            self.basemodel = tvm.resnet50(pretrained=True)
        elif model_name == '101':
            self.basemodel = tvm.resnet101(pretrained=True)
        self.basemodel.avgpool = nn.AdaptiveAvgPool2d(1)
        # 这行意思是： 稍后用到 fine_tuning 需要冻结前面两层参数，
        # 由于resnet 预训练模型 前四步是 conv+bn+relu+pool, 为了方便，整理到一个layer0中；
        self.basemodel.layer0 = nn.Sequential(self.basemodel.conv1,
                                              self.basemodel.bn1,             
                                              self.basemodel.relu,            
                                              self.basemodel.maxpool)         
        emb_size = 2048
        
        self.arcFace=arcFace
        if arcFace:
            self.qb_layer = ArcFace(num_classes=num_class, emb_size=emb_size, easy_margin=True, device=device)
        else:
            self.qb_layer = nn.Linear(emb_size, num_class)

    def forward(self, x, y=None):
        # y: [bs,]
        mean = [0.485, 0.456, 0.406]  # rgb
        std = [0.229, 0.224, 0.225]     

        x = torch.cat([
            (x[:, [0]] - mean[0]) / std[0], 
            (x[:, [1]] - mean[1]) / std[1], 
            (x[:, [2]] - mean[2]) / std[2], 
        ], 1)

        x = self.basemodel.layer0(x)    
        x = self.basemodel.layer1(x)    
        x = self.basemodel.layer2(x)    
        x = self.basemodel.layer3(x)    
        x = self.basemodel.layer4(x)    
        x = self.basemodel.avgpool(x)
        fea = x.view(x.size(0), -1)    
        if self.arcFace:
            logit, loss = self.qb_layer(fea, y)
            return logit, loss
        else:
            fea = self.qb_layer(fea)
            return fea
    
if __name__ == '__main__':
    device = 'cuda'
    
    WD = WhaleData(mode='train')
    x = torch.cat([WD[0][0].view(1, 3, 128, -1), WD[1][0].view(1, 3, 128, -1)]).to(device)
    y = torch.LongTensor([WD[0][1], WD[1][1]]).to(device)
    
#     resnet = Net(num_class=5005, arcFace=False).to(device)
#     logit = resnet(x)
#     print(logit.shape)
#     print(logit)
    
    resnet = Net(num_class=5005, arcFace=True, device=device).to(device)
    logit, loss = resnet(x, y=y)
    print(loss.item())