# model resnet101 
import sys
sys.path.append('../')
from torch.nn import functional as F
import json
import torch
from torch import nn
import torchvision.models as tvm
from data import WhaleData

###########################################################################################3
class Net(nn.Module):
    # 看看to_be_transer_readme, 经典的resnet101一共四层，算上头尾一共6层；

    def __init__(self, num_class):
        super(Net,self).__init__()      
        self.basemodel = tvm.resnet101(pretrained=True)
        self.basemodel.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 这行意思是： 稍后用到 fine_tuning 需要冻结前面两层参数，
        # 由于resnet 预训练模型 前四步是 conv+bn+relu+pool, 为了方便，整理到一个layer0中；
        self.basemodel.layer0 = nn.Sequential(self.basemodel.conv1,
                                              self.basemodel.bn1,             
                                              self.basemodel.relu,            
                                              self.basemodel.maxpool)         
        emb_size = 2048
        self.qb_layer = nn.Linear(emb_size, num_class)

    def forward(self, x):
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
        fea = self.qb_layer(fea)
        return fea
    
if __name__ == '__main__':
    WD = WhaleData(mode='train')
    test_sample_2 = torch.cat([WD[0][0].view(1, 3, 128, -1), WD[1][0].view(1, 3, 128, -1)])

    resnet = Net(num_class=5005)
    logit = resnet(test_sample_2)
    print(logit.shape)
    print(logit)
