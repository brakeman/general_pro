from torch import nn
import torch
import ipdb


class Spatial_SE(nn.Module):
    '''
    conv2d(channel_in, channel_out, kernel=(1,1))
        # H_new = [(H_old+padding-1)/stride]+1
        # 这一层卷积操作会是的原始feature map[bs, channel_in, H, W] --> [bs, channel_out, H, W]
    '''
    def __init__(self, channel):
        super(Spatial_SE, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
#         ipdb.set_trace()
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class Channel_SE(nn.Module):
    '''
    conv1
    '''
    def __init__(self, channel, reduction=4):
        super(Channel_SE, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
#         ipdb.set_trace()
        z = self.global_avgpool(x)  # bs, channel, 1, 1
        z = self.relu(self.conv1(z)) # bs, channel//reduction, 1, 1
        z = self.sigmoid(self.conv2(z)) # bs, channel, 1, 1
        return x * z


class Spatial_Channel_SE(nn.Module):
    def __init__(self, channel):
        super(Spatial_Channel_SE, self).__init__()
        self.spatial_att = Spatial_SE(channel)
        self.channel_att = Channel_SE(channel)

    def forward(self, x):
        return self.spatial_att(x) + self.channel_att(x)
    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import ipdb


class PyramidAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PyramidAttention, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(input_dim),
        nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
        nn.BatchNorm2d(output_dim),
        nn.ELU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(input_dim),
        nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output_dim),
        nn.ELU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
        nn.BatchNorm2d(output_dim),
        nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.upsample(x_glob, scale_factor=16, mode='bilinear', align_corners=True)  # 256, 16, 16
        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4
        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4
        d3 = F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3
        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2
        x = x + x_glob
        return x
    
    
class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = torchvision.models.resnet34(True)
        self.conv1 = nn.Sequential( self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.encode2 = nn.Sequential(self.resnet.layer1, Spatial_Channel_SE(64))
        self.encode3 = nn.Sequential(self.resnet.layer2, Spatial_Channel_SE(128))
        self.encode4 = nn.Sequential(self.resnet.layer3, Spatial_Channel_SE(256))
        self.encode5 = nn.Sequential(self.resnet.layer4, Spatial_Channel_SE(512))
        self.center = nn.Sequential(PyramidAttention(512, 256), nn.MaxPool2d(2, 2))
        
    def forward(self, x):
        # x: (batch_size, 3, 256, 256)
        x = self.conv1(x)  # 64, 128, 128
        e2 = self.encode2(x)  # 64, 128, 128
        e3 = self.encode3(e2)  # 128, 64, 64
        e4 = self.encode4(e3)  # 256, 32, 32
        e5 = self.encode5(e4)  # 512, 16, 16
        f = self.center(e5)  # 256, 8, 8
        return f

    
class Decoder(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        '''
        up_in: channel of skip conneted layer; 
        x_in: channel of last layer;
        n_out: channel of output;
        '''
        super(Decoder, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(True)
        self.scSE = Spatial_Channel_SE(channel=n_out)

    def forward(self, up_p, x_p):
#         ipdb.set_trace()
        up_p = self.tr_conv(up_p) # [bs, 256, 8, 8] --> [bs, 32, 16, 16]
        x_p = self.x_conv(x_p) # [bs, 512, 16, 16] -->  [bs, 32, 16, 16]
        cat_p = torch.cat([up_p, x_p], 1) # [bs, 64, 16, 16]
        cat_p = self.relu(self.bn(cat_p))
        sc = self.scSE(cat_p) # [bs, 64, 16, 16]
        return sc
    
    
class Decoder_last(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder_last, self).__init__()
        self.conv1 = nn.Sequential(
                         nn.Conv2d(in_channels, channels, kernel_size=3, dilation=1, padding=1, bias=False),
                         nn.BatchNorm2d(channels), nn.ELU(True))
        
        self.conv2 = nn.Sequential(
                         nn.Conv2d(channels, out_channels, kernel_size=3, dilation=1, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ELU(True))
        self.scSE = Spatial_Channel_SE(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        sc = self.scSE(x)
        return sc

    
class Unet(nn.Module):
    def __init__(self, num_class=11):
        '''
        up_in: channel of skip conneted layer; 
        x_in: channel of last layer;
        n_out: channel of output;
        '''
        super(Unet, self).__init__()
        self.resnet = torchvision.models.resnet34(True)
        self.conv1 = nn.Sequential( self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.encode2 = nn.Sequential(self.resnet.layer1, Spatial_Channel_SE(64))
        self.encode3 = nn.Sequential(self.resnet.layer2, Spatial_Channel_SE(128))
        self.encode4 = nn.Sequential(self.resnet.layer3, Spatial_Channel_SE(256))
        self.encode5 = nn.Sequential(self.resnet.layer4, Spatial_Channel_SE(512))
        self.center = nn.Sequential(PyramidAttention(512, 256), nn.MaxPool2d(2, 2)) 
        self.decode5 = Decoder(256, 512, 64)
        self.decode4 = Decoder(64, 256, 64)
        self.decode3 = Decoder(64, 128, 64)
        self.decode2 = Decoder(64, 64, 64)
        self.decode1 = Decoder_last(64, 32, 64)
        self.logit = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(64, num_class, kernel_size=1, bias=False))
#         self.logit_image = nn.Linear(256, num_class)
        
    def forward(self, x):
        # x: (batch_size, 3, 256, 256)
        x = self.conv1(x)  # 64, 128, 128
        e2 = self.encode2(x)  # 64, 128, 128
        e3 = self.encode3(e2)  # 128, 64, 64
        e4 = self.encode4(e3)  # 256, 32, 32
        e5 = self.encode5(e4)  # 512, 16, 16
        f = self.center(e5)  # 256, 8, 8
#         for_cls = F.adaptive_avg_pool2d(f, output_size=1) # 256
        d5 = self.decode5(f, e5)  # 64, 16, 16
        d4 = self.decode4(d5, e4)  # 64, 32, 32
        d3 = self.decode3(d4, e3)  # 64, 64, 64
        d2 = self.decode2(d3, e2)  # 64, 128, 128
        d1 = self.decode1(d2)  # 64, 256, 256
        
        f = torch.cat((d1,
                       F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 320, 256, 256

        logit = self.logit(f)  # 1, 256, 256
#         clf = self.logit_image(for_cls.view(-1, 256)) # bs, 10
        return logit
    
    
    
if __name__ == '__main__':
    # scSE
    x = torch.randn(4, 64, 128, 128)
    sSE = Spatial_SE(channel=64)
    cSE = Channel_SE(channel=64)
    scSE = Spatial_Channel_SE(channel = 64)
    r1 = sSE(x)
    r2 = cSE(x)
    r3 = scSE(x)
    print(r1.shape)
    print(r2.shape)
    print(r3.shape)
    
    
    # encoder
    encoder = Encoder()
    img = torch.randn(4, 3, 256, 256)
    print(encoder(img).shape)
    
    
    # decoder
    img_up = torch.randn(4, 256, 8, 8)
    img = torch.randn(4, 512, 16, 16)
    decoder = Decoder(256, 512, 64)
    dec = decoder(img_up, img)
    print(dec.shape)

    # unet
    img = torch.randn(4, 3, 256, 256)
    unet = Unet()
    print(unet(img)[0].shape)