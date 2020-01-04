## Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks

1. 印象
代码写好了 其实就是spatial SE + channel SE
channel SE 代表的是channel 维度上的attention 操作；
spatial SE 代表的是pixel 维度上的attention 操作；

- 两者attention score 都是通过 conv 实现的，
	后者是 conv2d(channel_in=channel_in, chanenl_out=1, no_pad, stride=1) 即 same 卷积；
	前者是 两个conv2d, 
		先下采样: 
		conv2d(channel_in= channel_in, channel_out=channel_in//reduction, no_pad, stride=1)
		再上采样： 
		conv2d(channel_in=channel_in//reduction, channel_out= channel_in, no_pad, stride=1) 也是same卷积；

### 代码
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
	    
	if __name__ == '__main__':
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