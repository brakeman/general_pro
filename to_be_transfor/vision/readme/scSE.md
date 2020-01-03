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

