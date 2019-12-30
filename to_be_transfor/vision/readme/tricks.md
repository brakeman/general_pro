## Bag of Tricks for Image Classification with Convolutional Neural Networks

## 印象

-  Linear scaling learning rate: 大的batch size不会降低偏差，但是可以降低方差，因此可以用大的lr; 但是在init的时候，如果lr太大 往往 unstable; 

- Learning rate warmup： 刚开始用小的lr, 等到训练稳定了再加大lr。

- BN 层的设计思路
	[BN](../pics/bn.pic_hd.jpg)
	γxˆ + β

	In the zero γ initialization heuristic, we initialize
	γ = 0 for all BN layers that sit at the end of a residual block.
	Therefore, all residual blocks just return their inputs, mimics network that has less number of layers and is easier to
	train at the initial stage.

	他意思是， 例如每一层最后都是 res 结构： x_new = x + BN(block(x)) = x + γxˆ + β, 在训练初期，
	设置γ = 0， 好处是ez to train in the init stage

- No bias decay	
	平常的 L2 正则是针对 所有的 Weights & Bias & other learnable paras;
	这里推荐 只针对所有层的 Weights 做L2正则， 对于Bias 和 BN 层的 γ, β 都不做l2正则；

- Low-precision training
	the overall training speed is accelerated by 2 to 3 times after switching from FP32 to FP16 on V100
	[float16](../pics/float16.pic_hd.jpg)



