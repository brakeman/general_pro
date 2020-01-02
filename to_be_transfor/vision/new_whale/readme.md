## 

1. 关于lr
	- 实践上的技巧是 先做lr range search to get max, min thresh;
		- 0.00001
		- 0.1
		- 0.01
	- 然后用循环式的lr schedual;

2. 关于augmentation
	- 0. 采样策略：0.8的非other图片， 0.2的other图片;
	- 1. train aug, valid 除了flip_fake_label，其它都不aug;
	- 2. flip_fake_label 策略：train 每张图片flip并委派新的label; valid 也是如此;
	- 3. aug flow;

3. 


