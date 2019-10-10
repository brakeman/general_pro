## paddle 笔记

- 经典例子
<https://github.com/brakeman/general_pro/blob/master/to_be_transfor/paddle/models/run.ipynb>


- AI studio 中 怎么启动 GUI?
	- 复制 url
	- 把 notebooks 之后的改为
https://aistudio.baidu.com/bdvgpu/user/127073/136105/notebooks/136105.ipynb?redirects=1
https://aistudio.baidu.com/bdvgpu/user/127073/136105/visualdl/static/index.html

- conv2d
	- 输入和输出是NCHW格式
	- 如果组数大于1，C等于输入图像通道数除以组数的结果
	- 滤波器shape： (C_out,C_in,H_f,W_f)
	- ![conv_formula](../pics/conv2d_1.png)
	- ![conv_formul](../pics/conv2d_2.png)
	- 理解一下 默认padding=0, dilation=1, stride=1; 
	- 30 = (32+0-1*(3-1)+1)/1  + 1
	
- pool2d
	- 默认参数 stride =1, padding=0
	- ![conv_formul](../pics/pool2d.png)
	
	

