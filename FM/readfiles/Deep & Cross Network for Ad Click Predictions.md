# Deep & Cross Network for Ad Click Predictions

Deep & Cross Network for Ad Click Predictions
## 链接

## 对该文章的印象：
- 每层用到残差结构 
- cross layer 
![Drag Racing](../pics/deep_cross/deep_cross_1.png)
	- 两个注意：
		- 1. 没有bs 维度间的交互；你先不要 bs,F 这样去看（这样看会变成bs 维度间的交互），先单个样本去看，发现没有bs 间交互;
		- 2. 无论求第几层，x0 永远不变，不会随着层的变化而变化；
		- 3. 注意参数w不是个参数矩阵，是个vector；
		- 4. 先后面两项做矩阵乘法变成【bs,1】，然后后前面元素相乘[bs, F]；
- model structure 
![Drag](../pics/deep_cross/deep_cross_2.png)
