# WiveNet 代码阅读


## [链接](https://github.com/vincentherrmann/pytorch-wavenet)


### 理解数据：
          |----receptive_field----|
                                |--output_length--|
example:  | | | | | | | | | | | | | | | | | | | | |
target:                           | | | | | | | | | |  


就是一段音频，前一部分是x, 后一部分是y;
样本总长度 = receptive_field + output_length - 1 = 3085
x:[bs,f=256,ts=3085]
y:[bs,ts=16], output_length=16，最后16个ts作为label; 每个label 有256个class; 只不过此处没有用one-hot 扩展为3D 的y, 只需进入损失函数 torch.cross_entropy(), 自动完成；


#### 诡异， 输入shape不是恒定的, 这个 样本总长度 由 receptive_field 决定， 而这个参数，tmd 是根据model 层数等 算出来的，即样本取决于模型，我第一次见这个；理论上说，可以调整到一个点，试验出使得感受野为最大时的模型参数


### 模型：

