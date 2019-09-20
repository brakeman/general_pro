# some of the torch details

## 1. 卷积函数理解
    import torch
    import torch.nn as nn
    
    m = nn.Conv1d(2, 1, 3, stride=2) # in, out, kernal_size; 这是官方说法；
                                    # 自己理解就是 卷积核行数，卷积核个数，卷积核列数；
    print(m)
    input = torch.randn(2, 2, 3)   # 要求输入是 bs, ts, F
    print(input)
    output = m(input)
    print(m.weight)
    print(m.bias)
    print(output)              # 卷积每一次滑动 两个等shape的矩阵 元素级别 乘法 + bias;
    
    print(m.weight[0][0][0] * input[0][0][0] + m.weight[0][0][1] * input[0][0][1] + m.weight[0][0][2] * input[0][0][2] 
    + m.weight[0][1][0] * input[0][1][0] + m.weight[0][1][1] * input[0][1][1] + m.bias[0] + m.weight[0][1][2] * input[0][1][2]) 
    
    print(m.weight[0][0][0] * input[1][0][0] + m.weight[0][0][1] * input[1][0][1] 
    + m.weight[0][1][0] * input[1][1][0] + m.weight[0][1][1] * input[1][1][1] + m.bias[0])

## 2. 关于torch 自动推断 层输入；
- 在自定义model使用场景中常常面临一个情况，就是forward 要执行一层，比如全联接，需要在构造函数中 提前算好维度，这个很麻烦；
- 解决方案就是自己封装一个 fake_layer_class, 然后在 forward 中加一个shape 参数，这样就相当于动态的了; 