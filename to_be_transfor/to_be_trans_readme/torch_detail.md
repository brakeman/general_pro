# some of the torch details
## 0. shape 计算公式：
    H_new = [(H_old+2*pad-filter_size)/stride] + 1

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

## 2. torch 中padding = 1, 只代表左右扩充1，不包括上下两个方向；
    a = torch.randint(4, (bs,in_channels, f)).type(torch.float32)
    print('input:\n{}\n'.format(a))
    pad = torch.cat([torch.zeros(bs,in_channels,1), a, torch.zeros(bs,in_channels,1)],dim=2)
    print('after padding:\n{}\n'.format(pad))

    input:
    tensor([[[1., 1., 1.],
             [0., 0., 1.]]])

    after padding:
    tensor([[[0., 1., 1., 1., 0.],
             [0., 0., 0., 1., 0.]]])
## 3. torch 实现 causal conv
    class CausalConv1d(nn.Conv1d):
        def __init__(self, in_channels, out_channels, kernel_size=2, stride=1,
                     padding=1, dilation=1, groups=1, bias=False):
            super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               padding, dilation, groups, bias)

        def forward(self, inputs):
            outputs = super(CausalConv1d, self).forward(inputs)
            return outputs[:,:,:-1]

    in_channels = 2
    out_channels = 2
    kernel_size = 2
    dilation = 1
    bs = 1
    f = 3
    padding = 1
    
    causal_conv = CausalConv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size = kernel_size,
                              padding = padding,
                              dilation=dilation)
    w = torch.nn.Parameter(torch.FloatTensor(torch.randint(3, (out_channels, in_channels, kernel_size)).type(torch.float32)))
    causal_conv.weight = w
    print(causal_conv.padding)
    print(causal_conv.bias)

- 输出：
        (1,)
        None

- 展示
        a = torch.randint(4, (bs,in_channels, f)).type(torch.float32)
        print('input:\n{}\n'.format(a))
        pad = torch.cat([torch.zeros(bs,in_channels,1), a, torch.zeros(bs,in_channels,1)],dim=2)
        print('after padding:\n{}\n'.format(pad))
        print('weight:\n{}\n'.format(w))
        print(causal_conv(a))
        

- 结果分析： 
-输入：bs, f, ts
-filters: [num_fulters=2, in_channal = 2, kernel_size = 2] # 1d conv,  可以理解kernal_size 是卷积核列数；
-由于步长为1， 故卷积核在时间维度（输入的第三个维度）上单步移动，第一次移动可以看到 输入的第一个ts, 第二步移动可以看到 输入的第二个ts, ..., 最后一步移动可以看到输入的最后一个ts, 因果卷积在这里不看最后一个ts, 因此return 的结果为 outputs[:, :, :-1]； 因此因果卷积层核心就是不看最后一个ts;
        input:
        tensor([[[2., 0., 0.],
                 [3., 1., 3.]]])

        after padding:
        tensor([[[0., 2., 0., 0., 0.],
                 [0., 3., 1., 3., 0.]]])

        weight:
        Parameter containing:
        tensor([[[1., 1.],
                 [1., 2.]],

                [[1., 0.],
                 [2., 2.]]], requires_grad=True)

        tensor([[[ 8.,  7.,  7.],
                 [ 6., 10.,  8.]]], grad_fn=<SliceBackward>)
                 
## 4. [torch 空洞卷积理解](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) 
所有代码跟上面完全一致，只有一个参数dilation 改为2；
结果分析：由于dilation =2, 故平面卷积核在ts维度插入一列zero vector; 然后正常去跟paded input 做卷积；但是这里出现了信息在时间上的泄漏，卷积核第一次移动，看到了ts=0, ts=2； 卷积核第二次移动，看到了[ts=1, ts=3], 已知ts=3是数据最后一个 ts, 输出不可以看到，故泄漏； 所以不是随便dilation的，目测需要配合合适的padding;


        input:
        tensor([[[1., 0., 3.],
                    [3., 1., 0.]]])

        after padding:
        tensor([[[0., 1., 0., 3., 0.],
                    [0., 3., 1., 0., 0.]]])

        weight:
        Parameter containing:
        tensor([[[1., 0.],
                    [0., 1.]]], requires_grad=True)

        tensor([[[1., 1.]]], grad_fn=<SliceBackward>)