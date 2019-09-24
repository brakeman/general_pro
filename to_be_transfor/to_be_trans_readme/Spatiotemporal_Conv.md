# WiveNet


## 链接

## 对该文章的印象：
- 作用于多个任务：
    - text2audio, 
    - generates novel and often highly realistic musical fragments
    -  it can be employed as a discriminative model, returning promising results for phoneme recognition.
- 模块包括：
    - DILATED CAUSAL CONVOLUTIONS 
    - reconstructed signal;  ![Drag Racing](../pics/wavenet_1.png)
    - GATED ACTIVATION UNITS ![Drag Racing](../pics/wavenet_2.png)
    -  RESIDUAL AND SKIP CONNECTIONS
    - CTC loss again;
    - 
- WiveNet 引入了causal 卷积 & 空洞 卷积，前者为 cnn 编码序列数据提供依据，后者在不增加层数的前提下提高感受野；
    - For images, the equivalent of a causal convolution is a masked convolution (van den Oord et al., 2016a) which can be implemented by constructing a mask
tensor and doing an elementwise multiplication of this mask with the convolution kernel before applying it. 

    - For 1-D data such as audio one can more easily implement this by shifting the output of a
normal convolution by a few timesteps.

- code

        class CausalConv1d(nn.Conv1d):
            def __init__(self, in_channels, out_channels, kernel_size=2, stride=1,
                         padding=1, dilation=1, groups=1, bias=False):
                super(CausalConv1d, self).__init__(in_channels, out_channels, 
                        kernel_size, stride, padding, dilation, groups, bias)
            
            def forward(self, inputs):
                outputs = super(CausalConv1d, self).forward(inputs)
                return outputs[:,:,:-1]
        
            
    class DilatedConv1d(nn.Conv1d):
        def __init__(self, in_channels, out_channels, kernel_size=2, stride=1,
                     padding=0, dilation=1, groups=1, bias=False):
            super(DilatedConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                padding, dilation, groups, bias)
        
        def forward(self, inputs):
            outputs = super(DilatedConv1d, self).forward(inputs)
            return outputs