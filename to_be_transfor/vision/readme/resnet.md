## Deep Residual Learning for Image Recognition

## 印象：
- 越深真的越好
- problem of vanishing/exploding gradients
- easier to optimize the residual mapping than to optimize the original, unreferenced mapping
- 有一些方法可以弥补，如归一初始化，各层输入归一化，使得可以收敛的网络的深度提升为原来的十倍。然而，虽然收敛了，但网络却开始退化了，
- 好处：
    - Shortcut Connections. 结构不增加任何参数
    - 优化残差更容易（导数为1+dF(x)）