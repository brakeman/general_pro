# Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks

## 链接

## 对该文章的印象：
- 这文章就是 NFM(BIP层) 加一个简单的attention 机制，这都水上论文了；
- 注意, 这里的W,b是共享参数，任何v_i,v_j 元素相乘之后都是乘以这个W;
- 模型结构：![Drag Racing](../pics/AFM/AFM_1.jpg)
- attention 层： ![Drag Racing](../pics/AFM/AFM_3.jpg)
- 最终公式：![Drag Racing](../pics/AFM/AFM_2.jpg)
![Drag Racing](../pics/NFM/NFM_1.jpg)
