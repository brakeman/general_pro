# Neural Factorization Machines for Sparse Predictive Analytics

## 链接

## 对该文章的印象：
- 同样的，文章扩展出一个field 维度[bs, fields, F]，每个样本在field 维度上做交互（BIP 核心是 元素相乘）
- 模型结构：![Drag Racing](../pics/NFM/NFM_1.jpg)
- BIP层 ![Drag Racing](../pics/NFM/NFM_2.jpg)
- 等价于![Drag Racing](../pics/NFM/NFM_3.jpg)