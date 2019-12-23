## FM general ideas

从基本结构上来说，交互有两种方式：元素对应乘法 & 向量内积；
举个例子，假设只有四个维度： 年龄，性别，城市，国家。
- 元素对应乘法（NFM, AFM, DeepCross， xDeepFM） type(年龄@性别) == vector;
    - NFM 交互后直接相加了（沿着fields 维度）
    - AFM 交互后做了attention 相加
    - DCN 先是Dense(1) 然后广播元素乘法 + ResNet
    - xDeepFM 先做外积, 每个tx, 每个ty, 做Fx @ Fy； 然后用CNN做降维； + ResNet;

- 向量内积（DeepFM）: type(年龄@性别) == scalar， 年龄@城市，年龄@国家，性别@城市，性别@国家，城市@国家，每个单元为一个scalar,恰好为上三角阵。


