# xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

## 链接

## 对该文章的印象：
- 这个公式看半天，初看之下应该是这样：
	- 公式里 元素相乘的两个大X, 分别代表 第一层X_0 和 每个更新后的层X_k, 跟DCN 递归结构一模一样；
	- 里面元素相乘，故shape不变，注意W_k_h, 代表第k个layer，第h个‘卷积核’（类比这个概念）， 先不管W看，每一次 这种运算： 得到一个D维向量，代表第h 个‘卷积核’ 的feature map; 那么最终结果是 shape = [H, D]
	- 以上是只看公式；
	- 注意，图与公式很难第一眼看出对应的，先独立去看；
		- 图是两步走，第一步是 求出立方体Z_k, 第二步是求X_k， 这两部对应整个公式；
		- 第一步需要做外积，'btf, byf --> btyf'
		- 卷积这里具体是： 卷积核是全部面积大小，但是维度只是2维，一次卷机操作，会把整个平面 变成一个点， 有H_k个卷积核，故变成一个长度为H_k的向量，有D个通道（人家本来是3D，可是你的卷积核是2D, 所以在第三个维度上独立做卷积操作），最后卷积操作 弄出 H_k*D 的平面, 这就是一个layer;
	- 再回头看公式，看那个W,  在想象卷积，是不是对应的，同一的？
- 代码实现上, 看好多人的源码都是先reshape 成3D, input 本来是应该是4D[bs,ts1,ts2, F], 然后3D输入配合 列卷积核【-1, 1], 我实在看不懂怎么就能 用列卷积核了，
从我自己角度来看，工程上是可以按照我上面描述的去做的：
- 立方体Z_k 3D矩阵，reshape 为2D, 只需要用一个 平面卷积核[ts2,F], 配合步长为 ts2, 即可做到；
- CIN 层：![Drag Racing](../pics/xDeepFM/xDeepFM_1.jpg)
- CIN 公式： ![Drag Racing](../pics/xDeepFM/xDeepFM_2.jpg)
- 整体： ![Drag Racing](../pics/xDeepFM/xDeepFM_3.png)