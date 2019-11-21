### kernel record

1. [IEEE kaggle](https://www.kaggle.com/nroman/eda-for-cis-fraud-detection)


2. 通过做实验，慢慢有了一种猜想
	- 基于事实： 当你添加无效特征[tra， test 分布不一致，区分能力不明显]的时候，精度会降；
	- 基于事实： 当添加有效特征的时候，精度会上升；
	- 检验事实： 当分布只有量级差异，没有趋势差异，会怎么影响精度？ 可以单变量lgb看看；
	



	因此要检查的，检测每个特征的有效性；
	- 有效性的定义是tra, test 分布一致性 + 区分能力的差异性
	- 图上当然可以看，也可以采取量化的方式， rank_cut + Count_enc + normalize; 
	- 计算两个分布的 diff.abs.sum()

3. 这个已经写好了，现在可以自动化遍历
	- 先单变量检查， 删除不好的变量，
	- 注意与疑问
		- 首先我这个只针对 数值变量， 且cats 不能太小，
		- 另外 iv值 也是如此作用，比赛为啥没人用呢？