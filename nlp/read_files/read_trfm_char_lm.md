# Character-Level Language Modeling with Deeper Self-Attention
## 本文实际上是transformer + seq2seq 的一些实践方案；同时该文章也是transformer-XL 的前置文章之一；
## 链接
 [https://arxiv.org/pdf/1808.04444.pdf](https://arxiv.org/pdf/1808.04444.pdf)

## 梗概
- 文章的训练目标是 process fixed-length inputs and predict upcoming characters； 即字符级别语言模型；
- 文章在seq2seq 框架下 ，使用标准 transformer 结构，编码器中使用causal mask 技术，解码过程设置了3个辅助损失函数；
- 强化了我对causal mask 的理解，其对seq2seq 架构的工程用法也拓宽了我的眼界；


## 一些有用的摘要
- lstm 'long' dependency is not long enough;
For example Khandelwal et al. (2018) find that a word-based LSTM language model only effectively uses around 200 tokens of context (even if more is provided), and that word order only has an effect within approximately the last 50 tokens

- 模型训练框架 单个样本实例：
- X：[w h o s]
- Y:   [e]
- causal mask 机制使得 每个ts 都只能吃到自己和自己左侧；在 ts = 3（字母s） 是 encoding 出一个【bs， 1， F】的结果，作为对 ts = 4 的预测编码；


### 辅助损失1:Multiple Positions
- 即不只预测ts=4(字母e)， 每个位置都预测；毕竟编码器中的 causal mask 不会使得信息泄露；
- X: [w, h, o, s]
- Y: [h, o, s, e]


### 辅助损失2：Intermediate Layer Losses
- 基于多层 transformer 结构， 每一层都可以有预测，都可以有损失，那么设计思路就是底层损失 --> 高层损失 权重递增；
- 多层损失贡献立即消失 当一个epoch 完成一半的时候，就不再用这些层损失了；

### 辅助损失3: Multiple Targets
- At each position in the sequence, the model makes two (or more) predictions of future characters. For each new target we introduce a separate classifier. The losses of the extra targets get weighted by a multiplier of 0.5 before being added to their corresponding layer loss.

### transformer-xl 对该文缺点的阐述：
- 对于单个样本没啥问题：
	- X: [w, h, o, s]
	- Y: [e]
- 可是如果 以一整篇文章的角度看，缺点两个：
	- 1。 不尊重 句子边界，可能真个句子都被打散了；
	- 2。 trfm 虽然可以多远都能看到，但是前提是 不超过len(X)这么长，当你把一篇文章打散（在language model 中 构建样本时候 这尤其自然），对于文章中段，任意单个样本，不可能看到更早的样本；
	- 原来如此，之所以相对位置编码，是因为如果还继承 origin trfm 的绝对位置编码，那么对一整篇文章的所有切片来说，每个切片都共享相同的位置编码；自然而然的需要 跨切片 的 相对位置编码技术； 


