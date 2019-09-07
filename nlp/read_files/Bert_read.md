# Bert 阅读
## Files
- https://www.aclweb.org/anthology/N19-1423 论文原文
- https://github.com/google-research/bert/blob/master/run_pretraining.py 代码源码
- https://daiwk.github.io/posts/nlp-bert-code.html 预训练文件配置


## read source
https://github.com/google-research/bert/blob/master/modeling.py
https://github.com/google-research/bert/blob/master/run_pretraining.py
```
# 这个函数会初始化 BertModel() 类
# BertModel 类会在初始化时即完成任务
# 1. input encoding with embedding_lookup(初始化with initializer) & position embedding;
# 2. transformer encoder part; 具体部分见；
model_fn = model_fn_builder()

# 初始化BertModel类后会根据transfomer 输出执行两个任务；
# 1. mask language model;
# 2. next sentence prediction;

(masked_lm_loss,
masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
bert_config, model.get_sequence_output(), model.get_embedding_table(),
masked_lm_positions, masked_lm_ids, masked_lm_weights)

(next_sentence_loss, next_sentence_example_loss,
next_sentence_log_probs) = get_next_sentence_output(
bert_config, model.get_pooled_output(), next_sentence_labels)
```

### 对于任务1 mask language model
get_masked_lm_output()

首先gather_indexes() 函数会把 transformer encoder的输出【bs, ts, F】和 相应的每个句子的masked token 词编号 [bs,  max_predictions_per_seq] 做gather 操作，输出结果为【Ts', F】, 代表整个batch所有句子（样本）中 被mask后 的词的词向量（transformer 编码后的词向量）；

然后dense 编码一下 即为input_tensor【Ts', F】，然后做如下操作
output_weights 实际上就是 emb_lookup, shape = [vocab_size, F]
所以logits shape: [Ts', vocab_size],  加个bias； 随机初始化是0向量， 但是会得到训练；

```
output_bias = tf.get_variable("output_bias", shape=[bert_config.vocab_size], initializer=tf.zeros_initializer())
logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
logits = tf.nn.bias_add(logits, output_bias)
log_probs = tf.nn.log_softmax(logits, axis=-1)
```

单个 label_id  类似如下， 由于padding 操作，后面会被补0；
label_weights 会记录哪些padding ,哪些没有；

```
INFO:tensorflow:masked_lm_ids: 22741 1010 2007 1012 2010 2001 20771 0 0 0 0 0 0 0 0 0 0 0 0 0
```

```
INFO:tensorflow:masked_lm_weights: 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
```

one_hot_labels() 函数会构成一个稀疏矩阵with shape:[Ts',  vocab_size];
log_probs * one_hot_labels 对应元素乘法， 得到稀疏矩阵，每行（Ts' 纬度）只有一个非零值，即该行代表的那个词的词序号的位置； reduce_sum  累加模式 得到 每个被masked(需要预测的)词的loss;
然后删除掉 padding word 对损失的影响；再reduce sum 得到整个batch 的损失；
```
label_ids = tf.reshape(label_ids, [-1])
label_weights = tf.reshape(label_weights, [-1])
one_hot_labels = tf.one_hot(
label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
numerator = tf.reduce_sum(label_weights * per_example_loss)
denominator = tf.reduce_sum(label_weights) + 1e-5
loss = numerator / denominator
```

### 对于任务2 next sentence prediction
input_tensor 实际上是
即【bs, F】 据说预处理过程中，每个样本开头都填充了一个 句子标记词【CLS】,
这样transformer 编码过后 作为整个句子的 embedding; 

```
[CLS] and there burst on phil ##am ##mon ' s astonished eyes a vast semi ##ci ##rcle of blue sea [MASK] ring ##ed with palaces and towers [MASK] [SEP] like most of [MASK] fellow gold - seekers , cass was super ##sti [MASK] . [SEP]
```
然后直接做dense 成【bs, 2】得到logits;

```
input_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
```

```
def get_next_sentence_output(bert_config, input_tensor, labels):

	with tf.variable_scope("cls/seq_relationship"):
		output_weights = tf.get_variable(
		"output_weights",
		shape=[2, bert_config.hidden_size],	      
		initializer=modeling.create_initializer(bert_config.initializer_range))

	output_bias = tf.get_variable("output_bias", shape=[2], initializer=tf.zeros_initializer())
	logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
	logits = tf.nn.bias_add(logits, output_bias)
	log_probs = tf.nn.log_softmax(logits, axis=-1)
	labels = tf.reshape(labels, [-1])
	one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
	per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
	loss = tf.reduce_mean(per_example_loss)
	return (loss, per_example_loss, log_probs)

```

