import tensorflow as tf


# 之前写的pytorch 版本；
# def _nec_forward(self, nodes, neighs, negNeighs):
#     '''unsupervised nce loss;
#     :nodes: barch node ids [bs];
#     :neighs: positive samples with regard to nodes [bs, K], K is num of neighs for each node;
#     :negNeighs: negative samples with regard to nodes [bs, K];'''
#     bs = len(nodes)
#     # bs, F, 1
#     nodes = self.features(torch.LongTensor(nodes)).unsqueeze(-1)
#     # bs, 1, F
#     pos_nodes = torch.stack([self.features(torch.LongTensor(neighs[i])) for i in range(len(neighs))])
#     # bs, 5, F
#     neg_nodes = torch.stack([self.features(torch.LongTensor(negNeighs[i])) for i in range(len(negNeighs))])
#
#     sum_log_neg = torch.bmm(neg_nodes, nodes).neg().sigmoid().log().squeeze().sum()
#     sum_log_pos = torch.bmm(pos_nodes, nodes).sigmoid().log().squeeze().sum()
#     #         print('sum neg loss:{}/n sum pos loss:{}'.format(sum_log_neg, sum_log_pos))
#     return (sum_log_pos + sum_log_neg) / bs



def nec_loss(inputs1, inputs2, neg_samples, neg_sample_weights):
    aff = tf.reduce_sum(inputs1 * inputs2, axis=1)
    neg_aff = tf.matmul(inputs1, tf.transpose(neg_samples))

    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
    negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)

    loss = tf.reduce_sum(true_xent) + neg_sample_weights * tf.reduce_sum(negative_xent)
    return loss

