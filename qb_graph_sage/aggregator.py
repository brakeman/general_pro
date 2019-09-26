# GCN agg;
import tensorflow as tf


class MeanAggregator:
    """
    Aggregates via mean followed by matmul and non-linearity.
    I: self_feats, neigh_feats; [bs, emb1];  [bs, neighs, emb1];
    O: output [bs, emb2]
    """
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def bulid(self, self_vecs, neigh_vecs):
        # [bs, emb];  [bs, neighs, emb];
        self_vecs = tf.expand_dims(self_vecs, axis=1)  # [bs, 1, emb];
        means = tf.reduce_mean(tf.concat([neigh_vecs, self_vecs], axis=1), axis=1)  # [bs, neighs+1, emb]--> [bs, emb]
        emb_size = means.get_shape().as_list()[-1]
        output = tf.layers.dense(means, emb_size, activation='relu')
        return output  # [bs, emb]