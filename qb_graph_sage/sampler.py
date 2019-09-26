import tensorflow as tf
import copy


"""
Classes that are used to sample node neighborhoods
"""


class LayerNeighborSampler:
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info):
        super(LayerNeighborSampler, self).__init__()
        self.adj_info = adj_info

    def build(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)  # 从全局adj中查到该layer nodes 的邻居list;
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])
        return adj_lists


# ！做完draft再改；
def negtive_nodes_sample(adj_mat, inputs2, id2idx, degrees, num_neg_samples):
    batch_nodes_neighs_ids = tf.nn.embedding_lookup(adj_mat, inputs2)  # suppose inputs2 are target nodes;

    # 保证不会抽到 该batch 所有点任何邻居；
    degrees_tmp = copy.deepcopy(degrees.tolist())
    batch_nodes_neighs_ids = tf.constant(batch_nodes_neighs_ids)
    for id_ in batch_nodes_neighs_ids:  # 必须保证 id 是 idx;
        for node in id_:
            idx = id2idx(node)
            degrees_tmp[idx] = 0

    labels = tf.reshape(tf.cast(inputs2, dtype=tf.int64), [-1, 1])
    neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels,
        num_true=1,
        num_sampled=num_neg_samples,
        unique=True,
        range_max=len(degrees),
        distortion=0.75,
        unigrams=degrees_tmp))
    return neg_samples
