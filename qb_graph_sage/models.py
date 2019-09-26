import tensorflow as tf
from qb_graph_sage.aggregator import MeanAggregator
from qb_graph_sage.sampler import LayerNeighborSampler
from qb_graph_sage.minibatch import EdgeMinibatchIter, EdgeMinibatchIterSupervised
from qb_graph_sage.utils import *
import pandas as pd


class GraphSage:
    def __init__(self, features, edges, adj_list, degrees, id_map, bs, neg_samples):
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
        self.adj_list = adj_list
        self.degrees = degrees  # a list of all nodes'degree; same order with id2nodes;
        self.sampler = LayerNeighborSampler(adj_list)
        self.agg = MeanAggregator()
        self.emb_size = features.shape[1]
        self.batch_size = bs
        self.neg_samples = neg_samples
        self.id2idx = id_map
        self.minibatch = EdgeMinibatchIter(id_map, adj_list, edges, batch_size=bs)

    def sample(self, inputs, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            layer_infos: [3, 2] # 第一层每个node采样3个， 第二层每个node采样2个
            inputs: batch nodes idx; # [bs]
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t]
            # samples[0] -- layers[1]
            # samples[1] -- layers[0]
            node = self.sampler.build((samples[k], layer_infos[t]))  # 从该layer nodes idx 关联到全局 adj_mat, 找到该层指定的邻居数；
            # 全域adj【num_nodes, max_sample_num], 如果不足，那么重复采样;
            # [layer_nodes, max_num_neighs];
            samples.append(tf.cast(tf.reshape(node, [support_size * batch_size]), tf.int32))  # emb_lookup 必须是int
            support_sizes.append(support_size)

        # samples: [bs*support_sizes[0], bs*support_sizes[1], bs*support_sizes[2], ...];
        # samples: [5个点， 10个点， 30个点];
        # layer_infos: [1, 2, 3];
        # support_sizes: [1, 2, 6];
        return samples, support_sizes

    def aggregate(self, samples, num_samples, support_sizes, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # [5个点的emb, 10个点的emb, 30个点的emb]
        hidden = [tf.nn.embedding_lookup(self.features, node_samples) for node_samples in samples]
        # [3, 2]
        for layer in range(len(num_samples)):
            next_hidden = []
            # layer = 0, hop = 0, self_feats = 5个点emb, neigh_feats = [5, 2, emb], h = 5个点emb ];
            #            hop = 1, self_feats = 10个点emb, neigh_feats = [10, 3, emb], h = 10个点emb];
            # 更新 hidden = next_hidden;
            # layer = 1, hop = 0, self_feats = 新点5个点emb; neigh_feats = [5, 2, emb], h = 新点5个点emb;
            for hop in range(len(num_samples) - layer):
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples) - hop - 1],
                              self.emb_size]
                self_feats = hidden[hop]
                neigh_feats = tf.reshape(hidden[hop + 1], neigh_dims)
                h = self.agg.bulid(self_feats, neigh_feats)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]  # [bs, new_emb];

    @staticmethod
    def nce_loss(inputs1, inputs2, neg_samples, neg_sample_weights):
        aff = tf.reduce_sum(inputs1 * inputs2, axis=1)
        neg_aff = tf.matmul(inputs1, tf.transpose(neg_samples))
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + neg_sample_weights * tf.reduce_sum(negative_xent)
        print('aff:{}\nneg_aff:{}\nloss:{}'.format(aff, neg_aff, loss))
        return loss

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])
        return batch1, batch2, len(batch_edges)

    @staticmethod
    def get_ph():
        inp1_ph = tf.placeholder(tf.int32, shape=(None))
        inp2_ph = tf.placeholder(tf.int32, shape=(None))
        batch_size_ph = tf.placeholder(tf.int32)
        return inp1_ph, inp2_ph, batch_size_ph

    def main(self, settings=[(2, 0.01)], layer_infos=[3, 2], neg_sample_size=4):
        '''function for graph, training, eval
        :layer_infos: num_neighs for each layer;
        :setting: training settting (epoch, lr);
        '''
        RESULT = []
        for (num_epochs, lr) in settings:
            train_loss_list, test_loss_list = [], []
            # tf.reset_default_graph()
            gb_step = tf.Variable(initial_value=0, trainable=False)
            inputs1, inputs2, batch_size = self.get_ph()  # [bs], [bs];
            samples1, support_sizes1 = self.sample(inputs1,
                                                   layer_infos)  # [bs=5个点， 10个点， 30个点], [1, 2, 6] when layer_infos = [3, 2];
            samples2, support_sizes2 = self.sample(inputs2,
                                                   layer_infos)  # [bs=5个点， 10个点， 30个点], [1, 2, 6] when layer_infos = [3, 2];

            outputs1 = self.aggregate(samples1, [self.features], layer_infos, support_sizes1)  # [bs, emb]
            outputs2 = self.aggregate(samples2, [self.features], layer_infos, support_sizes2)  # [bs, emb]

            labels = tf.reshape(tf.cast(inputs2, dtype=tf.int64), [self.batch_size, 1])

            neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels, num_true=1,
                num_sampled=neg_sample_size,
                unique=False,
                range_max=len(self.degrees), distortion=0.75,
                unigrams=list(self.degrees.values())))

            neg_samples = tf.cast(neg_samples, tf.int32)
            neg_samples, neg_support_sizes = self.sample(neg_samples, layer_infos, batch_size=neg_sample_size)
            neg_outputs = self.aggregate(neg_samples, [self.features], layer_infos, neg_support_sizes,
                                         batch_size=neg_sample_size)
            loss = self.nce_loss(outputs1, outputs2, neg_outputs, 1)
            optimizer = tf.contrib.layers.optimize_loss(loss=loss, learning_rate=lr, optimizer='Adam',
                                                        global_step=gb_step)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                epoch = 0
                while epoch < num_epochs:
                    epoch += 1
                    print('''epoch_{} start, end:{}?'''.format(epoch - 1, self.minibatch.end()))
                    batch_idx = 0
                    while not self.minibatch.end():
                        batch_idx += 1
                        inp1, inp2, temp_bs = self.minibatch.next_minibatch_feed_dict()
                        sess.run(optimizer, feed_dict={inputs1: inp1, inputs2: inp2, batch_size: temp_bs})
                        if batch_idx % 50 == 0:
                            print('batch_idx_for_training:{}'.format(batch_idx))


class GraphSage_supervised:
    def __init__(self, features, edges, adj_list, labels, degrees, id_map, bs, neg_samples):
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
        self.adj_list = adj_list
        self.degrees = degrees  # a list of all nodes'degree; same order with id2nodes;
        self.sampler = LayerNeighborSampler(adj_list)
        self.agg = MeanAggregator()
        self.emb_size = features.shape[1]
        self.batch_size = bs
        self.neg_samples = neg_samples
        self.id2idx = id_map
        self.edges = edges
        self.labels = labels

    #         self.minibatch = EdgeMinibatchIterSupervised(id_map, adj_list, edges, batch_size=bs, labels)

    def sample(self, inputs, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            layer_infos: [3, 2] # 第一层每个node采样3个， 第二层每个node采样2个
            inputs: batch nodes idx; # [bs]
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t]
            # samples[0] -- layers[1]
            # samples[1] -- layers[0]
            node = self.sampler.build((samples[k], layer_infos[t]))  # 从该layer nodes idx 关联到全局 adj_mat, 找到该层指定的邻居数；
            # 全域adj【num_nodes, max_sample_num], 如果不足，那么重复采样;
            # [layer_nodes, max_num_neighs];
            samples.append(tf.cast(tf.reshape(node, [support_size * batch_size]), tf.int32))  # emb_lookup 必须是int
            support_sizes.append(support_size)

        # samples: [bs*support_sizes[0], bs*support_sizes[1], bs*support_sizes[2], ...];
        # samples: [5个点， 10个点， 30个点];
        # layer_infos: [1, 2, 3];
        # support_sizes: [1, 2, 6];
        return samples, support_sizes

    def aggregate(self, samples, input_features, num_samples, support_sizes, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # [5个点的emb, 10个点的emb, 30个点的emb]
        hidden = [tf.nn.embedding_lookup(self.features, node_samples) for node_samples in samples]
        # [3, 2]
        for layer in range(len(num_samples)):
            next_hidden = []
            # layer = 0, hop = 0, self_feats = 5个点emb, neigh_feats = [5, 2, emb], h = 5个点emb ];
            #            hop = 1, self_feats = 10个点emb, neigh_feats = [10, 3, emb], h = 10个点emb];
            # 更新 hidden = next_hidden;
            # layer = 1, hop = 0, self_feats = 新点5个点emb; neigh_feats = [5, 2, emb], h = 新点5个点emb;
            for hop in range(len(num_samples) - layer):
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples) - hop - 1],
                              self.emb_size]
                self_feats = hidden[hop]
                neigh_feats = tf.reshape(hidden[hop + 1], neigh_dims)
                h = self.agg.bulid(self_feats, neigh_feats)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]  # [bs, new_emb];

    def clf_loss(self, outputs2, labels):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs2, labels=labels))

    def nce_loss(self, inputs1, inputs2, neg_samples, neg_sample_weights):
        aff = tf.reduce_sum(inputs1 * inputs2, axis=1)
        neg_aff = tf.matmul(inputs1, tf.transpose(neg_samples))
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + neg_sample_weights * tf.reduce_sum(negative_xent)
        print('aff:{}\nneg_aff:{}\nloss:{}'.format(aff, neg_aff, loss))
        return loss

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])
        return batch1, batch2, len(batch_edges)

    def get_ph(self):
        inp1_ph = tf.placeholder(tf.int32, shape=(None))
        inp2_ph = tf.placeholder(tf.int32, shape=(None))
        label_ph = tf.placeholder(tf.float32, shape=(None))
        batch_size_ph = tf.placeholder(tf.int32)
        return inp1_ph, inp2_ph, label_ph, batch_size_ph

    def main(self, settings=[(2, 0.01)], layer_infos=[3, 2], neg_sample_size=4):
        '''function for graph, training, eval
        :layer_infos: num_neighs for each layer;
        :setting: training settting (epoch, lr);
        '''
        RESULT = []
        for (num_epochs, lr) in settings:
            train_loss_list, test_loss_list = [], []
            # tf.reset_default_graph()
            gb_step = tf.Variable(initial_value=0, trainable=False)
            inputs1, inputs2, labels, batch_size = self.get_ph()  # [bs], [bs];
            samples1, support_sizes1 = self.sample(inputs1,
                                                   layer_infos)  # [bs=5个点， 10个点， 30个点], [1, 2, 6] when layer_infos = [3, 2];
            samples2, support_sizes2 = self.sample(inputs2,
                                                   layer_infos)  # [bs=5个点， 10个点， 30个点], [1, 2, 6] when layer_infos = [3, 2];

            outputs1 = self.aggregate(samples1, [self.features], layer_infos, support_sizes1)  # [bs, emb]
            outputs2 = self.aggregate(samples2, [self.features], layer_infos, support_sizes2)  # [bs, emb]

            tmp = tf.reshape(tf.cast(inputs2, dtype=tf.int64), [self.batch_size, 1])
            neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=tmp, num_true=1,
                num_sampled=neg_sample_size,
                unique=False,
                range_max=len(self.degrees), distortion=0.75,
                unigrams=list(self.degrees.values())))

            neg_samples = tf.cast(neg_samples, tf.int32)
            neg_samples, neg_support_sizes = self.sample(neg_samples, layer_infos, batch_size=neg_sample_size)
            neg_outputs = self.aggregate(neg_samples, [self.features], layer_infos, neg_support_sizes,
                                         batch_size=neg_sample_size)
            outputs2 = tf.layers.dense(outputs2, units=1)
            clf_loss = self.clf_loss(outputs2, labels)
            loss = self.nce_loss(outputs1, outputs2, neg_outputs, 1)
            optimizer = tf.contrib.layers.optimize_loss(loss=clf_loss, learning_rate=lr, optimizer='Adam',
                                                        global_step=gb_step)

            ########## graph finish, training start############
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                epoch = 0
                while epoch < num_epochs:
                    epoch += 1
                    batch_idx = 0
                    minibatch = EdgeMinibatchIterSupervised(self.id2idx, self.adj_list, self.edges, self.batch_size,
                                                            self.labels)
                    while not minibatch.end():
                        batch_idx += 1
                        inp1, inp2, batch_label, temp_bs = minibatch.next_minibatch_feed_dict()
                        sess.run(optimizer,
                                 feed_dict={inputs1: inp1, inputs2: inp2, labels: batch_label, batch_size: temp_bs})
                        if batch_idx % 50 == 0:
                            print('batch_idx_for_training:{}'.format(batch_idx))


if __name__ == '__main__':
    data = './data/col_4.csv'
    df = pd.read_csv(data, header=None)
    G, adj_list, userid2idx, degree, edges = make_adj_list(df.iloc[:, :2].values)
    fake_feats, fake_lables = make_fake_feat(G.nodes(), 50)
