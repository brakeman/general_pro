import numpy as np
np.random.seed(123)


class EdgeMinibatchIter(object):
    def __init__(self, id2idx, adj, context_pairs, batch_size, max_degree=25):
        self.id2idx = id2idx
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.adj = adj
        self.edges = context_pairs
        self.num_samples = len(self.edges)

    def end(self):
        return (self.batch_num + 1) * self.batch_size >= len(self.edges)

    def next_minibatch_feed_dict(self):
        print('--')
        print('end:{}'.format(self.batch_num * self.batch_size))
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_edges = self.edges[start_idx: end_idx]
        print('start_idx:{}, end_idx:{}'.format(start_idx, end_idx))
        return self.batch_feed_dict(batch_edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])
        return batch1, batch2, len(batch_edges)


class EdgeMinibatchIterSupervised(object):
    def __init__(self, id2idx, adj, context_pairs, batch_size, labels, max_degree=25):
        self.id2idx = id2idx
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.adj = adj
        self.edges = context_pairs
        self.num_samples = len(self.edges)
        self.labels = labels

    def end(self):
        return (self.batch_num + 1) * self.batch_size >= len(self.edges)

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_edges = self.edges[start_idx: end_idx]
        batch_labels = self.labels[start_idx: end_idx]
        return self.batch_feed_dict(batch_edges, batch_labels)

    def batch_feed_dict(self, batch_edges, batch_labels):
        batch1 = []
        batch2 = []
        labels = []
        for edge, label in zip(batch_edges, batch_labels):
            node1, node2 = edge
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])
            labels.append(label)
        return batch1, batch2, labels, len(batch_edges)
