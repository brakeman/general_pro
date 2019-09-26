import networkx as nx
import numpy as np

def make_adj_list(path_array):
    '''
    para: path_array: df.values with shape:[nodes, 2]
    return: adj_list, userid2idx, node_degree
    '''
    G=nx.Graph()
    G.add_edges_from(path_array)
    adj_list = G.adjacency_list()
    userid2idx = {j:i for i,j in enumerate(G.nodes())}
    adj_list = construct_adj_list(G, userid2idx, max_degree=25)
    return G, adj_list, userid2idx, G.degree(), G.edges()

def make_fake_feat(nodes, emb_size):
    '''
    para: nodes: a list of user_id;
    return: fake fatures of all nodes;'''
    features = np.ones(shape = (len(nodes), emb_size))
    labels = np.ones(len(nodes))
    return features, labels

def construct_adj_list(G, id2idx, max_degree):
    adj_list = len(id2idx)*np.ones((len(id2idx)+1, max_degree))
    for nodeid in G.nodes():
        neighbors = np.array([id2idx[neighbor] for neighbor in G.neighbors(nodeid)])
        if len(neighbors) == 0:
            continue
        if len(neighbors) > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif len(neighbors) < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)
        adj_list[id2idx[nodeid], :] = neighbors
    return adj_list.astype(int)

