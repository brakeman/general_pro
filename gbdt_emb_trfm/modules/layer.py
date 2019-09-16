# 1. 添加了MultiHeadAttentionSmall 类 '不用dense的方式去制造qs, ks, vs'
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List


class MultiHeadAttention(nn.Module):
    'dense 的方式制造qs, ks, vs'

    def __init__(self, num_heads, dim_q_k, dim_v, dim_m, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_q_k = dim_q_k  # Q,K 公用一个DENSE维度，因为要在这个轴上做att, 才能构造【ts, ts】的矩阵；
        self.dim_v = dim_v  # V 没什么约束；
        self.d_m = dim_m  # 这个一般要和输入维度保持一致，可以多层叠；
        self.dropout = dropout

        self.to_qs = nn.Linear(dim_m, num_heads * dim_q_k)
        self.to_ks = nn.Linear(dim_m, num_heads * dim_q_k)
        self.to_vs = nn.Linear(dim_m, num_heads * dim_v)
        self.ffn = nn.Linear(num_heads * dim_v, dim_m)
        self.dropoutLayer = nn.Dropout(p=dropout)

    def forward(self, q, k, v):
        bs, ts, _ = q.size()
        qs = F.relu(self.to_qs(q)).view(bs, ts, self.num_heads, self.dim_q_k)
        ks = F.relu(self.to_ks(k)).view(bs, ts, self.num_heads, self.dim_q_k)
        vs = F.relu(self.to_vs(v)).view(bs, ts, self.num_heads, self.dim_v)
        att_mat = torch.einsum('bthf,byhf->bhty', qs, ks)  # qs: [bs ts head, F], ks:[bs, ts2(即h), head, F]
        att_mat = F.softmax(att_mat / (self.dim_q_k ** 0.5), dim=-1)
        output = torch.einsum('bhty,byhf->bhtf', att_mat, vs)  # [bs, head, ts, F]
        output = output.view(bs, ts, -1)  # [bs, ts, head*dim_v]
        return self.dropoutLayer(F.relu(self.ffn(output)))  # [bs, ts, F]


class MultiHeadAttentionSmall(nn.Module):
    '不用dense的方式去制造qs, ks, vs'

    def __init__(self, num_heads, dim_m, dropout):
        super(MultiHeadAttentionSmall, self).__init__()
        assert dim_m % num_heads == 0
        self.num_heads = num_heads
        self.dropout = dropout
        self.ffn = nn.Linear(dim_m, dim_m)
        self.dropoutLayer = nn.Dropout(p=dropout)

    def forward(self, q, k, v):
        bs, ts, fq = q.size()
        _, _, fk = k.size()
        _, _, fv = v.size()
        assert fq % self.num_heads == 0
        assert fk % self.num_heads == 0
        assert fv % self.num_heads == 0
        qs = F.relu(q.view(bs, ts, self.num_heads, fq // self.num_heads))
        ks = F.relu(k.view(bs, ts, self.num_heads, fk // self.num_heads))
        vs = F.relu(v.view(bs, ts, self.num_heads, fv // self.num_heads))
        att_mat = torch.einsum('bthf,byhf->bhty', qs, ks)  # qs: [bs ts head, F], ks:[bs, ts2(即h), head, F]
        att_mat = F.softmax(att_mat / ((fq // self.num_heads) ** 0.5), dim=-1)
        output = torch.einsum('bhty,byhf->bhtf', att_mat, vs)  # [bs, head, ts, F]
        output = output.view(bs, ts, -1)  # [bs, ts, head*dim_v]
        return self.dropoutLayer(F.relu(self.ffn(output)))  # [bs, ts, F]


class Encoder(nn.Module):

    def __init__(self, num_heads, dim_q_k, dim_v, dim_m, layers, dropout, small=False):
        super(Encoder, self).__init__()
        if small == False:
            self.multihead = MultiHeadAttention(num_heads, dim_q_k, dim_v, dim_m, dropout=dropout)
        else:
            self.multihead = MultiHeadAttentionSmall(num_heads, dim_m, dropout)
        self.layer_norm = nn.LayerNorm(dim_m)
        self.layers = layers

    def forward(self, encoder_inputs):
        init = encoder_inputs  # [bs, ts ,f]
        for _ in range(self.layers + 1):
            output = self.multihead(q=init, k=init, v=init)  # 【bs, ts, f】
            output = self.layer_norm(output)
            output = self.multihead(q=output, k=output, v=output)
            init = self.layer_norm(output + init)
        return init


class TRFM(nn.Module):

    def __init__(self, num_uniq_leaf: int, dim_leaf_emb: int, num_heads: int,
                 dim_q_k: int, dim_v: int, dim_m: int, layers: int, dropout: float):
        super(TRFM, self).__init__()
        # num_uniq_leaf = num_leaf_per_tree * num_trees + 1 ;
        # dim_leaf_emb: embbeding dim;
        # num_heads: trfm para;
        # dim_q_k: trfm para;
        # dim_v: trfm para;
        # dim_m: input & output tensor final dim length;
        assert dim_m == dim_leaf_emb
        self.Encoder = Encoder(num_heads, dim_q_k, dim_v, dim_m, layers, dropout)
        self.Emb = nn.Embedding(num_uniq_leaf, dim_leaf_emb)
        self.tobinary = nn.Linear(dim_m, 1)

    def forward(self, input_leaf_seq: List) -> float:
        # input_leaf_seq: bs, ts; 代表叶子序号，例如总共3000个树，每个3个叶子，那么总共9000个独特的叶子；
        # 样本即叶子序列with shape: [bs, 3000];
        enc_in = self.Emb(input_leaf_seq)  # [bs, 3000, emb_dim]
        enc_out = self.Encoder(enc_in)
        out = F.sigmoid(self.tobinary(enc_out[:, 0, :]))  # [bs, 1]
        return out