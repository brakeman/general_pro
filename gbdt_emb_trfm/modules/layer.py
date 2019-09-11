import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, dim_q_k, dim_v, dim_m, dropout):
        self.num_heads = num_heads,
        self.dim_q_k = dim_q_k,  # Q,K 公用一个DENSE维度，因为要在这个轴上做att, 才能构造【ts,ts】的矩阵；
        self.dim_v = dim_v,  # V 没什么约束；
        self.d_m = dim_m,  # 这个一般要和输入维度保持一致，可以多层叠；
        self.dropout = dropout

        self.to_qs = nn.Linear(dim_m, num_heads * dim_q_k)
        self.to_ks = nn.Linear(dim_m, num_heads * dim_q_k)
        self.to_vs = nn.Linear(dim_m, num_heads * dim_v)
        self.ffn = nn.Linear(num_heads * dim_v, dim_m)

    def forward(self, q, k, v):
        bs, ts, _ = q.size()

        qs = F.relu(self.to_qs(q)).view(bs, ts, self.num_heads, self.dim_q_k)
        ks = F.relu(self.to_ks(k)).view(bs, ts, self.num_heads, self.dim_q_k)
        vs = F.relu(self.to_vs(v)).view(bs, ts, self.num_heads, self.dim_v)
        att_mat = torch.einsum('bthf,byhf->bhty', qs, ks)  # qs: [bs ts head, F], ks:[bs, ts2(即h), head, F]
        att_mat = self.softmax(att_mat/(self.dim_q_k ** 0.5))
        output = torch.einsum('bhty,bhyf->bhtf', att_mat, vs)  # [bs, head, ts, F]
        output = output.view(bs, ts, -1)  # [bs, ts, head*dim_v]
        return self.dropout(F.relu(self.ffn(output)))  # [bs, ts, F]


class Encoder(nn.Module):

    def __init__(self, num_heads, dim_q_k, dim_v, dim_m, dropout):
        self.multihead = MultiHeadAttention(num_heads, dim_q_k, dim_v, dim_m, dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim_m)

    def forward(self, encoder_inputs, layers):
        init = encoder_inputs  # [bs, ts ,f]
        for _ in layers:
            output = self.multihead(q=init, k=init, v=init)  # 【bs, ts, f】
            init = self.layer_norm(output + init)
        return init
