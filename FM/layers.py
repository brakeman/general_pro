import torch
import torch.nn as nn
from torch.nn import functional as F


class WideDeep(nn.Module):

    def __init__(self, num_uniq_leaf, num_trees, dim_leaf_emb):
        super(WideDeep, self).__init__()
        self.Emb = nn.Embedding(num_uniq_leaf, dim_leaf_emb)
        self.deep1 = nn.Linear(num_trees * dim_leaf_emb, num_trees * dim_leaf_emb // 2)
        self.deep2 = nn.Linear(num_trees * dim_leaf_emb // 2, num_trees * dim_leaf_emb // 4)
        self.deep3 = nn.Linear(num_trees * dim_leaf_emb // 4, num_trees * dim_leaf_emb // 8)
        self.ffn = nn.Linear(num_trees * dim_leaf_emb // 8 + num_trees * dim_leaf_emb, 1)

        assert num_trees * dim_leaf_emb // 8 >= 2

    def forward(self, x):
        'x: bs, num_trees'
        x = self.Emb(x)  # bs, num_trees, F
        bs, _, _ = x.size()
        x0 = x.view(bs, -1)  # bs, num_trees*F
        x = F.relu(self.deep1(x0))
        x = F.relu(self.deep2(x))
        x = F.relu(self.deep3(x))
        concat = torch.cat([x, x0], dim=-1)
        logits = F.sigmoid(self.ffn(concat))
        return logits


class DeepFM(nn.Module):

    def __init__(self, num_uniq_leaf, num_trees, dim_leaf_emb):
        super(DeepFM, self).__init__()
        self.num_trees = num_trees
        self.Emb = nn.Embedding(num_uniq_leaf, dim_leaf_emb)
        self.deep1 = nn.Linear(dim_leaf_emb, dim_leaf_emb // 2)
        self.deep2 = nn.Linear(dim_leaf_emb // 2, dim_leaf_emb // 4)
        #         in_dim = (dim_leaf_emb//4)*num_trees +
        # self.deep3 = nn.Linear(num_trees * dim_leaf_emb // 4, num_trees * dim_leaf_emb // 8)
        self.ffn = nn.Linear(532983, 1)  # to do;
        assert num_trees * dim_leaf_emb // 4 >= 2

    def _get_tri_idx(self, rank):
        assert rank >= 2
        aa = torch.range(0, rank * rank - 1).view(rank, rank).type(torch.long)
        bb = torch.triu(aa, 1)
        return torch.unique(bb)[1:]

    def _get_batch_idx(self, bs, rank):
        b = self._get_tri_idx(rank)
        return torch.cat([b + i for i in range(0, rank * rank * bs, rank ** 2)], dim=0)

    def forward(self, x):
        'x: bs, num_trees'
        x = self.Emb(x)  # bs, num_trees, F
        att_mat = torch.einsum('btf,byf->bty', x, x)
        bs, ts, _ = x.size()
        fm_part = torch.take(att_mat, self._get_batch_idx(bs, ts)).view(bs, -1)  # [bs, F2]
        deep_part = F.relu(self.deep1(x))  # [bs, new_trees, F_]
        deep_part = F.relu(self.deep2(deep_part)).view(bs, -1)  # [bs, new_trees, F_]
        concat = torch.cat([deep_part, fm_part], -1)
        logits = F.sigmoid(self.ffn(concat))
        return logits


class NFM(nn.Module):


    def __init__(self, num_uniq_leaf, dim_leaf_emb):
        super(NFM, self).__init__()
        self.Emb = nn.Embedding(num_uniq_leaf, dim_leaf_emb)
        self.deep1 = nn.Linear(dim_leaf_emb, dim_leaf_emb // 2)
        self.deep2 = nn.Linear(dim_leaf_emb // 2, dim_leaf_emb // 4)
        # self.deep3 = nn.Linear(num_trees * dim_leaf_emb // 4, num_trees * dim_leaf_emb // 8)
        self.ffn = nn.Linear(dim_leaf_emb//4, 1)
        assert dim_leaf_emb // 4 >= 2

    def _BIP(self, z):
        assert len(z.size()) == 3  # [bs, ts, F]
        sum_square = z.sum(dim=1)**2  # [bs, F]
        square_sum = z**2
        square_sum = square_sum.sum(dim=1)  # [bs, F]
        return (sum_square - square_sum) / 2  # [bs, F]

    def forward(self, x):
        'x: bs, num_trees'
        x = self.Emb(x)  # bs, num_trees, F
        x = self._BIP(x)  # bs, F
        x = F.relu(self.deep1(x))
        x = F.relu(self.deep2(x))
        logits = F.sigmoid(self.ffn(x))
        return logits


class DeepCross(nn.Module):


    def __init__(self, num_uniq_leaf, dim_leaf_emb):
        super(DeepCross, self).__init__()
        self.Emb = nn.Embedding(num_uniq_leaf, dim_leaf_emb)
        self.deep1 = nn.Linear(dim_leaf_emb, dim_leaf_emb // 2)
        self.deep2 = nn.Linear(dim_leaf_emb // 2, dim_leaf_emb // 4)
        # self.deep3 = nn.Linear(num_trees * dim_leaf_emb // 4, num_trees * dim_leaf_emb // 8)
        self.ffn = nn.Linear(dim_leaf_emb // 4, 1)
        assert dim_leaf_emb // 4 >= 2

    def _BIP(self, z):
        assert len(z.size()) == 3  # [bs, ts, F]
        sum_square = z.sum(dim=1) ** 2  # [bs, F]
        square_sum = z ** 2
        square_sum = square_sum.sum(dim=1)  # [bs, F]
        return (sum_square - square_sum) / 2  # [bs, F]

    def forward(self, x):
        'x: bs, num_trees'
        x = self.Emb(x)  # bs, num_trees, F
        x = self._BIP(x)  # bs, F
        x = F.relu(self.deep1(x))
        x = F.relu(self.deep2(x))
        logits = F.sigmoid(self.ffn(x))
        return logits