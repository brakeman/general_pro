import torch
import torch.nn as nn
import itertools
from torch.nn import functional as F
from torch.nn import Parameter, init


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
        # x: bs, num_trees
        x = self.Emb(x)  # bs, num_trees, F
        bs, _, _ = x.size()
        x0 = x.view(bs, -1)  # bs, num_trees*F
        x = F.relu(self.deep1(x0))
        x = F.relu(self.deep2(x))
        x = F.relu(self.deep3(x))
        concat = torch.cat([x, x0], dim=-1)
        logits = F.sigmoid(self.ffn(concat))
        return logits


# fm part: field 向量内积，3D->2D;
# deep part: filed 独立dense后concat; 3D-->3D--2D
class DeepFM(nn.Module):

    def __init__(self, num_uniq_leaf, num_trees, dim_leaf_emb, concat_wide):
        super(DeepFM, self).__init__()
        self.num_trees = num_trees
        self.Emb = nn.Embedding(num_uniq_leaf, dim_leaf_emb)
        self.deep1 = nn.Linear(dim_leaf_emb, dim_leaf_emb // 2)
        fm_part_dim = len(self._get_tri_idx(num_trees))  # ts'
        self.concat_wide = concat_wide
        if concat_wide:
            out_dim = fm_part_dim + num_trees * (dim_leaf_emb // 2) + num_trees * dim_leaf_emb
        else:
            out_dim = fm_part_dim + num_trees * (dim_leaf_emb // 2)
        self.ffn = nn.Linear(out_dim, 1)  # to do
        assert num_trees * dim_leaf_emb // 2 >= 2

    def _get_tri_idx(self, rank):
        assert rank >= 2
        aa = torch.range(0, rank * rank - 1).view(rank, rank).type(torch.long)
        bb = torch.triu(aa, 1)
        return torch.unique(bb)[1:]  # 0不算；

    def forward(self, x):
        # x: bs, num_trees
        x = self.Emb(x)  # bs, num_trees,
        #         print(x.shape)
        bs, ts, f = x.shape

        fm_part = torch.einsum('btf,byf->bty', x, x)
        #         print(x.shape)
        fm_part = torch.triu(fm_part, diagonal=0)  # bs,ts,ty
        #         print(fm_part.shape)
        select_idx = self._get_tri_idx(ts)
        #         print(select_idx.shape)
        fm_part = fm_part.view(bs, -1)[:, select_idx]  # bs, flatten
        #         print(fm_part.shape)
        deep_part = F.relu(self.deep1(x))  # [bs, ts, F]
        deep_part = deep_part.view(bs, -1)  # [bs, ts*F]
        #         print(deep_part.shape)
        if self.concat_wide:
            wide_part = x.view(bs, ts * f)  # bs, ts*F
            concat = torch.cat([deep_part, fm_part, wide_part], -1)
        else:
            concat = torch.cat([deep_part, fm_part], -1)  #
        logits = F.sigmoid(self.ffn(concat))
        return logits


# fm part: field 上三角向量元素积后 SUM pooling，3D->3D->2D;
class NFM(nn.Module):

    def __init__(self, num_uniq_leaf, num_trees, dim_leaf_emb, concat_wide):
        super(NFM, self).__init__()
        self.Emb = nn.Embedding(num_uniq_leaf, dim_leaf_emb)
        self.deep1 = nn.Linear(dim_leaf_emb, dim_leaf_emb // 2)
        self.deep2 = nn.Linear(dim_leaf_emb // 2, dim_leaf_emb // 4)
        self.concat_wide = concat_wide
        if concat_wide:
            self.ffn = nn.Linear(dim_leaf_emb // 4 + (num_trees * dim_leaf_emb), 1)
        else:
            self.ffn = nn.Linear(dim_leaf_emb // 4, 1)
        assert dim_leaf_emb // 4 >= 2

    def _bip(self, z):
        assert len(z.size()) == 3  # [bs, ts, F]
        sum_square = z.sum(dim=1) ** 2  # [bs, F]
        square_sum = z ** 2
        square_sum = square_sum.sum(dim=1)  # [bs, F]
        return (sum_square - square_sum) / 2  # [bs, F]

    def forward(self, x):
        # x: bs, num_trees
        x_emb = self.Emb(x)  # bs, num_trees, F
        x = self._bip(x_emb)  # bs, F
        x = F.relu(self.deep1(x))
        x = F.relu(self.deep2(x))
        bs, ts, f = x_emb.shape
        if self.concat_wide:
            x = torch.cat([x, x_emb.view(-1, ts * f)], dim=1)
        out = F.sigmoid(self.ffn(x))
        return out


# 递归part: emb reshape 后成为x_0, 然后每一层结果和自己元素乘 变成下一层；  3D-->2D-->2D-->...;
class DeepCross(nn.Module):

    def __init__(self, num_uniq_leaf, num_trees, dim_leaf_emb, num_layers, concat_wide):
        super(DeepCross, self).__init__()
        self.Emb = nn.Embedding(num_uniq_leaf, dim_leaf_emb)
        self.layer_paras = []
        for i in range(num_layers):
            weight = Parameter(torch.Tensor(dim_leaf_emb * num_trees, 1))
            init.xavier_uniform(weight)
            bias = Parameter(torch.Tensor(1))
            self.layer_paras.append((weight, bias))

        self.deep1 = nn.Linear(dim_leaf_emb * num_trees, dim_leaf_emb // 2)
        self.deep2 = nn.Linear(dim_leaf_emb // 2, dim_leaf_emb // 4)
        self.concat_wide = concat_wide
        if concat_wide:
            out_dim = dim_leaf_emb // 4 + num_trees * dim_leaf_emb
        else:
            out_dim = dim_leaf_emb // 4
        self.ffn = nn.Linear(out_dim, 1)
        assert dim_leaf_emb // 4 >= 2

    def _cross_layer(self, x0, x, w, b, bias):
        assert len(x.size()) == 2
        assert len(x0.size()) == 2
        if bias:
            xw = x @ w + b  # [bs, 1]
        else:
            xw = x @ w  # [bs, 1]
        return x0 * xw  # [bs, F]

    def forward(self, x):
        # x: bs, num_trees
        x = self.Emb(x)  # bs, num_trees, F
        bs, _, _ = x.size()
        x0 = x.view(bs, -1)  # bs, num_trees*F
        # cross & residual layer;
        temp_x = x0
        for layer, paras in enumerate(self.layer_paras):
            cross_x = self._cross_layer(x0, temp_x, paras[0], paras[1], True)  # [bs,F]
            if layer % 2 == 0:
                temp_x = temp_x + cross_x
            else:
                temp_x = cross_x

        x = F.relu(self.deep1(temp_x))
        x = F.relu(self.deep2(x))
        if self.concat_wide:
            x = torch.cat([x, x0], dim=1)
        out = F.sigmoid(self.ffn(x))
        return out


# 排列组合操作爆炸 itertools.combinations(idx, 2)
class AFM(nn.Module):

    def __init__(self, num_uniq_leaf, dim_leaf_emb):
        super(AFM, self).__init__()
        self.Emb = nn.Embedding(num_uniq_leaf, dim_leaf_emb)
        self.deep1 = nn.Linear(dim_leaf_emb, dim_leaf_emb // 2)
        self.deep2 = nn.Linear(dim_leaf_emb // 2, dim_leaf_emb // 4)
        att_dim = dim_leaf_emb // 2
        self.W = Parameter(torch.Tensor(dim_leaf_emb, att_dim))
        self.b = Parameter(torch.Tensor(att_dim))
        self.h = Parameter(torch.Tensor(att_dim))
        init.xavier_uniform(self.W)
        # self.deep3 = nn.Linear(num_trees * dim_leaf_emb // 4, num_trees * dim_leaf_emb // 8)
        self.ffn = nn.Linear(dim_leaf_emb // 4, 1)
        assert dim_leaf_emb // 4 >= 2

    def half_bip(self, x, axis):
        idx = range(0, x.shape[axis])
        row, col = zip(*itertools.combinations(idx, 2))  # ts+1)*ts/2
        return x[:, row] * x[:, col]

    # 还是不行，不是一个变量大，是由于ts' 超级大， 所以后面每一个变量都贼几把大。
    def forward(self, x):
        # x: bs, num_trees
        x = self.Emb(x)  # bs, num_trees, F
        half_bip = self.half_bip(x, axis=1)  # bs, ts', F
        x = F.relu(half_bip @ self.W + self.b)  # [bs, ts', F2]
        att = F.softmax(x @ self.h, dim=-1)  # [bs, ts']
        out = half_bip * att[:, :, None]  # [bs, ts', F]
        out = out.sum(dim=1)  # [bs, F]
        out = F.relu(self.deep1(out))  # bs, F2
        out = F.relu(self.deep2(out))
        logits = F.sigmoid(self.ffn(out))
        return logits




# 直接把立方体Zk 压缩成 num_filters 个点;
# kernel die;
class CinLayer(nn.Module):

    def __init__(self, num_filters, ts_0, ts_k):
        # ts_0 代表x_0时间长度， ts_k 代表第k 个layer的时间长度
        super(CinLayer, self).__init__()
        self.num_filters = num_filters
        self.conv = nn.Conv1d(1, num_filters, (ts_0, ts_k), stride=ts_k)

    def forward(self, x_0, x_k):
        # x_0: bs, ts_0, f;
        # x_k: bs, ts_k, f;
        _, ts_0, _ = x_0.shape
        bs, ts_k, f = x_k.shape
        print('start cin ... ')
        z = torch.einsum('btf,byf->bfty', x_0, x_k)  # [bs, f, ts, ty]
        print(z.shape)
        z = z.contiguous().view(bs, 1, ts_0, f * ts_k)  # [bs, 1, ts, ty*F] # to be check;
        z = self.conv(z).squeeze()
        return z  # [bs, num_filters, F]


class xDeepFM(nn.Module):

    def __init__(self, num_layers, layer_filters, num_uniq_leaf, num_trees, dim_leaf_emb, concat_wide):
        super(xDeepFM, self).__init__()
        self.Emb = nn.Embedding(num_uniq_leaf, dim_leaf_emb)

        if isinstance(layer_filters, list):
            assert num_layers == len(layer_filters)
        elif isinstance(layer_filters, int):
            layer_filters = [layer_filters] * num_layers
        else:
            raise Exception('layer_filters should be int or list of int obj')

        ts_k_list = [num_trees]
        ts_k_list.extend(layer_filters[:-1])  # 每一层最开始的ts_k，第一层即ts_0, 随后每一层由上一层的 filter_num 决定;
        print('ts_k_list:{}'.format(ts_k_list))
        self.layer_list = nn.ModuleList()
        for _, filters, ts_k in zip(range(num_layers), layer_filters, ts_k_list):
            self.layer_list.append(CinLayer(filters, ts_0=num_trees, ts_k=ts_k))

        self.concat_wide = concat_wide
        if concat_wide:
            out_dim = num_layers * dim_leaf_emb + num_trees * dim_leaf_emb
        else:
            out_dim = num_layers * dim_leaf_emb
        self.ffn = nn.Linear(out_dim, 1)  # 因为最后concat 的轴是 ts_k， ts_k 是每一层最后输出的新的ts轴，由每一层的filter_size 决定

    def forward(self, x):
        x0 = xk = self.Emb(x)  # bs, ts_0, F
        bs, _, _ = x0.shape

        layer_out = None
        for cin_model in self.layer_list:
            xk = cin_model(x0, xk)  # [bs, num_filters, F]
            if layer_out is None:
                layer_out = xk.sum(dim=1)  # bs, F
            else:
                layer_out = torch.cat([layer_out, xk.sum(dim=1)], dim=-1)
            print('layer_out shape:{}'.format(layer_out.shape))
        print('wide part shape:{}'.format(x0.view(bs, -1).shape))
        if self.concat_wide:
            print('x0 shape:{}'.format(x0.shape))
            layer_out = torch.cat([layer_out, x0.view(bs, -1)], dim=1)
        print('final shape:{}'.format(layer_out.shape))
        return F.sigmoid(self.ffn(layer_out))
