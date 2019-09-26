# NCE  噪声对比估计

-------------------------
    def _nec_forward(self, nodes, neighs, negNeighs):
        '''unsupervised nce loss;
        :nodes: barch node ids [bs];
        :neighs: positive samples with regard to nodes [bs, K], K is num of neighs for each node;
        :negNeighs: negative samples with regrad to nodes [bs, K];'''
        bs = len(nodes)
        #bs, F, 1
        nodes = self.features(torch.LongTensor(nodes)).unsqueeze(-1)
        #bs, 1, F
        pos_nodes = torch.stack([self.features(torch.LongTensor(neighs[i])) for i in range(len(neighs))])
        #bs, 5, F
        neg_nodes = torch.stack([self.features(torch.LongTensor(negNeighs[i])) for i in range(len(negNeighs))])
        sum_log_neg = torch.bmm(neg_nodes, nodes).neg().sigmoid().log().squeeze().sum()
        sum_log_pos = torch.bmm(pos_nodes, nodes).sigmoid().log().squeeze().sum() 
        return (sum_log_pos + sum_log_neg)/bs

-----------------------
## 解析
核心是内积， target 与 正负样本 分别做内积，坍塌成一个scalar, 然后损失函数是 梯度下降，
希望 越小越好，等价于希望 pos 部分越小越好，希望neg部分越大越好（代码里用了负号）
- pos 部分越小，则2者内积越小，符合cosine 相似度 = a@b/(|a|*|b|) 在normlize之后，必然是内积等价于cosine 相似度； pos 越小，cos 越小， 方向上越接近（0度）；
- neg 部分越大，则2者内积越大， cos 越大，方向上越正交（90度）