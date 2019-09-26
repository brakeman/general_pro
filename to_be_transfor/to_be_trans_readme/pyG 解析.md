# pyG 解析
核心是MessagePassing类
其中最关键的是 scatter_系函数，例如scatter_add(out, edge_index)  
### edge_index 
![pyg2](../pics/pyG_2.png)

	out = torch.index_select(node_Embedding_mat, edge_index[idx])  # index_select 可能会复制一个点多次；
	out = scatter_add(out, edge_index[i])  # 这里edge_index见图

![pyg1](../pics/pyG_1.png)
看这个例子，5个点，每个点N_dim=3;
对于0号位， 找不到，故0vec
对于1号位， 最后一个，故【2，1，3】
对于2号位，前俩，故【0，3，4】+【2，1，1】
没错啊，就是这个逻辑啊；
 我是target_node, 谁指向我，谁就是我的邻居，被我agg_scatter;


 ### 这样想：
 edge_index.Transopose 例如为 
 [0,1]
 [1,2]
 [3,2]
 [2,3]

 第一列代表src, 第二列代表target;
 左侧第一列经过 torch.index_select(node_Embedding_mat, edge_index[0]) 后
 从idx 变成 实际向量
 [emb_0]
 [emb_1]
 [emb_3]
 [emb_2]

 这个作为scatter_add 第一个参数；
 第二个参数是 edge_index[1] = 【1，2，2, 3】, 代表target

 那么 先看 
0号位： 在target中找不到，说明没有邻居指向它；
1号位： 在target中对应第一个，即它邻居是emb_0
2号位： 在target中对应第二行，第三行， 
3号位： 。。。
