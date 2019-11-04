# 针对非时序表做特征工程，核心依赖 cat列之间的交互
# v0 半自动，一组cat列交互之后 看lgb auc + top k features
# v1 自动，给出全部cat列，有限资源下，按阶数做 lgb auc + top k features, 不需要再人工参与筛选了


import pandas as pd
import random

def to_bins(col, q):
    # 感觉 qcut 不是很好
    # 分箱前要先做 离群点分析；
    # 连续性变量离散化, 先随意搞一个
#         new_col = pd.qcut(col, q=q)
    new_col, bins = pd.cut(col, bins=q, retbins=True)
    return new_col, bins

def to_bins_trans(col, bins):
    # 感觉 qcut 不是很好
    # 分箱前要先做 离群点分析；
    # 连续性变量离散化, 先随意搞一个
#         new_col = pd.qcut(col, q=q)
    new_col, bins = pd.cut(col, bins=bins, retbins=True)
    return new_col, bins





    
# cols = [final['dist'], final['target']]
# col_names = ['dist', 'target']
def _auto_combine(cols, col_names, max_limit = 10000):
    # 多种类别形 直接 字符串相加 变成新列;
    # cols: List[series]
    # col_names: List[str]
    # return: pd.Series with name;
    name, val, unique_cat = '', '', 0
    for i in range(len(col_names)):
        if unique_cat > max_limit:
            break            
        name += col_names[i]+'|'
        val += col_names[i]+'='+cols[i].astype('str')+'|'
        unique_cat = len(val.value_counts())
        print('generated new column:{}\n.................with unique category length:{}\n'.format(name, unique_cat))
    return pd.Series(val, name=name)
# _auto_combine(cols, col_names)


