# 看tra, val, 俩数据集分布一致性；
# 看离群样本空置率；

#     - HumanLook
#         -看数字
#             - 看列方差，列空置，列类别个数，列分布一致性
from featexp import get_trend_stats
import pandas as pd


def _col_unique(df):
    def unqiue_element(x):
        a=len(x.unique())
        return a
    return df.apply(unqiue_element)

def _col_var(df):
    return df.apply(lambda x: x.var())

def _col_null(df, null_val):
    def equal_to(x, null_val):
        a=sum(x==null_val)
        return a        
    return df.apply(equal_to, null_val=null_val)

def _trend_consistency(Train, Valid, target_col='target'):
    return get_trend_stats(data=Train, target_col=target_col, data_test=Valid)

def show_all(Train, Valid, target_col, null_val):
    res = {}
    res['null'] = _col_null(Train, null_val)
    res['unique'] = _col_unique(Train)
    res['var'] = _col_var(Train)
    res['trend_consistency'] = _trend_consistency(Train, Valid, target_col)
    return res