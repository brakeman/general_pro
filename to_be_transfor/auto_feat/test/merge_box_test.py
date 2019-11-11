import numpy as np
import pandas as pd

# 隐患是tra, val 可能保留不同的 dummy
# 方案是val 中出现了tra 中没出现的就直接-888
def merge_box(tra, val, columns, thresh):
    
    '''
    group merge big cats columns
    tra: train df
    val: valid df
    columns: list(str)
    thresh: 尾部类别被忽略为-888
    '''
    tra_df, val_df = pd.DataFrame(index=tra.index), pd.DataFrame(index=val.index)
    for col in columns:
        name = 'mb('+col+')'
        tra_tmp, val_tmp = _merge_box(tra, val, col, thresh)
        tra_df[name], val_df[name] = tra_tmp, val_tmp
    return tra_df, val_df

def _merge_box(tra, val, column, thresh):
    '''
    merge a single column with big cats;
    tra: train df
    val: valid df
    column: str
    thresh: 尾部类别被忽略为-888
    '''
    assert isinstance(tra, pd.DataFrame) and isinstance(val, pd.DataFrame)
    tra_df, val_df = tra[column].copy(), val[column].copy()
    tmp = tra_df.value_counts(normalize=True).cumsum()
    drop_cats = tmp[np.greater(tmp, thresh)].index.tolist()
    if len(drop_cats)==len(tra[column].unique()):
        drop_cats = tmp[np.greater(tmp, thresh)].index[1:]
    # 隐患是tra, val 可能保留不同的 dummy
    drop_tes = list((set(val_df.unique()) - set(tra_df.unique())).union(set(drop_cats)))
    tra_df[tra_df.isin(drop_cats)] = -888
    val_df[val_df.isin(drop_tes)] = -888
    return tra_df, val_df


if __name__ == '__main__':
    a1 = pd.DataFrame()
    a1['h1'] = ['a', 'b', 'c']
    a1['h2'] = ['1', '2', '3']

    a2 = pd.DataFrame()
    a2['h1'] = ['a', 'b', 'c1']
    a2['h2'] = ['1', '2', '31']

    a, b =merge_box(a1, a2, ['h1', 'h2'], 1)