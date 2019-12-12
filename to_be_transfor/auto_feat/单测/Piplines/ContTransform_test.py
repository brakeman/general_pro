from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class ContTransform(BaseEstimator, TransformerMixin):
    '''
    '''
    def __init__(self, cols, op_list=['log', 'square', 'sqrt', 'recip']):  # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}
        self.op_list = op_list

    def fit(self, x, y=None):
        if self.cols is None:
            self.cols = x.columns
        return self

    def transform(self, x):
        DF = pd.DataFrame(index=x.index)
        for col in self.cols:
            for op in self.op_list:
                if op =='log':
                    new_col_name = 'Log(' + col + ')'
                    DF[new_col_name] = x[col].map(np.log)
                elif op == 'square':
                    new_col_name = 'Square(' + col + ')'
                    DF[new_col_name] = x[col].map(np.square)
                elif op == 'sqrt':
                    new_col_name = 'Sqrt(' + col + ')'
                    DF[new_col_name] = x[col].map(np.sqrt)
                elif op == 'recip':
                    new_col_name = 'Recip(' + col + ')'
                    DF[new_col_name] = 1/x[col]
                else:
                    raise Exception('op [{}] not designed'.format(op))
        return DF


if __name__ == '__main__':
    tra_x = pd.read_csv('../data/train.csv', encoding="utf-8").set_index('ID')
    test = pd.read_csv('../data/test.csv', encoding="utf-8").set_index('ID')
    Y = pd.read_csv('../data/train_label.csv', encoding="utf-8").set_index('ID')
    submission = pd.read_csv('../data/submission.csv')
    id_cols = ['企业类型', '登记机关', '行业代码', '行业门类', '企业类别', '管辖机关']
    CT = ContTransform(cols=id_cols)
    tra_rc = CT.fit_transform(tra_x)
    val_rc = CT.transform(test)
    print(val_rc.head())