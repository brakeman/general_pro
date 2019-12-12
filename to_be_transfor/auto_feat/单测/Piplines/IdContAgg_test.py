from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class IdContAgg(BaseEstimator, TransformerMixin):
    '''
    '''

    def __init__(self, cols, cont_col, agg_types):  # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}
        self.cont_col = cont_col
        self.agg_types = agg_types

    def fit(self, x, y=None):
        for col in self.cols:
            self.col_dics[col] = {}
            for agg_type in self.agg_types:
                self.col_dics[col][agg_type] = x.groupby([col])[self.cont_col].agg([agg_type]).to_dict()
        return self

    def transform(self, x):
        DF = pd.DataFrame(index=x.index)
        for col in self.cols:
            for agg_type in self.agg_types:
                new_col_name = 'Agg(' + '_'.join([col, self.cont_col, agg_type]) + ')'
                DF[new_col_name] = x[col].map(self.col_dics[col][agg_type][agg_type])
                if agg_type == 'mean':
                    new_col = 'Agg(' + '_'.join([col, self.cont_col, agg_type]) + '_diff' + ')'
                    DF[new_col] = x[self.cont_col] - DF[new_col_name]
        return DF


if __name__ == '__main__':
    data_path = './data'
    tra_x = pd.read_csv('../data/train.csv', encoding="utf-8").set_index('ID')
    test = pd.read_csv('../data/test.csv', encoding="utf-8").set_index('ID')
    Y = pd.read_csv('../data/train_label.csv', encoding="utf-8").set_index('ID')
    submission = pd.read_csv('../data/submission.csv')
    id_cols = ['企业类型', '登记机关', '行业代码', '行业门类', '企业类别', '管辖机关']
    ICA = IdContAgg(cols=id_cols, cont_col='城建税', agg_types=['mean'])
    tra_rc = ICA.fit_transform(tra_x)
    val_rc = ICA.transform(test)
    print(val_rc.head())