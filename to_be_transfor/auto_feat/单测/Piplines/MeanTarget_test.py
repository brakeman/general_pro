from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class MeanTarget(BaseEstimator, TransformerMixin):
    '''
    允许非scalar;
    不适合高基类，会在test生产大量None,以及过拟合
    rain_df.groupby([col])[TARGET].agg(['mean'])
    tobedone: k-fold
    '''

    def __init__(self, cols):  # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}

    def fit(self, x, y):
        self.col_dics = {}
        if self.cols is None:
            self.cols = x.columns
        df = x.copy()
        del x
        df['target'] = y
        for col in self.cols:
            keep = df.groupby(col)['target'].mean()
            self.col_dics[col] = keep
        return self

    def transform(self, x):
        df = pd.DataFrame()
        for col in self.cols:
            if col not in self.col_dics:
                raise Exception('col:{} not in col_dics'.format(col))
            col_dic = self.col_dics[col]
            new_name = 'MeanTarget(' + col + ')'
            df[new_name] = x[col].map(col_dic)
        return df

if __name__ == '__main__':
    tra_x = pd.read_csv('../data/train.csv', encoding="utf-8").set_index('ID')
    test = pd.read_csv('../data/test.csv', encoding="utf-8").set_index('ID')
    Y = pd.read_csv('../data/train_label.csv', encoding="utf-8").set_index('ID')
    submission = pd.read_csv('../data/submission.csv')
    # id_cols = ['城建税']
               # '登记机关', '行业代码', '行业门类', '企业类别', '管辖机关']
    id_cols2 = ['城建税','登记机关', '行业代码', '行业门类', '企业类别', '管辖机关']
    MT = MeanTarget(cols=id_cols2)
    tra_rc = MT.fit_transform(tra_x, Y)
    val_rc = MT.transform(test)
    print(val_rc.head(20))
