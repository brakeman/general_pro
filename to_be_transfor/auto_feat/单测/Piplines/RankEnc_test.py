from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class RankEnc(BaseEstimator, TransformerMixin):
    '''
    numerical columns to rank columns;
    only works for competitions since I diretctly append train, test together, while in normal situation it doesnt work;
    '''
    def __init__(self, cols, test): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.test = test

    def fit(self, x, y=None):
        if self.cols is None:
            self.cols = x.columns
        self.col_dics = {}
        X = x.append(self.test[self.cols])
        del x
        for col in self.cols:
            dic = {k:v for v,k in dict(enumerate(X[col].value_counts().sort_index().index)).items()}
            self.col_dics[col] = dic
        return self
    
    def transform(self, x):
        df = pd.DataFrame()
        for col in self.cols:
            if col not in self.col_dics:
                raise Exception('col:{} not in col_dics'.format(col))
            name_new = 'RankEnc({})'.format(col)
            df[name_new]= x[col].map(self.col_dics[col])
        return df


if __name__ == '__main__':
    data_path = './data'
    tra_x = pd.read_csv('../data/train.csv', encoding="utf-8").set_index('ID')
    test = pd.read_csv('../data/test.csv', encoding="utf-8").set_index('ID')
    Y = pd.read_csv('../data/train_label.csv', encoding="utf-8").set_index('ID')
    submission = pd.read_csv('../data/submission.csv')
    id_cols = ['企业类型', '登记机关', '行业代码', '行业门类', '企业类别', '管辖机关']
    RE = RankEnc(cols=id_cols, test=test)
    tra_rc = RE.fit_transform(tra_x)
    val_rc = RE.transform(test)
    print(val_rc.head())