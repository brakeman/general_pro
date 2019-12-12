from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class LabelEnc(BaseEstimator, TransformerMixin):
    '''
    低于阈值的小类别都去掉变成nan;
    '''
    def __init__(self, cols, thresh=0.01): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}
        self.thresh = thresh
    
    def fit(self, x, y=None):
        if self.cols is None:
            self.cols = x.columns
        x_len = len(x)
        thresh_cnt = int(x_len*self.thresh)
        for col in self.cols:
            tmp_val_counts = x[col].value_counts()
            tmp_val_counts = tmp_val_counts[tmp_val_counts>thresh_cnt]
            self.col_dics[col] = {k:v for v,k in dict(enumerate(tmp_val_counts.index.tolist())).items()}
        return self
    
    def transform(self, x):
        DF = pd.DataFrame(index=x.index)
        for col in self.cols:
            new_col_name = 'LE('+col+')'
            DF[new_col_name] = x[col].map(self.col_dics[col])
        return DF


if __name__ == '__main__':
    data_path = './data'
    tra_x = pd.read_csv('../data/train.csv', encoding="utf-8").set_index('ID')
    test = pd.read_csv('../data/test.csv', encoding="utf-8").set_index('ID')
    Y = pd.read_csv('../data/train_label.csv', encoding="utf-8").set_index('ID')
    submission = pd.read_csv('../data/submission.csv')
    id_cols = ['企业类型', '登记机关', '行业代码', '行业门类', '企业类别', '管辖机关']
    LE = LabelEnc(cols=id_cols, thresh=0.02)
    tra_rc = LE.fit_transform(tra_x)
    val_rc = LE.transform(test)
    print(val_rc.head())