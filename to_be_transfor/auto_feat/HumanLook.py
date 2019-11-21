import pandas as pd
from featexp import get_trend_stats
import matplotlib.pyplot as plt
import seaborn as sns


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



class Plot:
    def __init__(self, data, isTrain, isFraud):
        self.isTrain = isTrain
        self.isFraud = isFraud
        self.data = data
#         self.consit_df = self._trend_consistency(data[data.isTrain], data[~data.isTrain], self.isFraud)
        
    def plot_numerical_bylabel(self, data, col, target,bins,null_value, figsize=[6, 6]):
        plt.figure(figsize=figsize)
        # Plot the distribution for target == 0 and target == 1

        d1 = data.loc[data[target] == 0, col]
        d2 = data.loc[data[target] == 1, col]
        diff = self.difference(d1, d2, bins=bins, null_value=null_value)
        fig = sns.kdeplot(d1, label = '{} == 0'.format(target))
        sns.kdeplot(d2, label = '{} == 1'.format(target))
        consit_df = self._trend_consistency(data[data[self.isTrain]][[col, self.isFraud]],
                                 data[~data[self.isTrain]][[col, self.isFraud]],
                                 self.isFraud)
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.title('{} nunique:{}/{}\nqb_diff:{}\ntrain test consistency: {}\nKS check:{} '.format(col, data[col].nunique(),
                                                                                      data.shape[0], diff,
                                                                       consit_df.loc[col, 'Trend_correlation'],
                                                                       self.KS_check(data[data[self.isTrain]], 
                                                                                     data[~data[self.isTrain]],
                                                                                     col)))
        plt.legend(); 
        # fig.get_figure()
        return fig
    
    def diff_summary(self, data, cols, bins, null_value):
        df = pd.DataFrame(index = cols)
        label_diff = []
        tra_val_diff = []
        for col in cols:
            d1 = data.loc[data[self.isTrain] == 0, col]
            d2 = data.loc[data[self.isTrain] == 1, col]
            d3 = data.loc[data[self.isFraud] == 0, col]
            d4 = data.loc[data[self.isFraud] == 1, col]            
            diff_label = self.difference(d1, d2, bins=bins, null_value=null_value)
            diff_tra_val = self.difference(d3, d4, bins=bins, null_value=null_value)
            label_diff.append(diff_label)
            tra_val_diff.append(diff_tra_val)
        df['label_diff'] = label_diff
        df['tra_val_diff'] = tra_val_diff
        return df

    def plot_cols(self, data, col, bins, null_value):
        print('-'*30+' col:{} '.format(col)+'-'*30)
        p1 = self.plot_numerical_bylabel(data, col, target=self.isTrain,bins=bins,null_value=null_value)
        p2 = self.plot_numerical_bylabel(data, col, target=self.isFraud,bins=bins,null_value=null_value)
        
    def KS_check(self, tra, test, col):
        from scipy.stats import ks_2samp
        return ks_2samp(tra[col], test[col])
    
    def _trend_consistency(self, Train, Valid, target_col):
        return get_trend_stats(data=Train, target_col=target_col, data_test=Valid).set_index('Feature')

    def difference(self, d1, d2, bins, null_value):
        # null_value: null 不会被RankCut做任何处理；
        ''' 数值型变量看两个分布的相似度，用来看tra, val一致性，和 正负样本差异性 '''
        DF1, DF2 = pd.DataFrame(), pd.DataFrame()
        DF1['dist'] = d1
        DF2['dist'] = d2
        RC = RankCut_test.RankCut(cols=['dist'], bins=bins, null_value=null_value, return_numeric=True)
        RC.fit(DF1)
        a = RC.transform(DF1, train=True)
        b = RC.transform(DF2, train=False)

        CE1 = CountEnc(cols=a.columns.tolist())
        CE1.fit(a)
        aa = CE1.transform(a)
        bb = CE1.transform(b)

        d1_count= aa['Count(RC(dist))'].value_counts().sort_index()
        d2_count = bb['Count(RC(dist))'].value_counts().sort_index()
        d1_dist = d1_count/d1_count.sum()
        d2_dist = d2_count/d2_count.sum()

        DF=pd.DataFrame()
        DF['d1_dist'] = d1_dist
        DF['d2_dist'] = d2_dist
        DF = DF.fillna(0)
        return sum(abs(DF.d1_dist-DF.d2_dist))
    
    
if __name__ == '__main__':

    data_path = './data/'
    # test.csv  train.csv  train_target.csv
    tra_x = pd.read_csv(data_path + '/train.csv')
    tra_y = pd.read_csv(data_path + '/train_target.csv')
    tes_x = pd.read_csv(data_path + '/test.csv')
    final = tra_x.merge(tra_y,on='id')

    final['certValidStop'] = final.certValidStop.astype(int)
    final.fillna(-999,inplace=True)

    file = open('/data-0/qibo/pickle_files/cv_idx_dic.pickle', 'rb')
    idx_dic = pickle.load(file)
    tra_id, val_id = idx_dic['cv_0']['train_idx'], idx_dic['cv_0']['valid_idx']

    Train = final.iloc[tra_id,:].set_index(keys='id')
    Valid = final.iloc[val_id,:].set_index(keys='id')

    disc_vars = ['job', 'linkRela']
    
    X = Train.append(Valid)
    X['isTrain'] = X.index.isin(Train.index)
    pl = Plot(data=X, isTrain='isTrain', isFraud='target')
    pl.plot_cols(X, 'linkRela',bins=200, null_value=-999)
        