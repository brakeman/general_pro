import pandas as pd
import numpy as np

#ROC曲线
def Roc_Curve(y_true,y_pre,have_auc=False):
    '''df as gbie p'''
    from sklearn import metrics
    fpr_lst,tpr_lst=metrics.roc_curve(y_true,y_pre)[:2]
    auc=metrics.auc(fpr_lst,tpr_lst)
    roc_curve=pd.DataFrame(np.array([tpr_lst,fpr_lst]).T,columns=["Sensitivity","1-Specificity"])
    roc_curve=roc_curve.drop_duplicates("1-Specificity")
    roc_curve.plot('1-Specificity','Sensitivity',\
                   title="ROC Curve \n AUC=%s" % str(auc))
    if have_auc:
        return auc,roc_curve

#KS曲线
def PlotKS(preds, labels, n, asc):
    
    # preds is score: asc=1
    # preds is prob: asc=0
    import matplotlib.pyplot as plt
    pred = preds  # 预测值
    bad = labels  # 取1为bad, 0为good
    ksds = pd.DataFrame({'bad': bad, 'pred': pred})
    ksds['good'] = 1 - ksds.bad
    
    if asc == 1:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    elif asc == 0:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0*ksds1.good.cumsum()/sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0*ksds1.bad.cumsum()/sum(ksds1.bad)
    
    if asc == 1:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    elif asc == 0:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0*ksds2.good.cumsum()/sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0*ksds2.bad.cumsum()/sum(ksds2.bad)
    
    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2'])/2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2'])/2
    
    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0*ksds['tile0']/len(ksds['tile0'])
    
    qe = list(np.arange(0, 1, 1.0/n))
    qe.append(1)
    qe = qe[1:]
    
    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q = qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)
    
    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds, columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])
    
    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    print ('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(np.round(ks_pop, 4)))
    
    # chart
    plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good',
                         color='blue', linestyle='-', linewidth=2)
                         
    plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad',
                        color='red', linestyle='-', linewidth=2)
                        
    plt.plot(ksds.tile, ksds.ks, label='ks',
                   color='green', linestyle='-', linewidth=2)
                       
    plt.axvline(ks_pop, color='gray', linestyle='--')
    plt.axhline(ks_value, color='green', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'], color='blue', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(),'cumsum_bad'], color='red', linestyle='--')
    plt.title('KS=%s ' %np.round(ks_value, 4) +  
                'at Pop=%s' %np.round(ks_pop, 4), fontsize=15)
    return ksds

#PSI计算
def Psi(y_future, y_expect, n=10,bins=None):
    '''<0.1 :well ;0.1-0.25 : normal;>0.25:terrible'''
    if bins is not None:
        cut_list=bins
    else:
        cut_list=sorted(list(set(np.percentile(y_expect,np.arange(0,110,10)))))
    tmp_lst1=pd.cut(y_future,cut_list,right=False,precision=7).\
    value_counts().sort_index()
    tmp_lst2=pd.cut(y_expect,cut_list,right=False,precision=7).\
    value_counts().sort_index()
    tmp_lst1=tmp_lst1/tmp_lst1.sum()
    tmp_lst2=tmp_lst2/tmp_lst2.sum()
    return tmp_lst1,tmp_lst2,((tmp_lst1-tmp_lst2)*np.log(tmp_lst1/tmp_lst2)).sum()


def _auc_impo(tra_x, tra_y, val_x, val_y, params=None, verbose=1):
    import lightgbm as lgb
    print(' train:{}   valid:{}'.format(tra_x.shape, val_x.shape))
    if params is None:
        params = {'task': 'train',
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric': 'auc',
              'num_threads': 10,
              'num_leaves': 3,  # 31,
              'learning_rate': 0.008,  # 0.002
              'feature_fraction': 0.5,
              'lambda_l2': 140,
              'bagging_fraction': 0.5,
              'bagging_freq': 5}

    cv_train = lgb.Dataset(tra_x, tra_y.astype('int'))        
    cv_valid = lgb.Dataset(val_x, val_y.astype('int'))        
    gbm = lgb.train(params,       # 参数字典
                    cv_train,       # 训练集
                    num_boost_round=2000,       # 迭代次数
                    valid_sets=cv_valid,        # 验证集
                    early_stopping_rounds = 100,
                    verbose_eval=verbose)
#     print('with best auc:{}'.format(gbm.best_score['valid_0']))
    return gbm


if __name__ == '__main__':
    data_path = './data'
    # test.csv  train.csv  train_target.csv
    tra_x = pd.read_csv(data_path + '/train.csv')
    tra_y = pd.read_csv(data_path + '/train_target.csv')
    final = tra_x.merge(tra_y,on='id')
    final['dist']= final.dist.apply(lambda x: int(x/100))
    random.seed(1)
    tra_id = set(random.sample(range(final.shape[0]),70000))
    val_id = set(range(final.shape[0])) - tra_id
    tra_id = [i for i in tra_id]
    val_id = [i for i in val_id]
    Train = final.iloc[tra_id,:]
    Valid = final.iloc[val_id,:]
    tra_x, tra_y = Train.drop('target', axis=1), Train.target
    val_x, val_y = Valid.drop('target', axis=1), Valid.target

    gbm = _auc_impo(tra_x, tra_y, val_x, val_y)
    pred = gbm.predict(val_x)
    Roc_Curve(val_y, pred, True)
    KS(val_y, pred)
    expect = gbm.predict(tra_x)
    a = Psi(y_future=pred, y_expect=expect)