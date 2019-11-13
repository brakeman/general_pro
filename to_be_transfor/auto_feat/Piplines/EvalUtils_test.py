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
def KS(y_true,y_pre,group_sep=0.01):
    from copy import deepcopy
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    '''df as gbie p'''
    DfIn=pd.DataFrame([y_true,y_pre]).T
    DfIn.columns=['gbie','p']
    DfIn=DfIn.sort_values('p',ascending=False)
    length=len(DfIn)
    group=pd.cut(list(range(length)),bins=1.0/group_sep,labels=np.arange(0+group_sep,1+group_sep,group_sep))
    DfIn['group']=list(group)
    length_1=sum(DfIn.gbie)
    length_0=length-length_1
    cum_0=[]
    cum_1=[]
    for i in sorted(set(group)):
        cum_tmp=DfIn.loc[DfIn.group<=i,'gbie']
        cum_0.append(float(len(cum_tmp)-sum(cum_tmp))/length_0)
        cum_1.append(float(sum(cum_tmp))/length_1)
    DfPlot=pd.DataFrame(np.array([sorted(list(set(group))),cum_0,cum_1]).T, 
                        columns=['Percent','Good','Bad'])
    Ks=max(DfPlot.Bad-DfPlot.Good)
    argmax=DfPlot.loc[(DfPlot.Bad-DfPlot.Good).argmax(),'Percent']
    DfPlot['ks']=DfPlot.Bad-DfPlot.Good
    DfPlot.plot('Percent',['Good','Bad','ks'],\
                title='KS Curve\n ks=%.3f and cut_off=%.2f' % (Ks,argmax))
    return plt,Ks

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