# 计算损失和计算auc 的输入不一致；
from sklearn.metrics import roc_auc_score

def roc_auc_compute(y_targets, y_preds):
    '''
    :param y_targets: torch tensor [bs,1] or [bs];
    :param y_preds: torch tensor  [bs,1] or [bs];
    :return:auc_score;
    '''
    y_pred = y_preds.numpy().flatten()
    y_true = y_targets.numpy().flatten()
    return roc_auc_score(y_true, y_pred)