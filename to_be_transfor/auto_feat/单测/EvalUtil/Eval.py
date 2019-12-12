import pandas as pd
import numpy as np
from Piplines import IndCountScore_test
from multiprocessing import Pool

def PlotKS(preds, labels, n, asc=0, plot=None):
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
    ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)

    if asc == 1:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    elif asc == 0:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum() / sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum() / sum(ksds2.bad)

    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2']) / 2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2']) / 2

    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q=qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds, columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    print('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(np.round(ks_pop, 4)))

    # chart
    if plot:
        plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good',
                 color='blue', linestyle='-', linewidth=2)

        plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad',
                 color='red', linestyle='-', linewidth=2)

        plt.plot(ksds.tile, ksds.ks, label='ks',
                 color='green', linestyle='-', linewidth=2)

        plt.axvline(ks_pop, color='gray', linestyle='--')
        plt.axhline(ks_value, color='green', linestyle='--')
        plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'], color='blue', linestyle='--')
        plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_bad'], color='red', linestyle='--')
        plt.title('KS=%s ' % np.round(ks_value, 4) +
                  'at Pop=%s' % np.round(ks_pop, 4), fontsize=15)
    return ksds, ks_pop

def score2eval(logits, Y, good_qbbad, bad_qbgood):
    Z = Y.copy()
    Z['pred'] = logits
    _, ks = PlotKS(Z['pred'], Y.values.flatten(), 20)
    Z['good_qbbad'] = Z.index.isin(good_qbbad)*1
    Z['bad_qbgood'] = Z.index.isin(bad_qbgood)*1
    Z['new_pred'] = (Z.pred >= ks) *1
    NewPred_GoodQbbad_Rate = Z[(Z.good_qbbad==1)&(Z.new_pred==0)].shape[0]/(Z[Z.good_qbbad==1].shape[0])
    NewPred_BadQbgood_Rate = Z[(Z.bad_qbgood==1)&(Z.new_pred==1)].shape[0]/(Z[Z.bad_qbgood==1].shape[0])
    print('good_qbbad recognize: {}/{}\nbad_qbgood recognize:{}/{}'.format(Z[(Z.good_qbbad==1)&(Z.new_pred==0)].shape[0],
                               Z[Z.good_qbbad==1].shape[0],
                               Z[(Z.bad_qbgood==1)&(Z.new_pred==1)].shape[0],
                               Z[Z.bad_qbgood==1].shape[0]))
    return NewPred_GoodQbbad_Rate, NewPred_BadQbgood_Rate

class BunchSingleEval:
    '''
    实践中获得一个经验，能够boosting cv 的特征，其基于单变量的检测也超级猛，故写这个单变量检测器，尤其适合ID类;
    另外，希望有针对性的找特征，对于hard samples 有针对性监控；
    '''

    def __init__(self, test, num_process=None, dive_process=None):
        self.test = test
        self.num_process = num_process
        self.dive_process = dive_process

    def _KS_check(self, tra, test, thresh=0.05, columns=None):
        '''删除掉 分布不一致的特征, p value<0.05的都扔掉；'''
        from scipy.stats import ks_2samp
        features_check = []
        if columns is None:
            assert set(tra.columns) == set(test.columns)
            columns_to_check = tra.columns
        else:
            assert (len(set(columns) - set(tra.columns)) == 0) & (len(set(columns) - set(test.columns)) == 0)
            columns_to_check = columns

        for i in columns_to_check:
            features_check.append(ks_2samp(tra[i], test[i])[1])

        features_check = pd.Series(features_check, index=columns_to_check).sort_values()
        self.weak = list(features_check[features_check <= thresh].index)
        print('drop {}/{} columns'.format(len(self.weak), len(columns_to_check)))
        return self.weak

    def _bunch_single_feat_eval(self, bunch_x, y):
        self.ICS = IndCountScore_test.IndCountScoreSingle(cols=None, num_process=self.num_process)
        ICS_x = self.ICS.fit_transform(bunch_x, y)
        return ICS_x, self.ICS.get_result

    def summary(self, x, y):
        ICS_x, imp = self._bunch_single_feat_eval(x, y)
        not_stable = self._KS_check(x, self.test, thresh=0.05, columns=None)
        imp['not_stable'] = 0
        imp.loc[not_stable, 'not_stable'] = 1
        return ICS_x, imp

    def sub_dive(self, sub_columns, y, bad_qbgood, good_qbbad):
        res = {}
        for col in sub_columns:
            res[col] = {}
            Z = y.copy()
            Z['pred'] = self.ICS_x[col]
            Z = Z[Z.index.isin(good_qbbad)]
            _, ks = PlotKS(self.ICS_x[col], y.values.flatten(), 20)
            Z['pred_good'] = (Z.pred < ks) * 1
            res[col]['good_qb2good_rate'] = Z.pred_good.sum() / Z.shape[0]
            res[col]['good_qb2good_sum'] = Z.pred_good.sum()
            res[col]['good_qb2bad_total'] = Z.shape[0]
            Z = y.copy()
            Z['pred'] = self.ICS_x[col]
            Z = Z[Z.index.isin(bad_qbgood)]
            _, ks = PlotKS(self.ICS_x[col], y.values.flatten(), 20)
            Z['pred_bad'] = (Z.pred > ks) * 1
            res[col]['bad_qb2bad_rate'] = Z.pred_bad.sum() / Z.shape[0]
            res[col]['bad_qb2bad_sum'] = Z.pred_bad.sum()
            res[col]['bad_qb2bad_total'] = Z.shape[0]
        eval_ = pd.DataFrame.from_dict(res, 'index')
        return eval_


    def dive_eval(self, x, y, bad_qbgood, good_qbbad):
        self.ICS_x, imp = self.summary(x, y)
        if self.dive_process != None:
            sub_len = len(self.ICS_x.columns) // self.dive_process + 1
            sub_list = [self.ICS_x.columns[x:x + sub_len] for x in range(0, len(self.ICS_x.columns), sub_len)]
            self.dive_process = min(self.dive_process, len(sub_list))
            print('program is going to use multiprocessing with {} Ps'.format(self.dive_process))
            p3 = Pool(self.dive_process)
            res = []
            for i in range(self.dive_process):
                aa = p3.apply_async(self.sub_dive, args=(sub_list[i], y, bad_qbgood, good_qbbad))
                res.append(aa)
            p3.close()
            p3.join()
            for idx, df in enumerate(res):
                if idx == 0:
                    df_ = df.get()
                else:
                    df_ = df.get().append(df_)
            return imp.join(df_).sort_values('bad_qb2bad_rate', ascending=False)
        else:
            return imp.join(self.sub_dive(self.ICS_x.columns, y, bad_qbgood, good_qbbad)).sort_values('bad_qb2bad_rate', ascending=False)


if __name__ == '__main__':
    tra_x = pd.read_csv('../data/train.csv', encoding="utf-8").set_index('ID')
    test = pd.read_csv('../data/test.csv', encoding="utf-8").set_index('ID')
    Y = pd.read_csv('../data/train_label.csv', encoding="utf-8").set_index('ID')
    submission = pd.read_csv('../data/submission.csv')
    id_cols = ['企业类型', '登记机关', '行业代码', '行业门类', '企业类别', '管辖机关']
    bad_qbgood = [504, 1577, 2221, 2499, 3829, 5089, 9449, 10882, 11068, 11286, 11563, 11717, 13102, 13379, 14018,
                  15329, 16763, 16883, 18114, 18182, 18538, 20567, 21418, 22352, 23312]
    good_qbbad = [10, 76, 104, 123, 132, 136, 171, 178, 223, 241, 261, 270, 279, 302, 341, 350, 383, 449, 519, 540, 577,
                  646, 661, 670, 679, 756, 794, 799, 824, 843, 873, 890, 902, 962, 969, 1129, 1169, 1208, 1210, 1245,
                  1259, 1272, 1286, 1306, 1340, 1343, 1457, 1474, 1493, 1495, 1500, 1574, 1645, 1648, 1777, 1817, 1867,
                  2033, 2082, 2111, 2125, 2160, 2197, 2251, 2271, 2279, 2290, 2303, 2305, 2306, 2435, 2486, 2488, 2493,
                  2562, 2649, 2651, 2709, 2761, 2799, 2841, 2938, 2966, 3002, 3035, 3079, 3090, 3218, 3285, 3287, 3296,
                  3314, 3333, 3335, 3342, 3343, 3363, 3404, 3407, 3468, 3503, 3524, 3575, 3584, 3636, 3721, 3771, 3845,
                  3878, 3912, 3962, 4020, 4068, 4097, 4128, 4155, 4210, 4250, 4262, 4265, 4321, 4373, 4376, 4418, 4480,
                  4544, 4603, 4676, 4687, 4794, 4858, 4986, 5001, 5007, 5019, 5075, 5117, 5128, 5144, 5172, 5216, 5228,
                  5317, 5331, 5408, 5420, 5449, 5541, 5553, 5676, 5684, 5751, 5791, 5851, 5908, 5926, 5940, 5955, 6081,
                  6106, 6144, 6188, 6267, 6285, 6339, 6384, 6385, 6391, 6404, 6450, 6452, 6471, 6531, 6533, 6578, 6606,
                  6648, 6658, 6677, 6812, 6813, 6935, 6978, 7059, 7131, 7190, 7326, 7340, 7408, 7410, 7413, 7431, 7557,
                  7567, 7571, 7578, 7582, 7608, 7706, 7745, 7830, 7871, 7872, 7945, 7946, 7957, 7987, 7993, 8012, 8022,
                  8086, 8118, 8166, 8195, 8216, 8288, 8381, 8394, 8538, 8582, 8584, 8592, 8603, 8655, 8720, 8734, 8808,
                  8966, 9001, 9020, 9065, 9087, 9105, 9115, 9134, 9138, 9249, 9273, 9325, 9344, 9412, 9416, 9475, 9486,
                  9525, 9542, 9544, 9589, 9594, 9598, 9615, 9619, 9629, 9635, 9642, 9671, 9697, 9709, 9768, 9800, 9808,
                  9872, 9953, 10019, 10066, 10097, 10099, 10101, 10142, 10180, 10196, 10228, 10245, 10248, 10328, 10352,
                  10375, 10396, 10399, 10433, 10494, 10515, 10526, 10542, 10584, 10725, 10736, 10738, 10746, 10896,
                  10909, 10941, 10982, 10994, 11041, 11057, 11078, 11093, 11151, 11186, 11194, 11225, 11294, 11330,
                  11338, 11357, 11427, 11454, 11466, 11525, 11533, 11634, 11661, 11663, 11682, 11731, 11834, 11858,
                  11886, 11917, 11949, 11976, 12012, 12037, 12113, 12223, 12249, 12310, 12325, 12388, 12422, 12435,
                  12444, 12454, 12483, 12555, 12559, 12594, 12606, 12640, 12644, 12692, 12733, 12753, 12832, 12867,
                  12915, 12960, 12978, 13000, 13060, 13128, 13150, 13224, 13286, 13291, 13328, 13331, 13387, 13413,
                  13436, 13495, 13543, 13553, 13603, 13604, 13645, 13677, 13696, 13711, 13783, 13923, 13966, 14023,
                  14038, 14086, 14108, 14114, 14223, 14235, 14265, 14273, 14351, 14380, 14381, 14414, 14443, 14451,
                  14491, 14516, 14644, 14650, 14684, 14688, 14772, 14840, 14865, 14870, 14903, 14934, 14950, 14959,
                  14963, 14987, 15008, 15024, 15080, 15118, 15151, 15222, 15228, 15287, 15320, 15356, 15386, 15389,
                  15420, 15428, 15444, 15519, 15529, 15531, 15543, 15571, 15622, 15648, 15686, 15756, 15775, 15786,
                  15793, 15884, 15885, 15925, 15943, 16006, 16034, 16039, 16111, 16125, 16133, 16148, 16233, 16235,
                  16260, 16356, 16379, 16515, 16564, 16583, 16592, 16595, 16602, 16638, 16676, 16762, 16775, 16783,
                  16792, 16869, 16870, 16879, 16969, 16980, 16999, 17058, 17082, 17118, 17129, 17157, 17170, 17211,
                  17257, 17290, 17362, 17398, 17441, 17450, 17500, 17512, 17515, 17526, 17545, 17587, 17588, 17593,
                  17619, 17650, 17673, 17702, 17723, 17774, 17793, 17841, 17851, 17874, 17880, 17894, 17909, 17940,
                  17958, 17962, 18073, 18193, 18220, 18274, 18303, 18325, 18382, 18413, 18447, 18494, 18541, 18556,
                  18680, 18689, 18765, 18805, 18864, 18906, 18995, 19031, 19072, 19132, 19143, 19145, 19159, 19183,
                  19200, 19224, 19264, 19288, 19353, 19364, 19408, 19428, 19446, 19484, 19497, 19502, 19655, 19659,
                  19713, 19733, 19763, 19768, 19772, 19839, 19871, 19897, 19962, 20025, 20108, 20194, 20201, 20245,
                  20247, 20264, 20276, 20280, 20298, 20342, 20364, 20370, 20404, 20408, 20433, 20455, 20513, 20538,
                  20563, 20634, 20636, 20684, 20688, 20737, 20747, 20788, 20791, 20833, 20950, 20961, 20971, 21073,
                  21087, 21116, 21140, 21170, 21242, 21251, 21315, 21316, 21368, 21391, 21393, 21399, 21429, 21437,
                  21438, 21466, 21475, 21480, 21564, 21584, 21586, 21606, 21695, 21714, 21803, 21830, 21897, 21919,
                  21922, 21990, 22030, 22031, 22044, 22114, 22167, 22170, 22273, 22305, 22316, 22332, 22379, 22396,
                  22425, 22431, 22452, 22478, 22481, 22498, 22547, 22590, 22596, 22601, 22603, 22615, 22631, 22686,
                  22713, 22742, 22796, 22837, 22845, 22857, 22883, 22894, 22929, 22971, 22972, 22996, 23011, 23059,
                  23066, 23120, 23157, 23172, 23204, 23205, 23208, 23260, 23299, 23317, 23385, 23430, 23475, 23476,
                  23486, 23508, 23604, 23606, 23612, 23626, 23652, 23673, 23677, 23741, 23743, 23774, 23775, 23869,
                  23915]
    from Piplines.RankEnc_test import RankEnc
    RE = RankEnc(cols=None, test=test[id_cols])
    tmp_tra = RE.fit_transform(tra_x[id_cols])
    tmp_test = RE.transform(test[id_cols])
    bsEval = BunchSingleEval(test=tmp_test, num_process=4, dive_process=None)
    res = bsEval.dive_eval(tmp_tra, Y, bad_qbgood, good_qbbad)
    print(res)
