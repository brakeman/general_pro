from data import *
def iou_thresh(w, h):
    '''
    iou_thresh(120, 400)
    '''
    min_axis = min(w, h)
    if min_axis < 40:
        val = 0.2
    elif (min_axis >= 40) & (min_axis<120):
        val = min_axis/200
    elif (min_axis >= 120) & (min_axis<420):
        val = min_axis/1500 + 0.52
    elif min_axis>=420:
        val = 0.8
    else:
        raise Exception('min_axis:{} not allowed'.format(min_axis))
    return val


def calcIOU(bbox1, bbox2):
    '''
    res  = calcIOU((1, 2, 2, 2), (2, 1, 2, 2))
    '''
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2
    if((abs(x1 - x2) < ((w1 + w2)/ 2.0)) and (abs(y1-y2) < ((h1 + h2)/2.0))):
        left = max((x1 - (w1 / 2.0)), (x2 - (w2 / 2.0)))
        upper = max((y1 - (h1 / 2.0)), (y2 - (h2 / 2.0)))

        right = min((x1 + (w1 / 2.0)), (x2 + (w2 / 2.0)))
        bottom = min((y1 + (h1 / 2.0)), (y2 + (h2 / 2.0)))

        inter_w = abs(left - right)
        inter_h = abs(upper - bottom)
        inter_square = inter_w * inter_h
        union_square = (w1 * h1)+(w2 * h2)-inter_square

        calcIOU = inter_square/union_square * 1.0
        return round(calcIOU, 3) 
    else:
        return 0
    return calcIOU

    


def cal_ap(c_pred, c_label, pred_bbox, real_bbox, img_w, img_h):
    '''
    c_pred: [bs];
    c_label: int;
    pred_bbox: list of [x,y,w,h];
    real_bbox: list of [x,y,w,h];
    img_w: [w]
    img_h: [h]
    '''
    
    round1_idx = np.where(c_pred == c_label)[0]
#     ipdb.set_trace()
    pred_list = []
    for idx in round1_idx:
        iou_th = iou_thresh(img_w[idx],img_h[idx])
        is_true = (calcIOU(pred_bbox[idx], real_bbox[idx]) > iou_th) * 1.0
        pred_list.append(is_true)
#     ipdb.set_trace()
    round2_idx = np.where(np.array(pred_list) == 1.0)[0]
    return len(round2_idx)/len(c_pred)


def mAP(weight, c_pred, c_label, pred_bbox, real_bbox, img_w, img_h):
    '''
    weight: {C: weight_value};
    c_pred: [bs];
    c_label: [bs];
    pred_bbox: list of [x,y,w,h];
    real_bbox: list of [x,y,w,h];
    img_w: [w]
    img_h: [h]
    '''
    mAp = 0
    c_label = np.array(c_label)
    for C in range(1,10):
        C_idx = np.where(c_label==C)[0]
#         ipdb.set_trace()
        
        C_ap = cal_ap(np.array(c_pred)[C_idx], np.array(c_label)[C_idx], np.array(pred_bbox)[C_idx], 
                                np.array(real_bbox)[C_idx], np.array(img_w)[C_idx], np.array(img_h)[C_idx])
        mAp+=C_ap*weight[C]
    return mAp
    
    
if __name__ == '__main__':
    
    # iou
    res  = calcIOU((1, 2, 2, 2), (2, 1, 2, 2))
    print(res)
    
    
    # ap & mAP
    c_label_list = np.random.randint(1,11,100)
    c_label = 2
    c_pred = np.random.randint(0,4,100)
    data = Jiu()
    bb1_list = []
    for i in range(100):
        bb1_list.append(data[i][4])
    bb2_list = bb1_list
    img_w = np.random.randint(20,450,100)
    img_h = np.random.randint(20,450,100)
    ap = cal_ap(c_pred, c_label, bb1_list, bb1_list, img_w, img_h)
    print(ap)
    # weight
    weight_dic = {1:0.15,
    2:0.09,
    3:0.09,
    4:0.05,
    5:0.13,
    6:0.05,
    7:0.12,
    8:0.13,
    9:0.07,
    10:0.12}
    map_ = mAP(weight_dic, c_pred, c_label_list, bb1_list, bb1_list, img_w, img_h)
    print(map_)
