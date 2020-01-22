from data import *

def mask2bbox(mask):
    # mask:np array with shape:[H,W]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if (cols.sum()==0) & (rows.sum()==0):
        return 0, 0, 0, 0
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return cmin, rmin, cmax - cmin, rmax - rmin

def mask2bbox_withscale(mask, ori_h, ori_w):
    # mask: H,W
    assert len(mask.shape)==2
    h, w = mask.shape
    fac_h, fac_w = ori_h.item()/h, ori_w.item()/w
    bbox = mask2bbox(mask)
    x, y, x_len, y_len = bbox
    new_x, new_y, new_x_len, new_y_len = fac_w*x, fac_h*y, fac_w*x_len, fac_h*y_len
    return new_x, new_y, new_x_len, new_y_len

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
    

def cls_map(batch_data):
    '''把多分类的logts[bs,cls,h,w] 转化为 元素级类别矩阵[bs,h,w]; '''
    assert batch_data.dim() == 4
    batch_data = batch_data.softmax(1).argmax(1)
#     plt.imshow(batch_data[0])
    return batch_data

# if __name__ == '__main__':
    
#     # iou
#     res  = calcIOU((1, 2, 2, 2), (2, 1, 2, 2))
#     print(res)
    
#     # ap & mAP
#     c_label_list = np.random.randint(1,11,100)
#     c_label = 2
#     c_pred = np.random.randint(0,4,100)
#     data = Jiu()
#     bb1_list = []
#     for i in range(100):
#         bb1_list.append(data[i][4])
#     bb2_list = bb1_list
#     img_w = np.random.randint(20,450,100)
#     img_h = np.random.randint(20,450,100)
#     ap = cal_ap(c_pred, c_label, bb1_list, bb1_list, img_w, img_h)
#     print(ap)
#     # weight
#     weight_dic = {1:0.15,
#     2:0.09,
#     3:0.09,
#     4:0.05,
#     5:0.13,
#     6:0.05,
#     7:0.12,
#     8:0.13,
#     9:0.07,
#     10:0.12}
#     map_ = mAP(weight_dic, c_pred, c_label_list, bb1_list, bb1_list, img_w, img_h)
#     print(map_)
    
#     valid_data = Jiu3(fold_id=0, mode='valid')
#     print(mask2bbox(valid_data[3][1][0].numpy()))
#     print(mask2bbox_withscale(valid_data[3][1][0].numpy(), 492, 658))
#     print(valid_data[3][3])
    
    
    
    
import torch
from data import JiuData, do_resize, JiuTest
from model import *
from augment import *
from sampler import JiuSampler
from loss import multi_class_entropy, lovasz_softmax, FocalLoss2d, DiceLoss2D, FocalLoss2D_2
from fastprogress.fastprogress import master_bar, progress_bar

def complicate_segment_post_process(mask, ori_h=torch.tensor([492]), ori_w=torch.tensor([658]), num_cls=11):
    dic = {}
    for c in range(1,num_cls):
        c_sum = (mask.numpy()==c).sum()
        if c_sum>3:
            tmp_mask = np.where(mask.numpy(), mask.numpy()==c, 0)
            
            # mask:np array with shape: [H,W];
            rows = np.any(tmp_mask, axis=1)
            cols = np.any(tmp_mask, axis=0)

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            h,w=tmp_mask.shape

            #check 分裂；
            def is_not_split(x):
                return len(x) == (x[-1] - x[0])

            if is_not_split(np.where(rows)[0]) & is_not_split(np.where(cols)[0]):
                print('single!!!!!!!!!!!')
                bbox_list = [[cmin, rmin, cmax - cmin, rmax - rmin]]
            else:
                if is_not_split(np.where(rows)[0]): # 行是不可分割的
#                     print('row eval')
                    gps = get_split_gps(rows)
                    bbox_list = sub_bbox(gps, axis=1, mask=tmp_mask)

                elif is_not_split(np.where(cols)[0]): # 列不可分割
#                     print('col eval')
                    gps = get_split_gps(cols)
                    bbox_list= sub_bbox(gps, axis=0, mask=tmp_mask)   
                    
                else: # 行是分开的， 列也是分开的， 走任意分支都可以算出来；
#                     print('both eval')
#                     gps = get_split_gps(rows)
#                     bbox_list = sub_bbox(gps, axis=1, mask=tmp_mask)
                
                    gps = get_split_gps(cols)
                    bbox_list= sub_bbox(gps, axis=0, mask=tmp_mask) 
                    
            
            new_box = [bbox_withscale(box, ori_h, ori_w, h, w) for box in bbox_list]
            if new_box:
                dic[c] = new_box
    return dic

def bbox_withscale(bbox, ori_h, ori_w, h,w):
    fac_h, fac_w = ori_h.item()/h, ori_w.item()/w
    x, y, x_len, y_len = bbox
    new_x, new_y, new_x_len, new_y_len = fac_w*x, fac_h*y, fac_w*x_len, fac_h*y_len
    return new_x, new_y, new_x_len, new_y_len


def get_split_gps(rows):
    '''
    rows or cols :array([ 84,  85,  86, 193, 194, 195, 196, ...])
    get split groups: [array([84, 85, 86]), array([193, 194, 195, 196, ...])]
    
    '''

    zz = np.where(rows)[0]
    pre_start = zz[0]
    length = len(zz)
    flag=True
    res = []
    while flag:
#         ipdb.set_trace()
        s1 = zz[zz == range(pre_start, pre_start+length)]
        if list(s1)==list(zz):
            res.append(s1)
            break
        zz = zz[len(s1):]

        pre_start = zz[0]
        length = len(zz)
        res.append(s1)

        if np.array(s1==zz).all():
            flag = False
    return res

def sub_bbox(gps, axis, mask):
    '''
    gps:  [array([84, 85, 86]), array([193, 194, 195, 196,...], array([, ,]))]
    axis: element in [0,1]  {col:0,  row:1};
    mask: [H, W]
    return: [[84, 103, 2, 10], [193, 63, 16, 28], [...]]
    '''
    boxs= []
    for gp in gps:
    #     print(gp)
        if axis == 1: # row;
            gp_tmp = mask[gp]
            rmin, rmax = gp[0], gp[-1]
            cols = np.any(gp_tmp, axis=0)
            cmin, cmax = np.where(cols)[0][[0, -1]]
        elif axis == 0: # col;
            gp_tmp = mask[:, gp]
            cmin, cmax = gp[0], gp[-1]
            rows = np.any(gp_tmp, axis=1)
            rmin, rmax = np.where(rows)[0][[0, -1]]
        bbox = [cmin, rmin, cmax - cmin, rmax - rmin]
        if bbox[2]*bbox[3]!=0:
            boxs.append(bbox)
    return boxs


def get2dict(load_dict):
    imgid2name = {}
    for i in load_dict['images']:
        img_id = i['id']
        imgid2name[img_id] = i

    name2imgid = {}
    for i in imgid2name:
        f_name = imgid2name[i]['file_name']
        name2imgid[f_name] = i
    return imgid2name, name2imgid

def get_id2bbox(load_dict):
    # 同一张图片可能有多个annotations;
    gt_id2bbox = {}
    for i in load_dict['annotations']:
        idx = i['image_id']
        if idx not in gt_id2bbox:
            gt_id2bbox[idx] = []
            gt_id2bbox[idx].append(i)
        else:
            gt_id2bbox[idx].append(i)
    return gt_id2bbox

def infer_upload(model, loader, pre_train_pth='./models/unet_qb_aug_focal99.pth', device='cuda', show_img_id=-1):
    
    if model is None:
        param = torch.load(pre_train_pth)  # stage3 use model pretrained with pseudo-labels
        model.load_state_dict(param)  # initialize with pretained weight
    model = model.to(device)
    model.eval()
    DIC = {}
    DIC['images'] = []
    DIC['annotations']=[]
    i=0
    for index, tup in enumerate(progress_bar(loader)):  
#         print(i)
        i+=1
        name, imgs, ori_h, ori_w, ori_img = tup
#             ipdb.set_trace()
        DIC['images'].append({'file_name': name[0],
                              'id': index+1})

        imgs = imgs.to(device)
        with torch.no_grad():

            mask_pred, cls_logit = model(imgs) # [num_cls, w, h]
            mask_pred = mask_pred.to('cpu').softmax(1).argmax(1)[0]
            if show_img_id == index:
                return mask_pred, imgs, ori_img, name
#                 print('------')
#                 plt.subplot(211)
#                 plt.imshow(mask_pred)
                
            dic_temp = complicate_segment_post_process(mask_pred, ori_h, ori_w, num_cls=11)
            if dic_temp: 
                for cls_ in dic_temp:
                    DIC['annotations'].append({'image_id':index+1,
                                   'bbox': [int(i) for i in dic_temp[cls_][0]],
                                   'category_id':cls_,
                                   'score':1})
            else:
                DIC['annotations'].append({'image_id':index+1,
                                   'bbox': [0,0,0,0],
                                   'category_id': 0,
                                   'score':1})
    return DIC

def eval_mAP(loader, model, pre_train_pth='./models/unet_qb_aug_focal99.pth'):
    annotation_path = "../chongq/chongqing1_round1_train1_20191223/annotations.json"
    import json
    with open(annotation_path, 'r') as load_f:
        load_dict = json.load(load_f)
        
    ress = infer_upload(model=model, pre_train_pth=pre_train_pth, loader=loader, show_img_id=-1)
    pred_id2name, pred_name2id = get2dict(ress)
    gt_id2name, gt_name2id = get2dict(load_dict)
    pred_id2bbox = get_id2bbox(ress)
    gt_id2bbox = get_id2bbox(load_dict)

    RES = {}
    num_cls=11
    for i in range(0, num_cls):
        RES[i] = []

    idx = 0
    for i in pred_id2name:

        f_name = pred_id2name[i]['file_name']
        gt_id = gt_name2id[f_name]
        h, w = gt_id2name[gt_id]['height'], gt_id2name[gt_id]['width']

        gt_bboxs_list = gt_id2bbox[gt_id]
        pred_bboxes_list = pred_id2bbox[i]

        # 每一个gt_bbox 都有一个score; 每个img都有k个gt_bbox 故 k个score;
        img_res = []
        for gt_bbox in gt_bboxs_list:
            tmp_gt_bbox = gt_bbox['bbox']
            tmp_gt_cls = gt_bbox['category_id']
            tmp_res = 0
            for pre_bbox in pred_bboxes_list:
                tmp_pre_bbox = pre_bbox['bbox']
                tmp_pre_cls = pre_bbox['category_id']

                if tmp_pre_cls == tmp_gt_cls:
                    if tmp_gt_cls == 0:
                        res = 1.
                    else:
                        iou_th = iou_thresh(w, h)
                        iou = calcIOU(tmp_pre_bbox, tmp_gt_bbox)
                        res = (iou > iou_th) * 1.0  
                else:
                    res = 0.

                if res>=tmp_res:
                    tmp_res = res
            RES[tmp_gt_cls].append(tmp_res)
            img_res.append(tmp_res)


    mAP = 0
    for i in RES:
        res = RES[i]
        ap = sum(res)/len(res)
        mAP += ap
    mAP = mAP/num_cls
    return mAP


if __name__ == '__main__':
    model = Unet_qb(num_class=11, HyperColumn=True)
    param = torch.load('./models/unet_qb_aug_focal99' + '.pth')  # stage3 use model pretrained with pseudo-labels
    model.load_state_dict(param)  # initialize with pretained weight
    device = 'cuda'
    model = model.to(device)
    model.eval()
    testset = JiuTest()
    name, imgs, ori_h, ori_w, ori_img = testset[1282]
    imgs = imgs.unsqueeze(0).to(device)
    ori_img = imgs.unsqueeze(0)
    ori_h, ori_w = torch.tensor(ori_h), torch.tensor(ori_w)

    with torch.no_grad():
        mask_pred, cls_logit = model(imgs) # [num_cls, w, h]
        mask_pred = mask_pred.to('cpu').softmax(1).argmax(1)[0]
        import matplotlib.pyplot as plt
        plt.imshow(mask_pred)
#         ipdb.set_trace()
        dic = complicate_segment_post_process(mask_pred)
    
