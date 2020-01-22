import cv2
import os
import torch
import json
import ipdb
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler
from augment import *
    

def do_resize(image, mask, H, W):
    '''
    image: H,W,C;
    mask: H,W
    '''
#     data2[5922][0]
    image = cv2.resize(image,dsize=(W,H))
    mask = cv2.resize(mask,dsize=(W,H))
    return image, mask


def train_aug(image, mask):
    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip2(image, mask)

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c == 0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2)

        if c == 1:
            image, mask = do_horizontal_shear2(image, mask, dx=np.random.uniform(-0.07, 0.07))

        if c == 2:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 15))

    if np.random.rand() < 0.5:
        c = np.random.choice(2)
        if c == 0:
            image = do_brightness_shift(image, np.random.uniform(-0.1, +0.1))
        if c == 1:
            image = do_brightness_multiply(image, np.random.uniform(1 - 0.08, 1 + 0.08))

    return image, mask


annotation_path = "../chongq/chongqing1_round1_train1_20191223/annotations.json"
img_path = './chongqing1_round1_train1_20191223/images/'

def all_comb(max_num):
    from itertools import combinations, permutations
    res = []
    for num in range(1, max_num+1):
        res.extend(list(combinations(range(1,11), num)))
    dic = {}
    st = 1
    for i in res:
        str_=''
        for j in i:
            str_+='{}|'.format(j)
        dic[str_] = st
        st += 1
    dic['1|2|4|'] = st+0
    dic['6|7|8|'] = st+1
    return dic


def mask_enc(mask):
    '''
    把mask_img 编码为trick形式；
    0-10 >>> 0, 10*{1}
    mask: 3,H,W
    reutrn: 3,H,W
    '''
    map_dict = {}
    for cls in range(1, 11):
        map_dict[cls] = int('1'*cls)
    
    for i in range(1, 11):
        mask[mask==i] =  map_dict[i]    
    
    return mask


def decode(num): 
    '''
    decode(111223)={1, 3, 6}
    '''
    num = int(num)
    tmp_leng = len(str(num))
    set_cls = [tmp_leng]
    tmp_sub = int('1'*tmp_leng)
    tmp_remain = num - tmp_sub
    while tmp_remain != 0:
        num =  tmp_remain
        tmp_leng = len(str(num))
        set_cls.append(tmp_leng)
        tmp_sub = int('1'*tmp_leng)
        tmp_remain = num - tmp_sub    
    return set(set_cls)


def mask_dec(enc, max_comb):
    '''
    把trick形式 解码为 mask_img;
    enc 元素必须整数;
    '''
    enc = enc.astype(np.int)
    enc_dic = all_comb(max_comb)
    USEFUL_CLS={0,
         1,
         2,
         3,
         4,
         5,
         6,
         7,
         8,
         9,
         10,
         11,
         12,
         13,
         14,
         18,
         21,
         22,
         29,
         46,
         47,
         50,
         56,
         57}
    CLS_MAP = {}
    idx = 0
    for i in USEFUL_CLS:
        CLS_MAP[i] = idx
        idx+=1

    clses = np.unique(enc)
    hh = []
    for cls in clses:
        if cls not in [0,1]:
            tmp = sorted(decode(cls))
            str_=''
            for i in tmp:
                str_+='{}|'.format(i)
            enc[enc==cls] = CLS_MAP[enc_dic[str_]]
            hh.append(enc_dic[str_])
    return enc.astype(np.float32), hh

# 假设sub_masks 之间不会有 交集;
# annotations 中的 id 豪无用处; 而我之前没注意;
# 假设我是主办方，我不可能要求test背景样本给出bbox;
# 因此我根本不用训练背景样本；即便你给出 annotation==0 的bbox;
# 优化，大部分根本不需要enc_dec;

img_path = './chongqing1_round1_train1_20191223/images/'

class Jiu2Data(Dataset):

    
    def __init__(self, fold_id=None, mode='train', return_ori_img=False):
        
        # 0. load & k fold split;
        import json
        with open(annotation_path, 'r') as load_f:
            self.load_dict = json.load(load_f)
        from sklearn.model_selection import GroupKFold
        ids = [i for i,j in enumerate(self.load_dict['images'])]
        if fold_id is not None:
            fold_dic = {}
            np.random.seed(0)
            gps = np.random.randint(0,5,(len(ids)))
            sss = GroupKFold(n_splits=5)
            for id, index_tuple in enumerate(sss.split(X = ids, groups=gps)):
                fold_dic[id] = index_tuple
            if mode == 'train':
                self.fold_id = fold_dic[fold_id][0]
            elif mode == 'valid':
                self.fold_id = fold_dic[fold_id][1]
            else:
                raise Exception('not allowed')
        else:
            self.fold_id = ids
          
        # 1. annotations 先修正 anno_id ; zero index;
        self.annotations = {}
        for idx, i in enumerate(self.load_dict['annotations']):
            i['id'] = idx
            self.annotations[idx] = i
            
        # 2. tmp_img_dic & k-fold in;
        self.tmp_img_dic = {}
        for dic in self.load_dict['images']:
            idx = dic['id']
            if idx-1 in self.fold_id:
                self.tmp_img_dic[idx] = dic
        self.index2imgid = dict(enumerate(self.tmp_img_dic.keys()))
        
        # 3. img 对应 多个 annotations; 同类取并集，不同类相加; 
        self.img2annotations = {}
        for anno_id, i in enumerate(self.annotations): 
            
            img_id = self.annotations[i]['image_id']
            cls = self.annotations[i]['category_id']
            if img_id not in self.img2annotations:
                self.img2annotations[img_id] = {}
                if cls not in self.img2annotations[img_id]:
                    self.img2annotations[img_id][cls] = []
                    self.img2annotations[img_id][cls].append(anno_id)
                else:
                    self.img2annotations[img_id][cls].append(anno_id)
            else:
                if cls not in self.img2annotations[img_id]:
                    self.img2annotations[img_id][cls] = []
                    self.img2annotations[img_id][cls].append(anno_id)
                else:
                    self.img2annotations[img_id][cls].append(anno_id)
                    
        # 4. mode. 
        self.mode = mode
        self.return_ori_img = return_ori_img
        
    def __len__(self):
        return len(self.tmp_img_dic)-1
    
    @property
    def get_cls_map(self):
        USEFUL_CLS={0,
                 1,
                 2,
                 3,
                 4,
                 5,
                 6,
                 7,
                 8,
                 9,
                 10,
                 11,
                 12,
                 13,
                 14,
                 18,
                 21,
                 22,
                 29,
                 46,
                 47,
                 50,
                 56,
                 57}
        CLS_MAP = {}
        idx = 0
        for i in USEFUL_CLS:
            CLS_MAP[i] = idx
            idx+=1
        return CLS_MAP
        
    def __getitem__(self, index):              
        index = self.index2imgid[index] # [one index]
        img_name, H, W = self.tmp_img_dic[index]['file_name'], self.tmp_img_dic[index]['height'], self.tmp_img_dic[index]['width']
        img_ori = cv2.imread(img_path + img_name).astype(np.float32) / 255
    
        final_mask_img = np.zeros((1, H, W))
        # 每一张图 可能有多个cls; 得到多个 C_mask_img;
        if len(self.img2annotations[index]) == 1: # 单个class不需要编码；
            cls = list(self.img2annotations[index].keys())[0]
            for anno_id in self.img2annotations[index][cls]:
                anno_dic = self.annotations[anno_id]
                tmp_bbox = anno_dic['bbox']
                tmp_mask_img = self.mask_from_bbox(W, H, cls, bbox=tmp_bbox)
                final_mask_img += tmp_mask_img
            final_mask_img[final_mask_img!=0] = cls
            cls_lis = [cls]
                           
            
        else: # 只有multi_cls才需要编码解码；
            for cls in self.img2annotations[index]:
                # 每个cls可能有多个annotations; 得到一个综合的C_mask_img;
                C_mask_img = np.zeros((1, H, W))
                for anno_id in self.img2annotations[index][cls]:
                    anno_dic = self.annotations[anno_id]
                    tmp_bbox = anno_dic['bbox']
                    tmp_mask_img = self.mask_from_bbox(W, H, cls, bbox=tmp_bbox)
                    C_mask_img += tmp_mask_img
                C_mask_img[C_mask_img!=0] = cls
                # 多个 C_mask_img 最终还是要合并， 先编码再解码；
                final_mask_img += mask_enc(C_mask_img)
            final_mask_img, cls_lis = mask_dec(final_mask_img, max_comb=2)

        if self.mode == 'train':
            img, mask = train_aug(img_ori, final_mask_img[0]) # [H, W, 3] [H, W]
            img, mask = do_resize(img, mask, 256, 256)
        else:
            img, mask = do_resize(img_ori, final_mask_img[0], 256, 256) # [H, W, 3] [H, W]
        if self.return_ori_img:
            return torch.from_numpy(img).permute([2,0,1]), torch.from_numpy(mask).unsqueeze(0).int(), H, W, img_ori
        return torch.from_numpy(img).permute([2,0,1]), torch.from_numpy(mask).unsqueeze(0).int(), H, W
#         return torch.from_numpy(img).permute([2,0,1]), torch.from_numpy(mask).unsqueeze(0).int(), torch.tensor(cls_lis), H, W
    
    def mask_from_bbox(self, W, H, C, bbox):
        # cv2的x, y轴与np的x,y轴不一致；需要交换坐标系x, y才能互相转换；
        mask = np.zeros((1, H, W))
        x, y, w, h = [round(i) for i in bbox]
        mask[0, y:y+h, x:x+w] = C
        return mask
    
    @property
    def label_dict(self):
        str2int = all_comb(max_num=2)
        int2str = {i:j for j,i in str2int.items()}
        return str2int, int2str
    
    def get_all_cls(self, dataset):
        res=  []
        for i in progress_bar(dataset):
            res.extend(i[2].tolist())
        return set(res)