import cv2
import os
import torch
import json
import ipdb
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
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

class JiuData(Dataset):
    
    def __init__(self, fold_id=None, mode='train', return_ori_img=False):
        from sklearn.model_selection import StratifiedShuffleSplit
        import json
        with open(annotation_path, 'r') as load_f:
            load_dict = json.load(load_f)
            
        ids = [i for i,j in enumerate(load_dict['annotations'])]
        self.mode = mode
        self.return_ori_img = return_ori_img
        if fold_id is not None:
            cls = [j['category_id'] for i,j in enumerate(load_dict['annotations'])]
            sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
            fold_dic = {}
            for id, index_tuple in enumerate(sss.split(ids, cls)):
                fold_dic[id] = index_tuple 
            if mode == 'train':
                self.fold_id = fold_dic[fold_id][0]
            elif mode == 'valid':
                self.fold_id = fold_dic[fold_id][1]
            else:
                raise Exception('not allowed')
        else:
            self.fold_id = ids
                
        with open(annotation_path, 'r') as load_f:
            self.load_dict = json.load(load_f)
            
        self.annotations = np.array(self.load_dict['annotations'])[self.fold_id]
        self.tmp_img_dic = {}
        for dic in self.load_dict['images']:
            idx = dic['id']
            self.tmp_img_dic[idx] = dic
            
        self.id2cls = {}
        for dic in self.load_dict['categories']:
            self.id2cls[dic['id']] = dic['name']
            
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, index):
        anno_dic = self.annotations[index]
        img_id = anno_dic['image_id']
        H, W, C = self.tmp_img_dic[img_id]['height'], self.tmp_img_dic[img_id]['width'], anno_dic['category_id']
        mask_img = self.mask_from_bbox(W, H, C, bbox=anno_dic['bbox'])
        img_name = self.tmp_img_dic[img_id]['file_name']
        img_ori = cv2.imread(img_path+img_name).astype(np.float32) / 255
        if self.mode == 'train':
            img, mask = train_aug(img_ori, mask_img[0]) # [H,W,3] [H,W]
            img, mask = do_resize(img, mask, 256, 256)
        else:
            img, mask = do_resize(img_ori, mask_img[0], 256, 256) # [H,W,3] [H,W]
        if (self.return_ori_img) & (self.mode=='valid'):
            return torch.from_numpy(img).permute([2,0,1]), torch.from_numpy(mask).unsqueeze(0).int(), C, torch.tensor(anno_dic['bbox']), H, W, img_ori
        
        return torch.from_numpy(img).permute([2,0,1]), torch.from_numpy(mask).unsqueeze(0).int(), C, torch.tensor(anno_dic['bbox']), H, W
    
    def mask_from_bbox(self, W, H, C, bbox):
        # cv2的x, y轴与np的x,y轴不一致；需要交换坐标系x,y才能互相转换；
        mask = np.zeros((1, H, W))
        x, y, w, h = [round(i) for i in bbox]
        mask[0, y:y+h, x:x+w] = C
        return mask


    
test_img_path = './chongqing1_round1_testA_20191223/images/'
class JiuTest(Dataset):
    
    def __init__(self):
        self.img_name_list = os.listdir(test_img_path)


    def __len__(self):
        return len(self.img_name_list)
     
    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        img = cv2.imread(test_img_path+img_name).astype(np.float32) / 255
#         print(img.shape)
        H,W,_ = img.shape
        img = cv2.resize(img,dsize=(256,256))
        return img_name, torch.from_numpy(img).permute([2,0,1]), H, W
    
    
# if __name__ == '__main__':
#     data = Jiu()
#     data.show_bbox(data[16])
    
#     data2 = Jiu2()