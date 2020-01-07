import cv2
import os
import torch
import json
import ipdb
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


annotation_path = "../chongq/chongqing1_round1_train1_20191223/annotations.json"
img_path = './chongqing1_round1_train1_20191223/images/'

# import os
# if not os.path.exists('chongqing1_round1_train1_mask_qb'):
#     os.makedirs('chongqing1_round1_train1_mask_qb')
    
# with open('round1_train1_mask.csv','r') as f:
#     lines = f.readlines()   


# data = Jiu() 
# for i in range(len(data)):
#     mask = data[i][1]
#     img_name, mask_name = lines[i].split(',')[0], lines[i].split(',')[1].split('\n')[0]
#     assert data[i][-1] == img_name
#     save_path = 'chongqing1_round1_train1_mask_qb/{}'.format(mask_name)
#     cv2.imwrite(save_path, mask)



# with open('round1_train1_mask.csv','w+') as f:
#     data = Jiu()
#     dic = {}
#     for i in range(len(data)):
#         img_name = data[i][-1]
#         if img_name not in dic:
#             dic[img_name] = 0
#         else:
#             dic[img_name] += 1
#         pre_fix = dic[img_name]
#         img_mask_name = 'mask_{}_'.format(pre_fix) + img_name.split('_')[-1]
#     #     print('{},{}\n'.format(img_name, img_mask_name))
#         f.write('{},{}\n'.format(img_name, img_mask_name))

class Jiu(Dataset):
    def __init__(self):
        
        with open(annotation_path, 'r') as load_f:
            self.load_dict = json.load(load_f)
            
        self.tmp_img_dic = {}
        for dic in self.load_dict['images']:
            idx = dic['id']
            self.tmp_img_dic[idx] = dic
            
        self.id2cls = {}
        for dic in self.load_dict['categories']:
            self.id2cls[dic['id']] = dic['name']
            
    def __len__(self):
        return len(self.load_dict['annotations'])
        
    def __getitem__(self, index):
        anno_dic = self.load_dict['annotations'][index]
        img_id = anno_dic['image_id']
        H, W, C = self.tmp_img_dic[img_id]['height'], self.tmp_img_dic[img_id]['width'], anno_dic['category_id']
        mask_img = self.mask_from_bbox(W, H, C, bbox=anno_dic['bbox'])
        img_name = self.tmp_img_dic[img_id]['file_name']
        img = cv2.imread(img_path+img_name).astype(np.float32) / 255
        return img, mask_img, C, self.id2cls[C], anno_dic['bbox'], img_name
    
    def mask_from_bbox(self, W, H, C, bbox):
        # 图片的x, y轴与np的 x,y轴不一致；需要交换坐标系x,y才能互相转换；
        mask = np.zeros((1, H, W))
#         print('mask shape:{}'.format(mask.shape))
        x, y, w, h = [round(i) for i in bbox]
        mask[0, y:y+h, x:x+w] = C
#         mask[0, x:x+w, y:y+h] = C # 不对；
#         print('rec:({},{})'.format((x,y), (x+w, y+h)))
        return mask
    
    def show_bbox(self, data):
        # img: W,H,C
        '''
        show_bbox(data[2][0], data[2][4])
        '''
        print(data[3])
        img, mask, C, bbox = data[0], data[1], data[2], data[4]
        x, y, w, h = [round(i) for i in bbox]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 1), 2)
        print('img rectange: {},{}'.format((x,y), (x+w, y+h)))
        plt.subplot(221)
        plt.imshow(img)
        
        plt.subplot(222)
        plt.imshow(mask[0])
        
    def mask2bbox(self, mask):
        assert len(mask.shape)==2
        # mask: [W,H]
        row = mask.argmax(axis=1)
        col = mask.argmax(axis=0)
        W = len(col[col==col.max()])
        H = len(row[row==row.max()])
        return row.max(), col.max(), W, H

if __name__ == '__main__':
    data = Jiu()
    data.show_bbox(data[16])