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


def do_resize(image, mask, H, W):
    '''
    image: H,W,C;
    mask: H,W
    '''
#     data2[5922][0]
    image = cv2.resize(image,dsize=(W,H))
    mask = cv2.resize(mask,dsize=(W,H))
    mask  = (mask>0.5).astype(np.float32)
    return image, mask


    
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

    def get_big_img(self, data):
        img_id = [i['id'] for i in data.load_dict['images'] if i['height']!=492]
        index = [i for i,j in enumerate(data.load_dict['annotations']) if j['image_id'] in img_id]
        return index
    
    def show_bbox(self, data):
        # img: W,H,C
        '''
        show_bbox(data[2][0], data[2][4])
        '''
        print(data[3])
        img, mask, C, bbox = data[0], data[1], data[2], data[4]
        x, y, w, h = [round(i) for i in bbox]
        cv2.rectangle(img, (x, y), (x + w, y + h), (1, 0, 0), 5)
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

class Jiu2(Dataset):
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
        return len(self.load_dict['annotations'][:100])
        
    def __getitem__(self, index):
        # img, mask, C, bbox, h, w
        anno_dic = self.load_dict['annotations'][:100][index]
        img_id = anno_dic['image_id']
        H, W, C = self.tmp_img_dic[img_id]['height'], self.tmp_img_dic[img_id]['width'], anno_dic['category_id']
        mask_img = self.mask_from_bbox(W, H, C, bbox=anno_dic['bbox'])
        img_name = self.tmp_img_dic[img_id]['file_name']
        img = cv2.imread(img_path+img_name).astype(np.float32)/255
        return torch.from_numpy(img).permute([2,0,1]), torch.from_numpy(mask_img), C, torch.tensor(anno_dic['bbox']), H, W
    
    def mask_from_bbox(self, W, H, C, bbox):
        # 图片的x, y轴与np的 x,y轴不一致；需要交换坐标系x,y才能互相转换；
        mask = np.zeros((1, H, W))
        x, y, w, h = [round(i) for i in bbox]
        mask[0, y:y+h, x:x+w] = C
        return mask
    

annotation_path = "../chongq/chongqing1_round1_train1_20191223/annotations.json"
img_path = './chongqing1_round1_train1_20191223/images/'


class Jiu3(Dataset):
    
    def __init__(self, fold_id=None, mode='train'):
        from sklearn.model_selection import StratifiedShuffleSplit
        import json
        with open(annotation_path, 'r') as load_f:
            load_dict = json.load(load_f)
        ids = [i for i,j in enumerate(load_dict['annotations'])]
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
        img = cv2.imread(img_path+img_name).astype(np.float32) / 255
#         if 1: # just for human look;
#             x, y, w, h = [round(i) for i in anno_dic['bbox']]
#             cv2.rectangle(img, (x, y), (x + w, y + h), (1, 0, 0), 2)
#         ipdb.set_trace()
        img, mask_img = do_resize(img, mask_img.squeeze(), 256, 256)
        mask_img[mask_img==1]=C
        return torch.from_numpy(img).permute([2,0,1]), torch.from_numpy(mask_img).unsqueeze(0), C, torch.tensor(anno_dic['bbox']), H, W
    
    def mask_from_bbox(self, W, H, C, bbox):
        # cv2的x, y轴与np的x,y轴不一致；需要交换坐标系x,y才能互相转换；
        mask = np.zeros((1, H, W))
        x, y, w, h = [round(i) for i in bbox]
        mask[0, y:y+h, x:x+w] = C
        return mask

    def get_big_img(self):
        img_id = [i['id'] for i in self.load_dict['images'] if i['height']!=492]
        index = [i for i,j in enumerate(self.annotations) if j['image_id'] in img_id]
        return index
    
    def show_bbox(self, data):
        # img: W,H,C
        '''
        show_bbox(data[2][0], data[2][4])
        '''
        print(data[-1])
        img, mask, C, bbox = data[0], data[1], data[2], data[3]
#         x, y, w, h = [round(i) for i in bbox]
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 1), 2)
#         print('img rectange: {},{}'.format((x,y), (x+w, y+h)))
        plt.subplot(221)
        plt.imshow(img)
        
        plt.subplot(222)
        plt.imshow(mask[0])
        
    def mask2bbox(self, mask):
        assert len(mask.shape) in [2,3]
        # mask: [W,H]
        row = mask.argmax(axis=1)
        col = mask.argmax(axis=0)
        W = len(col[col==col.max()])
        H = len(row[row==row.max()])
        return row.max(), col.max(), W, H
    
    
    
if __name__ == '__main__':
    data = Jiu()
    data.show_bbox(data[16])
    
    data2 = Jiu2()