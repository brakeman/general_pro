DataDir = '../data1/'
DataListDir = '../new_whale/pic_list/'
from torch.utils.data.dataset import Dataset
import numpy as np
import json
import cv2
import os
import torch


class WhaleData(Dataset):
    def __init__(self, mode='train', fold_id=0, image_size=(128,256)):
        super(WhaleData, self).__init__()
        assert fold_id in [0,1,2,3,4,5]
        assert mode in ['train', 'test', 'valid']
        self.pic_dir = '{}/train/'.format(DataDir)
        self.mode = mode
        self.image_size = image_size
        with open(DataListDir+'whale_dict.json', 'r') as f:
            self.image_label_dict = json.load(f)
        if mode == 'train':
            pic_list_path = '{}/train.txt'.format(DataListDir)
        elif mode == 'valid':
            pic_list_path = '{}/valid_{}.txt'.format(DataListDir, fold_id)
        elif mode == 'test':
            pic_list_path = '{}/test.txt'.format(DataListDir)
        with open(pic_list_path, 'r') as f:
#             self.pic_list = f.readlines()[:2**7]
            self.pic_list = f.readlines()
    
    def __getitem__(self, index):
        if self.mode== 'test':
            pic_name = self.pic_list[index].split('\n')[0]
            image_path = DataDir+'test/' + pic_name
            image = cv2.imread(image_path, 1)
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([-1, self.image_size[0], self.image_size[1]])
            image = image / 255.0
            return torch.FloatTensor(image)
        else:
            pic_name= self.pic_list[index].split(',')[0]
            pic_label = self.pic_list[index].split(',')[1].split('\n')[0]
            image_path = DataDir+'train/' + pic_name
            image = cv2.imread(image_path, 1)
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([-1, self.image_size[0], self.image_size[1]])
            image = image / 255.0
            return torch.FloatTensor(image), self.image_label_dict[pic_label]

    def __len__(self):
        return len(self.pic_list)
    
    
# # 1个训练集合, 5个验证集, 1个测试集 先存到本地；
# valid 40% new_whale, 60% 随机从 count>=2 里面采样；可以重复；
## train 必须拿到所有count==1的，且 其它每个id必须要有至少1个。剩下的是valid池子；

# z = pd.read_csv('../data1/train.csv')

# 构建 {id:[pics]}
# Dic = {}
# for i in range(z.shape[0]):
#     img, l = z.iloc[i].Image, z.iloc[i].Id
#     if l not in Dic:
#         Dic[l] = []
#         Dic[l].append(img)
#     else:
#         Dic[l].append(img)
        
# # 从Dic中采样train
# Train_list = []
# for i in Dic:
#     if i=='new_whale':
#         new_whale_train_list = random.sample(Dic[i], 5000)
#         Train_list.extend(new_whale_train_list)
#     else:
#         all_lis = Dic[i]
#         leng = len(all_lis)
#         if leng==1:
#             sample_num=1
#         elif leng>1:
#             sample_num = leng//2
#         else:
#             raise Exception('leng==0 not allowed')
#         sample_tra_list = random.sample(all_lis, sample_num)
#         Train_list.extend(sample_tra_list)

# # valid 池子 划分为5个
# Valid_list = list(set(z.Image) - set(Train_list)) 


# import os
# dir_path = '/home/qibo/all_project/vision/new_whale/pic_list2'
# if not os.path.exists(dir_path):
#     os.makedirs(dir_path)

# with open('../new_whale/pic_list2/train.txt', 'w+') as f:
#     for pic_name in Train_list:
#         l = z.set_index('Image').loc[pic_name].Id
#         f.write('{},{}\n'.format(pic_name,l))
    

# sub_num = len(Valid_list)//5
# ZZ = z.set_index('Image')
# for fold_id in range(5):
#     with open('../new_whale/pic_list2/valid_{}.txt'.format(fold_id), 'w+') as f:
#         sub_valid_list = random.sample(Valid_list, sub_num)
#         for pic_name in sub_valid_list:
#             l = ZZ.loc[pic_name].Id
#             f.write('{},{}\n'.format(pic_name,l))

# with open('../new_whale/pic_list2/valid_{}.txt'.format(6), 'w+') as f:
#     for pic_name in Valid_list:
#         l = ZZ.loc[pic_name].Id
#         f.write('{},{}\n'.format(pic_name,l))

# zz = pd.read_csv('../data1/sample_submission.csv')
# with open('../new_whale/pic_list2/test.txt', 'w+') as f:
#     for line in zz.Image:
#         f.write('{}\n'.format(line))
