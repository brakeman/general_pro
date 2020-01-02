DataDir = '../data1/'
DataListDir = '../new_whale/pic_list/'
from torch.utils.data.dataset import Dataset
import imgaug.augmenters as iaa
import numpy as np
import random
import json
import cv2
import os
import torch
import matplotlib.pyplot as plt

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    original = np.array([[0, 0],
                         [image.shape[1] - 1, 0],
                         [image.shape[1] - 1, image.shape[0] - 1],
                         [0, image.shape[0] - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(original, rect)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return warped


def Perspective_aug(img,   threshold1 = 0.25, threshold2 = 0.75):
    # img = cv2.imread(img_name)
    rows, cols, ch = img.shape

    x0,y0 = random.randint(0, int(cols * threshold1)), random.randint(0, int(rows * threshold1))
    x1,y1 = random.randint(int(cols * threshold2), cols - 1), random.randint(0, int(rows * threshold1))
    x2,y2 = random.randint(int(cols * threshold2), cols - 1), random.randint(int(rows * threshold2), rows - 1)
    x3,y3 = random.randint(0, int(cols * threshold1)), random.randint(int(rows * threshold2), rows - 1)
    pts = np.float32([(x0,y0),
                      (x1,y1),
                      (x2,y2),
                      (x3,y3)])

    warped = four_point_transform(img, pts)

    x_ = np.asarray([x0, x1, x2, x3])
    y_ = np.asarray([y0, y1, y2, y3])

    min_x = np.min(x_)
    max_x = np.max(x_)
    min_y = np.min(y_)
    max_y = np.max(y_)

    warped = warped[min_y:max_y,min_x:max_x,:]
    return warped

def aug_image(image):
    seq = iaa.Sequential([
        iaa.Affine(rotate= (-15, 15),
                   shear = (-15, 15),
                   mode='edge'),

        iaa.SomeOf((0, 2),
                   [
                       iaa.GaussianBlur((0, 1.5)),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
                       iaa.AddToHueAndSaturation((-5, 5)),  # change hue and saturation
                       iaa.PiecewiseAffine(scale=(0.01, 0.03)),
                       iaa.PerspectiveTransform(scale=(0.01, 0.1))
                   ],
                   random_order=True)])
    image = seq.augment_image(image)
    return image

class WhaleData(Dataset):
    '''
    train: read--(augment)--(flip_fake_label)--return;
    valid: read--(flip_fake_label)--return;
    test: read--return;
    '''
    
    def __init__(self, mode='train', fold_id=0, image_size=(128,256), augment=False, flip_label=False):
        super(WhaleData, self).__init__()
        assert fold_id in [0,1,2,3,4,5]
        assert mode in ['train', 'test', 'valid']
        self.pic_dir = '{}/train/'.format(DataDir)
        self.mode = mode
        self.image_size = image_size
        self.augment = augment
        self.flip_label = flip_label
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
            img = jpeg.JPEG(image_path).decode()
            image = cv2.imread(image_path, 1)
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([-1, self.image_size[0], self.image_size[1]])
            image = image / 255.0
            return torch.FloatTensor(image)
        
        elif self.mode == 'train':
            if (index >= len(self.pic_list)) & (self.flip_label) : 
                origin_index = index - len(self.pic_list)
                pic_name= self.pic_list[origin_index].split(',')[0]
                pic_label = self.pic_list[origin_index].split(',')[1].split('\n')[0]
                pic_label = self.image_label_dict[pic_label]
            else:
                pic_name= self.pic_list[index].split(',')[0]
                pic_label = self.pic_list[index].split(',')[1].split('\n')[0]
                pic_label = self.image_label_dict[pic_label]                

            image_path = DataDir+'train/' + pic_name
            image = cv2.imread(image_path, 1)
            
            if self.augment:
                if random.randint(0,1) == 0:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if random.randint(0, 1) == 0:
                    image = Perspective_aug(image)
                    image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
                image = aug_image(image)
                
            if self.flip_label:
                if pic_label == 5004: # if is_new_whle
                    seq = iaa.Sequential([iaa.Fliplr(0.5)])
                    image = seq.augment_image(image)
                    
                elif index >= len(self.pic_list): # 
                    seq = iaa.Sequential([iaa.Fliplr(1.0)])
                    image = seq.augment_image(image)
                    pic_label += 5005

            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([-1, self.image_size[0], self.image_size[1]])
            image = image / 255.0
            return torch.FloatTensor(image), pic_label                
                
        elif self.mode == 'valid':   
            pic_name= self.pic_list[index].split(',')[0]
            pic_label = self.pic_list[index].split(',')[1].split('\n')[0]
            pic_label = self.image_label_dict[pic_label]
            image_path = DataDir+'train/' + pic_name
            image = cv2.imread(image_path, 1)
            
            if self.flip_label:
                seq = iaa.Sequential([iaa.Fliplr(1.0)])
                image = seq.augment_image(image)
                if pic_label != 5004: 
                    pic_label += 5005
                    
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([-1, self.image_size[0], self.image_size[1]])
            image = image / 255.0
            return torch.FloatTensor(image), pic_label
        
        else:
            raise Exception('mode not available')

    def __len__(self):
        if (self.flip_label) & (self.mode=='train'):
            return len(self.pic_list)*2 # 只有train+flip 才是2倍数据；
        return len(self.pic_list)

def pic_show(img):
    img2 = img.numpy()
    img2 = np.transpose(img2, (1, 2, 0))  # 把channel那一维放到最后
    plt.imshow(img2)
        
if __name__ == '__main__':
    WD = WhaleData(mode='train',augment=True, flip_label=False)
    pic_show(WD[1][0])
    
    
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
