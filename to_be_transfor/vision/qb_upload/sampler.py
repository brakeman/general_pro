import random
from tqdm import tqdm
from data import JiuData, do_resize, JiuTest
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

class JiuSampler(Sampler):
    '''
    每个batch给出一组index, 包含一定比例的 背景图片和其它图片'''

    def __init__(self, src_data, bs, back_ratio=0.125, max_batchs_per_epoch=None):
        self.backgroud_ids = [idx for idx, tu in enumerate(src_data.annotations) if tu['category_id']==0]
        self.other_ids = list(set(range(len(src_data))) - set(self.backgroud_ids))
        self.bs = bs  
        if max_batchs_per_epoch is None:
            self.max_batchs_per_epoch = len(src_data)//bs
        else:
            self.max_batchs_per_epoch = max_batchs_per_epoch
        self.back_num = int(back_ratio*bs)
        self.other_num = bs - self.back_num 
        
    def __iter__(self):
        final_batchs = []
        num_batch = 0
#         ipdb.set_trace()
        while num_batch <= self.max_batchs_per_epoch:
            batch_idx = []
            num_batch+=1
            a1 = random.sample(self.backgroud_ids, self.back_num)
            a2 = random.sample(self.other_ids, self.other_num)
            batch_idx.extend(a1)
            batch_idx.extend(a2)
            final_batchs.extend(batch_idx)
        return iter(final_batchs)
    
    def __len__(self):
        return self.max_batchs_per_epoch
    
    
if __name__ == '__main__':
    fold_id = 1
    batch_size = 20
    train_data = JiuData(fold_id=fold_id, mode='train', return_ori_img=False)
    train_loader = DataLoader(
                        train_data,
                        sampler=JiuSampler(train_data, bs=batch_size, max_batchs_per_epoch=300),
                        batch_size=batch_size,
                        num_workers=8,
                        pin_memory=True)
    
    cnt = 0
    for i in train_loader:
        cnt+=1
        print(cnt)