
def lr_exp_list(min_lr=0.000001, max_lr=1.5, iters=300, mode='exp'):
    assert mode in ['exp', 'linear']
    if mode=='exp':
        lr_list = list(reversed([max_lr*0.915**i for i in range(iters)]))
    else:
        step = (lr_max-lr_min)/iters
        lr_list = [step*i for i in range(1,iters+1)]
    return lr_list

def lr_range_test(loader, model, dataset, device, iters=300):
    '''
    try to find best lr range;
    return batch_loss_list;
    '''
    batch_loss_list = []
    model = model.to(device)
    running_loss = 0.
    model.train()
    iter_id = 0 
    lr_list = lr_exp_list(iters=iters)
    vis = Visdom(env = 'lr_range_test')
    for imgs, masks, cls, bbox, H, W in progress_bar(loader):
        if iter_id <= iters:
            lr_tmp = lr_list[iter_id]
            iter_id += 1
            optimizer = torch.optim.SGD(params = model.parameters(), lr=lr_tmp)
            optimizer.zero_grad()
            imgs, masks, cls = imgs.to(device), masks.to(device), cls.to(device)
            with torch.set_grad_enabled(True):
                logits, cls_logits = model(imgs) # [bs, cls, H, W] [bs, cls]
                loss1 = multi_class_entropy(logits, masks.squeeze().int())
                loss2 = lovasz_softmax(logits.squeeze(), masks.squeeze().int(), per_image=False)
                loss3 = torch.nn.CrossEntropyLoss()(cls_logits, cls)
                loss = loss1 + loss2 + loss3
                loss.backward()
                optimizer.step()
            batch_loss_list.append([lr_tmp, loss.item()])
            vis.line(X=[lr_tmp], Y=[[loss.item(), loss1.item(), loss2.item(), loss3.item()]], opts=dict(markers=True, showlegend=True), win='loss', update='append')
        else:
            break
    return batch_loss_list

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import RandomSampler
    from data import JiuData
    from model import *
    from loss import multi_class_entropy, lovasz_softmax
    from fastprogress.fastprogress import master_bar, progress_bar
    from visdom import Visdom

    batch_size = 20
    model = Unet()

    device = 'cuda'
    fold_id = 1

    train_data = JiuData(fold_id=fold_id, mode='train', return_ori_img=False)
    train_loader = DataLoader(
                        train_data,
                        shuffle=RandomSampler(train_data),
                        batch_size=batch_size,
                        num_workers=8,
                        pin_memory=True)


    res = lr_range_test(train_loader, model, train_data, device, iters=300)
    # [0.00015, 0.01]


