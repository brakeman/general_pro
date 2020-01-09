## pytorch 可视化工具 visdom

	from visdom import Visdom
	vis = Visdom(env='jiu')

0. 服务器端 打开服务
	
	!python -m visdom.server

1. 展示图片

	def valid(loader, model, dataset, device):
	    model = model.to(device)
	    running_loss = 0.
	    model.eval()
	    for imgs, masks, cls, bbox, H, W in progress_bar(loader, parent=mb):    
	#     for imgs, masks, cls, bbox, H, W in loader:
	        imgs, masks = imgs.to(device), masks.to(device)
	        with torch.no_grad():
	            logits = model(imgs)
	            loss1 = multi_class_entropy(logits, masks.squeeze().int())
	            loss2 = lovasz_softmax(logits.squeeze(), masks.squeeze().int(), per_image=False)
	            loss = loss1 + loss2
	        running_loss += loss.item()*imgs.size(0)
	    vis.images(imgs[:5], nrow=5, win='valid_imgs',opts={'title':'imgs'})
	    vis.images(masks[:5], nrow=5, win='valid_masks',opts={'title':'imgs'})
	    return running_loss/len(dataset)


2. epoch 俩 loss 即时更新；

	for epoch in mb:
	    epoch_loss = train(train_loader, model, train_data, device)
	    epoch_val_loss = valid(valid_loader, model, valid_data, device)

	    vis.line(X=[epoch], Y=[[epoch_loss, epoch_val_loss]], opts=dict(markers=False),	win='loss', update='append' if epoch>0 else None)


