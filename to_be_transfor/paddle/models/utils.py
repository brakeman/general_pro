import paddle
from PIL import Image

def load_image(path):
    img = paddle.dataset.image.load_and_transform(path, 100, 100, False).astype('float32')#img.shape是(3, 100, 100)
    img = img / 255.0 
    return img
    
def to_rgb_img(read_path, save_path=None):
    im = Image.open(read_path)
    im_new = im.convert('RGB')
    if save_path:
        im_new.save(save_path)
    return im_new
    
# subprocess.call('mkdir /home/aistudio/work/task3/train/fake_revert', shell=True)
# ERR = []
# for sample in fake_samples:
#     path = sample[0]
#     name = path.split('.')[0].split('/')[-1]
#     try:
#         load_image(path)
#     except:
#         try:
#             to_rgb_img(path, '/home/aistudio/work/task3/train/fake_revert/'+name+'.jpg')
#         except:
#             ERR.append(path)


# for sample in truth_samples:
#     path = sample[0]
#     name = path.split('.')[0].split('/')[-1]
#     try:
#         load_image(path)
#     except:
#         print('deleting ...{}'.format(path))
#         os.remove(path)