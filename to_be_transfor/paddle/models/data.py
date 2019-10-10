from models.utils import load_image
import random

def train_valid_split(all_samples, train_rate = 0.8):
    random.shuffle(all_samples)
    split_point = int(len(all_samples)*train_rate)
    train, valid = all_samples[:split_point], all_samples[split_point:]
    return train, valid
    
def image_reader_creator(all_samples, n):
    def reader():
        error_count = 0 
        for i in range(n):
            x, y = all_samples[i]
            try:
                img = load_image(x)
            except:
                error_count = error_count+1
                continue
            else:
                yield img, y # a single entry of data is created each time
        print('error:{}'.format(error_count))
    return reader


# reader = image_reader_creator(fake_revert_samples, len(fake_revert_samples))
# batch_reader = paddle.batch(reader, 10)