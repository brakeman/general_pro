import paddle.fluid as fluid
import paddle

def conv_pool_layer(
               input,
               num_filters,
               filter_size,
               stride,
               groups=1,
               act=None,
               name=None):
    # input: bs, channels, h,w                
    channels = input.shape[1]
    print('chnnels:{}'.format(channels))
    param_attr = fluid.param_attr.ParamAttr(initializer=fluid.initializer.MSRAInitializer(), name=name + "_weights")
        
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding= 0,
        groups=groups,
        act=act,
        param_attr=param_attr,
        bias_attr=False,
        name=name)
        
    pool = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=0)  # 3*3 kernel
    return pool

def alex_net(data):
    # data = fluid.layers.data(name='x', shape=[3,100,100], dtype='float32')
    # label = fluid.layers.data(name="y", shape=[1])
    conv1 = conv_pool_layer(input=data, num_filters=64, filter_size=3, stride=1, name='conv1')
    conv2 = conv_pool_layer(input=conv1, num_filters=192, filter_size=3, stride=1, name='conv2')
    conv3 = conv_pool_layer(input=conv2, num_filters=384, filter_size=3, stride=1, name='conv3')
    conv4 = conv_pool_layer(input=conv3, num_filters=256, filter_size=3, stride=1, name='conv4')
    print('conv4 shape:{}'.format(conv4.shape))
    # conv5 = conv_pool_layer(input=conv4, num_filters=256, filter_size=3, stride=1, name='conv5')
    drop1 = fluid.layers.dropout(x=conv4, dropout_prob=0.5)
    fc1 = fluid.layers.fc(drop1, size=100, act='relu')
    drop2 = fluid.layers.dropout(x=fc1, dropout_prob=0.5)
    fc2 = fluid.layers.fc(drop2, size=100, act='relu')
    drop3 = fluid.layers.dropout(x=fc2, dropout_prob=0.5)
    # logit = fluid.layers.fc(drop3, size=1, act='sigmoid')
    return drop3
    # loss = fluid.layers.cross_entropy(input=logit, label=label)
    # loss = fluid.layers.mean(loss)
    
    