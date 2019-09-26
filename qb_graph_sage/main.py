import tensorflow as tf
import numpy as np

class TRFM(object):
    def __init__(self, X, Y, dim_k, start, end, dim_v, lambda_12, heads, layers, dm, fy, tx, ty, bs_for_training, num_batch_inter, dropout, GD_clipping):
        self.X = X # bs,fx
        self.Y = Y # bs,fy
        self.fx = X.shape[1]
        self.fy = fy
        self.tx = tx
        self.lambda_l2 = lambda_12
        self.ty =ty
        self.bs = bs_for_training
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.heads = heads
        self.layers = layers
        self.dm = dm
        self.dropout = dropout
        self.GD_clipping = GD_clipping
        self.num_batch_inter = num_batch_inter
        self.start = start
        self.end = end

    def get_train(self, X, y, batch_size, inp_len, out_len, start):
        res = y.ILI比例 - y.shift(1).ILI比例
        y['res'] = res
        stop_idx = y[y.week == start].index.values[0]
        y.index = range(len(y))
        X.index = range(len(X))
        start = int(inp_len/7) - out_len
        points_idx = np.random.choice(range(start, stop_idx), batch_size, replace=False)
        out_idxs = [list(range(i, i+out_len)) for i in points_idx]
        yy =  np.take(y.res.values, indices=out_idxs, axis=0)
        inp_idxs= [list(range(i*7-inp_len, i*7) for i in points_idx)]
        x = np.take(X.values, indices=inp_idxs, axis=0)
        return x, yy

    def get_test_label(self):
        y=self.Y
        start_idx = y[y.week==self.start].index.values[0]
        end_idx = y[y.week==self.end].index.values[0]
        return y.iloc[start_idx: end_idx+1]

    def get_test(self,start, end, X, y, inp_len, out_len):
        '''there is only one batch'''
        y.index = range(len(y))
        X.index = range(len(X))
        res = y.ILI比例 - y.shift(1).ILI比例
        y['res'] = res
        start_idx = y[y.week == start].index.values[0]
        end_idx = y[y.week == end].index.values[0]
        points_idxs =  list(range(start_idx, end_idx+1))
        out_idxs = [list(range(i, i+out_len)) for i in points_idxs]
        yy =  np.take(y.res.values, indices=out_idxs, axis=0)
        inp_idxs= [list(range(i*7-inp_len, i*7) for i in points_idxs)]
        x = np.take(X.values, indices=inp_idxs, axis=0)
        to_add= y.iloc[start_idx-1:end_idx]
        return x, yy, to_add

    def plot(self, y_hat, y_label, path, name, title, save=False):
        import matplotlib.pyplot as plt
        figsize = 8,9
        mse=1
        plt.subplot(figsize=figsize)
        ax1 = plt.subplot(1,1,1)
        x = range(1, len(y_hat)+1)
        plt.plot(x, y_hat, color ='black', label='$predict$', linewidth=0.8)
        plt.ylim(0, 5)
        h1 = plt.legend(loc='upper right', frameon=False)
        plt.plot(x, y_label, color = 'red', label = '$label$', linewidth=0.8)
        plt.legend('upper right', frameon = False)
        plt.xlim(0, len(y_hat)+2)
        plt.title(r'''{}, MSE:{}'''.format(title,mse))
        if save:
            plt.savefig(path+name+'.jpg')

    def get_ph(self):
        x = tf.placeholder(tf.float32, [None, self.tx, self.fx])
        y = tf.placeholder(tf.float32, [None, self.ty, self.fy])
        go = tf.placeholder(tf.float32, [None, 1, self.fy])
        return x, y, go

    def add_pos_emb(self, x, ts, dim):
        '''x: bs,ts,f
           ts: scalar, seq_len
           dim: scalar
           return: given a batch x: need to construct its pos emb with same tensor shape'''
        pos = positional_encoding(dim=dim, ts=ts)
        return tf.map_fn(lambda xx: xx+pos, x)

    def build_encoder(self, enc_in):
        '''enc_in: placeholder with shape(bs, tx, fx)
           return: bs,tx,dm'''
        with tf.variable_scope('encoder'):
            encoder = Encoder(dropout=self.dropout, d_m= self.dm, dim_k =self.dim_k, dim_v=self.dim_v, ffn_dim = self.dm, num_heads=self.heads, num_layers=self.layers)
            return encoder.build(enc_in)

    def build_decoder(self, dec_in, enc_out):
        '''dec_in: in this version, bs, ty=1, fy=1
           enc_out: bs, tx, dm
           return: bs, ty=1, dm'''
        with tf.variable_scope('decoder'):
            decoder = Decoder(dropout=self.dropout, d_m= self.dm, dim_k =self.dim_k, dim_v=self.dim_v, ffn_dim = self.dm, num_heads=self.heads, num_layers=self.layers)
            return decoder.build(dec_in, enc_out)

    def build_outputs(self, dec_out):
        '''dec_out: for this version, bs, ty=1, fy=1
           return : bs,1 '''
        with tf.variable_scope('outputs'):
            return tf.layers.dense(dec_out, 1) # bs, 1, 1

    def get_bs_loss_Alex(self, logits, label, sigma1, sigma2):
        '''logits: bs, 1
           label: bs,1
           return batch mean comp loss as mentioned in 2017_Alex_paper'''
        shape_a = logits.get_shape().as_list()
        if len(shape_a) == 3:
            logits = tf.reshape(logits, [-1,1])
        shape_b = label.get_shape().as_list()
        if len(shape_b) == 3:
            label = tf.reshape(label, [-1,1])
        def mse(x):
            return tf.reduce_mean(tf.pow(x[0]-x[1], 2))

        # regulizer:
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        # v1: add regulizer
        res = tf.reduce_mean(tf.map_fn(mse, [logits, label], dtype=tf.float32)) + reg_loss*self.lambda_l2
        entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=tf.sign(label+1e-14))) +reg_loss*self.lambda_l2

        total = (1/(2*tf.pow(sigma1,2)))*res + tf.log(sigma1+1e-14) + (1/tf.pow(sigma2, 2))*entropy + tf.log(sigma2+1e-14)
        print('''reg_mse:{}, clf_entropy:{}, total_loss:{}, sigma_1:{}. sigma_2:{}'''.format(res, entropy, total, sigma1, sigma2))
        return res, entropy, total

    def get_bs_loss(self, logits, label, e=1):
        '''logits: bs, 1
            label: bs,1
            return batch mean comp loss'''
        shape_a = logits.get_shape().as_list()
        if len(shape_a) == 3:
            logits = tf.reshape(logits, [-1, 1])
        shape_b = label.get_shape().as_list()
        if len(shape_b) == 3:
            label = tf.reshape(label, [-1, 1])

        def mse(x):
            return tf.reduce_mean(tf.pow(x[0] - x[1], 2))

        # regulizer:
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        # v1: add regulizer
        res = tf.reduce_mean(tf.map_fn(mse, [logits, label], dtype=tf.float32)) + reg_loss * self.lambda_l2
        entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.sign(
            label + 1e-14))) + reg_loss * self.lambda_l2
        total = res+entropy*e
        print('''reg_mse:{}, clf_entropy:{}, total_loss:{}'''.format(res, entropy, total))
        return res, entropy, total

    def get_acc(self, logits, label2):
        '''logits: bs,1
           label2: bs,1
           u definitely need it when try comp loss'''
        y1= tf.sign(logits)
        y2 =tf.sign(label2)
        corr = tf.equal(y1, y2)
        return tf.reduce_mean(tf.cast(corr, tf.float32))

    def main_alex(self, settings):
        '''function for graph, training, eval'''
        RESULT = []
        for (num_epochs, lr) in settings:
            train_loss_list, test_loss_list = [], []
            tf.reset_default_graph()
            gb_step= tf.Variable(initial_value=0, trainable=False)
            source_enc_inp, target_enc_inp, go = self.get_ph()
            source_enc_inp = self.add_pos_emb(source_enc_inp, ts=self.tx, dim=self.fx) # pos emb. return bs, tx, fx
            enc_out = self.build_encoder(source_enc_inp)
            ######### encoder part finish, decoder part start ##############
            target_inp = go
            outputs = []
            # in case of multi_step inference, to be done when u know how to do inference in any other way
            for t in range(self.ty):
                with tf.variable_scope('rooling{}'.format(t)):
                    dec_out = self.build_decoder(target_inp, enc_out) # bs, ty=1, 1
                    logits = self.build_outputs(dec_out) # bs, ty=1, 1
                    outputs.append(logits) # ty, bs, 1
                    target_inp = dec_out
            sig_init = tf.constant(np.random.rand())
            sigma1 = tf.get_variable('Sigma_1', initializer=sig_init)
            sigma2 = tf.get_variable('Sigma_2', initializer=sig_init)

            reg_loss, clf_loss, loss = self.get_bs_loss_Alex(logits=logits, label=target_enc_inp, sigma1=sigma1, sigma2=sigma2)
            optimizer = tf.contrib.layers.optimize_loss(loss=loss, learning_rate=lr, global_step=gb_step, optimizer='Adam', clip_gradients = self.GD_clipping)

            # acc=self.get_bs_acc()
            ########## graph finish, training start############
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                epoch = 0
                while epoch < num_epochs:
                    epoch +=1
                    print('''epoch_{} start'''.format(epoch-1))
                    for i in range(self.num_batch_inter):
                        bs_in, bs_out = self.get_train(X=self.X, y=self.Y, batch_size=self.bs, inp_len=self.tx, out_len=self.ty, start=self.start)
                        bs_out = np.expand_dims(bs_out, axis=-1)
                        batch_go = np.zeros_like(bs_out)
                        sess.run(optimizer, feed_dict={source_enc_inp: bs_in, target_enc_inp:bs_out, go:batch_go})
                        if i % 5 == 0:
                            print('batch_idx_for_training:{}'.format(i))
                            # generate a big batch;
                            big_x, big_y, _ = self.get_test(start = 201251, end=self.start, X=self.X, y=self.Y, inp_len=self.tx, out_len=self.ty)
                            big_y = np.expand_dims(big_y, axis=-1)
                            big_go = np.zeros_like(big_y)
                            train_loss = sess.run(loss, feed_dict={source_enc_inp: big_x, target_enc_inp: big_y, go: big_go})
                            train_loss_list.append(train_loss)

                            test_x, test_y, to_add = self.get_test(start=self.start, end=self.end, X=self.X, y=self.Y,
                                                            inp_len=self.tx, out_len=self.ty)
                            test_y = np.expand_dims(test_y, axis=-1)
                            test_go = np.zeros_like(test_y)

                            test_pred = sess.run(logits, feed_dict={source_enc_inp: test_x, target_enc_inp: test_y, go: test_go})

                            test_reg_loss, test_clf_loss, test_loss = sess.run([reg_loss,clf_loss,loss], feed_dict={source_enc_inp: test_x, target_enc_inp: test_y, go: test_go})
                            sigma_a = sess.run(sigma1)
                            sigma_b = sess.run(sigma2)
                            test_loss_list.append(test_loss)
                            print('train_loss:{}, test_res_loss:{}, test_clf_loss:{}, test_loss:{}, sigma1:{}, sigma2:{}'.format(train_loss, test_reg_loss, test_clf_loss,test_loss, sigma1, sigma2))
            RESULT.append([train_loss_list, test_loss_list, test_pred, to_add])
        return RESULT
