
import tensorflow as tf

from .attention import Attention
from .layer import FFN



class Decoder:
    """Decoder class"""

    def __init__(self,
                 num_layers=8,
                 num_heads=8,
                 dim_k=50,
                 dim_v=50,
                 d_m=50,
                 ffn_dim=50,
                 dropout=0.2):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.d_m = d_m
        self.ffn_dim = ffn_dim
        self.dropout = dropout


    def build(self, decoder_inputs, encoder_outputs):
        o1 = tf.identity(decoder_inputs)
        for i in range(1, self.num_layers+1):
            print(i, 'hehe')
            with tf.variable_scope(f"layer-{i}-self_att"):
                # no mask since never use future info no matter in training or test process
                o2 = self._add_and_norm(o1, self._masked_self_attention(q=o1,
                                                                            k=o1,
                                                                            v=o1), num=1)
            with tf.variable_scope(f"layer-{i}-enc_dec"):
                o3 = self._add_and_norm(o2, self._encoder_decoder_attention(q=o2,
                                                                            k=encoder_outputs,
                                                                            v=encoder_outputs), num=2)
            with tf.variable_scope(f"layer-{i}-ffn"):
                o4 = self._add_and_norm(o3, self._positional_feed_forward(o3), num=3)
                o1 = tf.identity(o4)
        return o4


    def _masked_self_attention(self, q, k, v):
        with tf.variable_scope("masked-self-attention",reuse=False):
            attention = Attention(num_heads=self.num_heads,
                                    masked=False,  # Not implemented yet
                                    dim_k=self.dim_k,
                                    dim_v=self.dim_v,
                                    d_m=self.d_m,
                                    dropout=self.dropout)
            return attention.multi_head(q, k, v)


    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.name_scope(f"add-and-norm--{num}"):
            return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection


    def _encoder_decoder_attention(self, q, k, v):
        with tf.variable_scope("encoder-decoder-attention",reuse=False):
            attention = Attention(num_heads=self.num_heads,
                                    masked=False,
                                    dim_k=self.dim_k,
                                    dim_v=self.dim_v,
                                    d_m=self.d_m,
                                    dropout=self.dropout)
            return attention.multi_head(q, k, v)


    def _positional_feed_forward(self, output):
        with tf.variable_scope("feed-forward"):
            ffn = FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.d_m,
                      dropout=self.dropout)
            return ffn.dense_relu_dense(output)