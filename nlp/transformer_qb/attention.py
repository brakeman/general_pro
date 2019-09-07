import numpy as np
import tensorflow as tf



__all__ = ["positional_encoding", "Attention"]


def positional_encoding(dim, ts, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(ts) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([ts, dim]), dtype=dtype)


class Attention:
    """Attention class"""
    def __init__(self,
                 num_heads=1,
                 masked=False,
                 dim_k=50,
                 dim_v=50,
                 d_m=100,
                 dropout=0.2):

        assert dim_k % num_heads == 0
        assert dim_v % num_heads == 0

        self.num_heads = num_heads
        self.masked = masked
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.model_dim = d_m
        self.dropout = dropout

    def multi_head(self, q, k, v):
        with tf.variable_scope('multi_head'):
            q, k, v = self._linear_projection(q, k, v)
            qs, ks, vs = self._split_heads(q, k, v)
            outputs = self._scaled_dot_product(qs, ks, vs)
            output = self._concat_heads(outputs)
            output = tf.layers.dense(output, self.model_dim)
            return tf.nn.dropout(output, 1.0 - self.dropout)

    def _linear_projection(self, q, k, v):
        with tf.variable_scope('linear'):
            q = tf.layers.dense(q, self.dim_k, use_bias=False, name='q')
            k = tf.layers.dense(k, self.dim_k, use_bias=False, name='k')
            v = tf.layers.dense(v, self.dim_v, use_bias=False, name ='v')
            return q, k, v

    def _split_heads(self, q, k, v):

        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1, t_shape[1], num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.dim_k)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.dim_k)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.dim_v)

        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs):
        key_dim_per_head = self.dim_k // self.num_heads

        o1 = tf.matmul(qs, ks, transpose_b=True)
        o2 = o1 / (key_dim_per_head**0.5)

        if self.masked:
            diag_vals = tf.ones_like(o2[0, 0, :, :]) # (batch_size, num_heads, query_dim, key_dim)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (q_dim, k_dim)
            masks = tf.tile(tf.reshape(tril, [1, 1] + tril.get_shape().as_list()),
                            [tf.shape(o2)[0], tf.shape(o2)[1], 1, 1])
            paddings = tf.ones_like(masks) * -1e9
            o2 = tf.where(tf.equal(masks, 0), paddings, o2)
        o3 = tf.nn.softmax(o2)
        return tf.matmul(o3, vs)

    def _concat_heads(self, outputs):

        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

        return transpose_then_concat_last_two_dimenstion(outputs)
