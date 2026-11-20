"""ST-GCN style model. The spatial graph convolution is implemented FROM SCRATCH (custom Keras layer).
Layout is channels-last: (B, T, J, C)."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .graph import adjacency


class GraphConv(layers.Layer):
    """y = sum_k  (A_k * M_k)^T-mixing of  X W_k   with learnable edge-importance masks M_k."""

    def __init__(self, cout, A, **kw):
        super().__init__(**kw)
        self.cout, self.K = cout, A.shape[0]
        self.A = tf.constant(A, tf.float32)
        self.M = self.add_weight(name="edge_importance", shape=A.shape, initializer="ones", trainable=True)
        self.proj = layers.Dense(cout * self.K)                    # 1x1 conv over channels

    def call(self, x):                                              # (B,T,J,C)
        B, T, J = tf.shape(x)[0], tf.shape(x)[1], x.shape[2]
        y = tf.reshape(self.proj(x), (B, T, J, self.K, self.cout))
        return tf.einsum("btjkc,kjw->btwc", y, self.A * self.M)    # aggregate neighbours per subset k


class STBlock(layers.Layer):
    def __init__(self, cout, A, stride=1, dropout=0.2, tk=9, residual_proj=False, **kw):
        super().__init__(**kw)
        self.gcn, self.bn1 = GraphConv(cout, A), layers.BatchNormalization(momentum=0.9)
        self.tcn = layers.Conv2D(cout, (tk, 1), strides=(stride, 1), padding="same")
        self.bn2, self.drop = layers.BatchNormalization(momentum=0.9), layers.Dropout(dropout)
        self.res = None
        if residual_proj or stride != 1:
            self.res = [layers.Conv2D(cout, 1, strides=(stride, 1)), layers.BatchNormalization(momentum=0.9)]

    def call(self, x, training=False):
        y = tf.nn.relu(self.bn1(self.gcn(x), training=training))
        y = self.drop(self.bn2(self.tcn(y), training=training), training=training)
        r = x if self.res is None else self.res[1](self.res[0](x), training=training)
        return tf.nn.relu(y + r)


class STGCN(tf.keras.Model):
    def __init__(self, J, C, num_classes, width=64, blocks=4, dropout=0.2, **kw):
        super().__init__()
        A = adjacency(J)
        self.data_bn = layers.BatchNormalization(momentum=0.9)
        chans = [width] * (blocks // 2) + [width * 2] * (blocks - blocks // 2)
        strides = [1] * blocks; strides[blocks // 2] = 2
        self.blocks, cin = [], C
        for c, s in zip(chans, strides):
            self.blocks.append(STBlock(c, A, s, dropout, residual_proj=(c != cin))); cin = c
        self.drop, self.head = layers.Dropout(dropout), layers.Dense(num_classes)

    def call(self, x, training=False):                              # (B,T,J,C)
        B, T, J, C = tf.shape(x)[0], tf.shape(x)[1], x.shape[2], x.shape[3]
        x = tf.reshape(self.data_bn(tf.reshape(x, (B, T, J * C)), training=training), (B, T, J, C))
        for b in self.blocks:
            x = b(x, training=training)
        return self.head(self.drop(tf.reduce_mean(x, axis=(1, 2)), training=training))
