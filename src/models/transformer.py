"""Transformer encoder for pose sequences. Multi-head self-attention is implemented FROM SCRATCH
(no keras MultiHeadAttention layer) - this is the 'component written by ourselves' of requirement #5."""
import math
import tensorflow as tf
from tensorflow.keras import layers


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d, heads, dropout=0.0, **kw):
        super().__init__(**kw)
        assert d % heads == 0
        self.d, self.h, self.dk = d, heads, d // heads
        self.qkv, self.out, self.drop = layers.Dense(3 * d), layers.Dense(d), layers.Dropout(dropout)

    def call(self, x, training=False):                              # (B,T,D)
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        qkv = tf.reshape(self.qkv(x), (B, T, 3, self.h, self.dk))
        q, k, v = [tf.transpose(qkv[:, :, i], (0, 2, 1, 3)) for i in range(3)]   # (B,h,T,dk)
        att = tf.matmul(q, k, transpose_b=True) / math.sqrt(self.dk)           # scaled dot-product
        att = self.drop(tf.nn.softmax(att, axis=-1), training=training)
        o = tf.reshape(tf.transpose(tf.matmul(att, v), (0, 2, 1, 3)), (B, T, self.d))
        return self.out(o)


class Block(layers.Layer):
    def __init__(self, d, heads, mlp_ratio, dropout, attn=True, **kw):
        super().__init__(**kw)
        self.use_attn = attn
        self.ln1, self.ln2 = layers.LayerNormalization(), layers.LayerNormalization()
        self.attn = MultiHeadSelfAttention(d, heads, dropout) if attn else None
        self.fc1, self.fc2 = layers.Dense(d * mlp_ratio), layers.Dense(d)
        self.d1, self.d2, self.d3 = layers.Dropout(dropout), layers.Dropout(dropout), layers.Dropout(dropout)

    def call(self, x, training=False):
        if self.use_attn:                                           # pre-LN residual blocks
            x = x + self.d1(self.attn(self.ln1(x), training=training), training=training)
        h = self.d3(self.fc2(self.d2(tf.nn.gelu(self.fc1(self.ln2(x))), training=training)), training=training)
        return x + h


class SignTransformer(tf.keras.Model):
    def __init__(self, J, C, num_classes, T=64, d_model=128, heads=4, layers_n=3, mlp_ratio=2,
                 dropout=0.2, attn=True, **kw):
        layers_n = kw.pop("layers", layers_n)
        super().__init__()
        self.T, self.d = T, d_model
        self.embed, self.embed_ln = layers.Dense(d_model), layers.LayerNormalization()
        self.pos = self.add_weight(name="pos", shape=(1, T, d_model),
                                   initializer=tf.keras.initializers.TruncatedNormal(0.02), trainable=True)
        self.mask_token = self.add_weight(name="mask_token", shape=(d_model,), initializer="zeros", trainable=True)
        self.blocks = [Block(d_model, heads, mlp_ratio, dropout, attn) for _ in range(layers_n)]
        self.norm = layers.LayerNormalization()
        self.drop, self.head = layers.Dropout(dropout), layers.Dense(num_classes)

    def encode(self, x, frame_mask=None, training=False):           # x (B,T,J,C); frame_mask (B,T) bool
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        h = self.embed_ln(self.embed(tf.reshape(x, (B, T, -1))))
        if frame_mask is not None:
            h = tf.where(frame_mask[..., None], self.mask_token, h)
        h = h + self.pos
        for b in self.blocks:
            h = b(h, training=training)
        return self.norm(h)

    def call(self, x, training=False):
        return self.head(self.drop(tf.reduce_mean(self.encode(x, training=training), 1), training=training))
