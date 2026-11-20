import tensorflow as tf
from .gru import RNNClassifier
from .transformer import SignTransformer
from .stgcn import STGCN


def build_model(cfg, num_classes, in_channels, J=49, T=64):
    m = dict(cfg["model"]); name = m.pop("name")
    if name == "gru":
        model = RNNClassifier(J, in_channels, num_classes, T=T, **m)
    elif name == "transformer":
        model = SignTransformer(J, in_channels, num_classes, T=T, **m)
    elif name == "stgcn":
        model = STGCN(J, in_channels, num_classes, **m)
    else:
        raise ValueError(f"unknown model {name}")
    model(tf.zeros((1, T, J, in_channels)), training=False)          # build all weights
    return model
