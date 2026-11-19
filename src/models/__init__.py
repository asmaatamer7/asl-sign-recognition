from .gru import RNNClassifier
from .transformer import SignTransformer
from .stgcn import STGCN


def build_model(cfg, num_classes, in_channels, J=49, T=64):
    m = dict(cfg["model"]); name = m.pop("name")
    if name == "gru":
        return RNNClassifier(J, in_channels, num_classes, **m)
    if name == "transformer":
        return SignTransformer(J, in_channels, num_classes, T=T, **m)
    if name == "stgcn":
        return STGCN(J, in_channels, num_classes, **m)
    raise ValueError(f"unknown model {name}")
