"""ST-GCN style model. The spatial graph convolution is implemented FROM SCRATCH."""
import torch, torch.nn as nn
from .graph import adjacency


class GraphConv(nn.Module):
    """y = sum_k  A_k (.) M_k  X W_k   with learnable edge-importance masks M_k."""

    def __init__(self, cin, cout, A):
        super().__init__()
        self.K = A.shape[0]
        self.cout = cout
        self.conv = nn.Conv2d(cin, cout * self.K, 1)
        self.register_buffer("A", torch.as_tensor(A))
        self.M = nn.Parameter(torch.ones_like(self.A))

    def forward(self, x):                                       # (B,C,T,J)
        B, _, T, J = x.shape
        y = self.conv(x).view(B, self.K, self.cout, T, J)
        return torch.einsum("bkctj,kjw->bctw", y, self.A * self.M)


class STBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, dropout=0.2, tk=9):
        super().__init__()
        self.gcn = GraphConv(cin, cout, A)
        self.bn1 = nn.BatchNorm2d(cout)
        self.tcn = nn.Sequential(nn.Conv2d(cout, cout, (tk, 1), (stride, 1), (tk // 2, 0)),
                                 nn.BatchNorm2d(cout), nn.Dropout(dropout))
        self.res = (nn.Identity() if cin == cout and stride == 1 else
                    nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout)))
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.tcn(self.act(self.bn1(self.gcn(x))))
        return self.act(y + self.res(x))


class STGCN(nn.Module):
    def __init__(self, J, C, num_classes, width=64, blocks=4, dropout=0.2):
        super().__init__()
        A = adjacency(J)
        self.data_bn = nn.BatchNorm1d(J * C)
        chans = [width] * (blocks // 2) + [width * 2] * (blocks - blocks // 2)
        strides = [1] * blocks
        strides[blocks // 2] = 2
        layers, cin = [], C
        for c, s in zip(chans, strides):
            layers.append(STBlock(cin, c, A, s, dropout)); cin = c
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, num_classes))

    def forward(self, x):                                       # (B,T,J,C)
        B, T, J, C = x.shape
        x = self.data_bn(x.reshape(B, T, J * C).transpose(1, 2))  # (B, J*C, T)
        x = x.view(B, J, C, T).permute(0, 2, 3, 1)               # (B,C,T,J)
        return self.head(self.layers(x).mean((2, 3)))
