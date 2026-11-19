import torch, torch.nn as nn


class RNNClassifier(nn.Module):
    """Classic recurrent model: frame embedding -> (Bi)GRU/LSTM -> temporal mean pooling -> linear."""

    def __init__(self, J, C, num_classes, hidden=128, layers=2, dropout=0.3, cell="gru", bidirectional=True):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(J * C, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(hidden, hidden, layers, batch_first=True, bidirectional=bidirectional,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * (2 if bidirectional else 1), num_classes))

    def forward(self, x):                      # x (B,T,J,C)
        h, _ = self.rnn(self.proj(x.flatten(2)))
        return self.head(h.mean(1))
