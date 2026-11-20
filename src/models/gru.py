import tensorflow as tf
from tensorflow.keras import layers


def RNNClassifier(J, C, num_classes, T=64, hidden=128, layers_n=2, dropout=0.3, cell="gru", bidirectional=True, **kw):
    """Classic recurrent model: frame embedding -> (Bi)GRU/LSTM stack -> temporal mean pooling -> linear."""
    layers_n = kw.pop("layers", layers_n)
    inp = layers.Input((T, J, C))
    x = layers.Reshape((T, J * C))(inp)
    x = layers.Dense(hidden)(x); x = layers.LayerNormalization()(x)
    x = layers.Activation(tf.nn.gelu)(x); x = layers.Dropout(dropout)(x)
    Rnn = layers.GRU if cell == "gru" else layers.LSTM
    for i in range(layers_n):
        r = Rnn(hidden, return_sequences=True)
        x = layers.Bidirectional(r)(x) if bidirectional else r(x)
        if i < layers_n - 1:
            x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    return tf.keras.Model(inp, layers.Dense(num_classes)(x), name="rnn")
