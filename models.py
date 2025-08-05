import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import bayesflow as bf


class GRU(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gru = keras.layers.GRU(64, dropout=0.1)
        self.summary_stats = keras.layers.Dense(18)

    def call(self, time_series, **kwargs):
        """Compresses time_series of shape (batch_size, T, 1) into summaries of shape (batch_size, 8)."""

        summary = self.gru(time_series, training=kwargs.get("stage") == "training")
        summary = self.summary_stats(summary)
        return summary


class RegularizedCNNBiGRU(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv1D(64, 5, padding="same", activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)

        self.bigru = layers.Bidirectional(
            layers.GRU(
                64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3
            )  # More dropout
        )
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)

        #  L2 regularization
        self.summary_stats = layers.Dense(18, kernel_regularizer="l2")

    def call(self, time_series, **kwargs):
        training = kwargs.get("stage") == "training"
        x = self.conv(time_series)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.bigru(x, training=training)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)

        return self.summary_stats(x)
