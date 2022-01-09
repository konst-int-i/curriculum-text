from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow import keras
import tensorflow as tf


class FastText(Model):
    def __init__(
        self,
        maxlen,
        max_features,
        embedding_dims,
        class_num,
        last_activation="softmax",
    ):
        super(FastText, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(
            self.max_features, self.embedding_dims, input_length=self.maxlen
        )
        self.avg_pooling = GlobalAveragePooling1D()
        self.classifier = Dense(
            self.class_num,
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            activity_regularizer=regularizers.l2(1e-5),
            activation=self.last_activation,
        )

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError(
                "The rank of inputs of FastText must be 2, but now is %d"
                % len(inputs.get_shape())
            )
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError(
                "The maxlen of inputs of FastText must be %d, but now is %d"
                % (self.maxlen, inputs.get_shape()[1])
            )
        embedding = self.embedding(inputs)
        # dropout_embedding = self.dropout(embedding)
        x = self.avg_pooling(embedding)
        output = self.classifier(x)
        return output
