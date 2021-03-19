import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        channels = tf.shape(inputs)[-1]
        nb_split_channels = channels // 2

        a = inputs[:, :, :, :nb_split_channels]
        b = inputs[:, :, :, nb_split_channels:]

        return a * tf.nn.sigmoid(b)
