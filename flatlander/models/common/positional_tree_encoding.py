import tensorflow as tf


class ShivQuirkPositionalEncoding(tf.keras.layers.Layer):
    """
    Paper:
    https://papers.nips.cc/paper/9376-novel-positional-encodings-to-enable-tree-based-transformers.pdf
    """
    def __init__(self, d_model):
        super(ShivQuirkPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.p_raw = self.add_weight(name='p_param',
                                     shape=(1,),
                                     trainable=True)

        self.p = tf.keras.activations.tanh(self.p_raw)

    def call(self, positional_encoding, **kwargs):
        self.positional_map = tf.squeeze(tf.stack([tf.pow(self.p, ex) for ex in range(self.d_model)]))
        self.positional_map = tf.multiply(self.positional_map, tf.sqrt(1 - tf.pow(self.p, 2)))
        self.positional_map = tf.multiply(self.positional_map, tf.sqrt(tf.divide(self.d_model, 2)))
        return tf.multiply(positional_encoding, self.positional_map)
