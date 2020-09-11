import tensorflow as tf

from flatlander.models.common.attention import MultiHeadAttention


class CooperatorAttention(tf.keras.Model):

    def get_config(self):
        pass

    def __init__(self, num_heads, d_model, ):
        super(CooperatorAttention, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, learn_scale=True)
        self.score = tf.keras.layers.Flatten()

    def call(self, input, train_mode=False):
        """
        Here we expand a dimension, to make the batch size be the input sequence length, in assumption that
        a forward pass includes all agents from the same environment in order
        """
        shape = tf.shape(input)
        batch_size = shape[0]
        seq_length = shape[1]
        d_model = shape[2]
        flat_observations = tf.reshape(input, (batch_size, seq_length * d_model))

        x = tf.expand_dims(flat_observations, 0)
        x, _ = self.mha(x, x, x, None)
        return tf.reshape(tf.squeeze(x), (batch_size, seq_length, d_model))
