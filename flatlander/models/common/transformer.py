import tensorflow as tf

from flatlander.models.common.attention import MultiHeadAttention


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dense_neurons, n_features,
                 n_actions, rate=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dense_neurons, n_features, rate)
        self.policy = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.linear)
        self.value = tf.keras.layers.Dense(1)

    def call(self, input, target, train_mode, enc_padding_mask, look_ahead_mask, positional_encoding):
        enc_output = self.encoder(input, train_mode, enc_padding_mask, positional_encoding)
        avg_encoder = tf.reduce_sum(enc_output, axis=1)
        policy_out = self.policy(avg_encoder)
        value_out = self.value(avg_encoder)

        return policy_out, value_out


class TransformerLayer(tf.keras.layers.Layer):

    @staticmethod
    def point_wise_feed_forward_network(d_model, dense_neurons):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dense_neurons, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])


class Encoder(TransformerLayer):
    def __init__(self, num_layers, d_model, num_heads, dense_neurons, n_features, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.dense = tf.keras.layers.Dense(d_model, activation='relu')

        self.enc_layers = [EncoderLayer(d_model, num_heads, dense_neurons, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, positional_encoding):

        x = self.dense(x)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += positional_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class EncoderLayer(TransformerLayer):
    def __init__(self, d_model, num_heads, dense_neurons, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dense_neurons)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2



