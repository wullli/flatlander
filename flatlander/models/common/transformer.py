import tensorflow as tf

from flatlander.models.common.attention import MultiHeadAttention


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dense_neurons,
                 n_actions, rate=0.1, use_positional_encoding=True):
        super(Transformer, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        self.encoder = Encoder(num_layers, d_model, num_heads, dense_neurons, rate, use_positional_encoding)

        self.pd1 = tf.keras.layers.Dense(512, activation="relu")
        self.pd2 = tf.keras.layers.Dense(512, activation="relu")
        self.policy = tf.keras.layers.Dense(n_actions)

        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model=d_model)

        self.vd1 = tf.keras.layers.Dense(512, activation="relu")
        self.vd2 = tf.keras.layers.Dense(512, activation="relu")
        self.value = tf.keras.layers.Dense(1)

    def call(self, input, train_mode, encoder_mask, positional_encoding=None):
        if self.use_positional_encoding:
            positional_encoding = self.pos_encoding(positional_encoding)
        enc_output = self.encoder(input, train_mode, positional_encoding, encoder_mask=encoder_mask)
        avg_encoder = tf.reduce_mean(enc_output, axis=1)

        p_x = self.pd1(avg_encoder)
        p_x = self.pd2(p_x)

        v_x = self.vd1(avg_encoder)
        v_x = self.vd2(v_x)
        policy_out = self.policy(p_x)
        value_out = self.value(v_x)

        return policy_out, value_out


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
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


class TransformerLayer(tf.keras.layers.Layer):

    @staticmethod
    def point_wise_feed_forward_network(d_model, dense_neurons):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dense_neurons, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])


class Encoder(TransformerLayer):
    def __init__(self, num_layers, d_model, num_heads, dense_neurons, rate=0.25, use_positional_encoding=True):
        super(Encoder, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        self.d_model = d_model
        self.num_layers = num_layers
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(d_model, activation='relu')

        self.enc_layers = [EncoderLayer(d_model, num_heads, dense_neurons, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, positional_encoding, encoder_mask):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        if self.use_positional_encoding:
            x += positional_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, encoder_mask)

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
