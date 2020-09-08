from typing import List

import tensorflow as tf

from flatlander.models.common.attention import MultiHeadAttention
from flatlander.models.common.positional_tree_encoding import ShivQuirkPositionalEncoding


class Transformer(tf.keras.Model):

    def __init__(self, num_encoder_layers,
                 d_model,
                 num_heads,
                 encoder_layer_neurons,
                 value_layers: List[int],
                 policy_layers: List[int],
                 embedding_layers: List[int],
                 n_actions,
                 dropout_rate=0.1,
                 use_positional_encoding=True):
        super(Transformer, self).__init__()
        if value_layers is None:
            value_layers = [512, 512]

        if policy_layers is None:
            policy_layers = [512, 512]
        self.use_positional_encoding = use_positional_encoding

        self.encoder = Encoder(num_encoder_layers,
                               d_model,
                               num_heads,
                               encoder_layer_neurons,
                               embedding_layers,
                               dropout_rate,
                               use_positional_encoding)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.value_layers = [tf.keras.layers.Dense(neurons, activation="relu")
                             for neurons in value_layers]
        self.policy_layers = [tf.keras.layers.Dense(neurons, activation="relu")
                              for neurons in policy_layers]

        self.policy = tf.keras.layers.Dense(n_actions)
        self.value = tf.keras.layers.Dense(1)

        if use_positional_encoding:
            self.pos_encoding = ShivQuirkPositionalEncoding(d_model=d_model)

    def call(self, input, train_mode, encoder_mask, positional_encoding=None):
        if self.use_positional_encoding:
            positional_encoding = self.pos_encoding(positional_encoding)
        enc_output = self.encoder(input, train_mode, positional_encoding, encoder_mask=encoder_mask)
        avg_encoder = tf.reduce_mean(enc_output, axis=1)

        p_x = self.dropout(avg_encoder, training=train_mode)
        for i in range(len(self.policy_layers)):
            p_x = self.policy_layers[i](p_x)
        policy_out = self.policy(p_x)

        v_x = self.dropout(avg_encoder, training=train_mode)
        for i in range(len(self.value_layers)):
            v_x = self.value_layers[i](v_x)
        value_out = self.value(v_x)

        return policy_out, value_out


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dense_neurons,
                 embedding_layers: List[int],
                 rate=0.1,
                 use_positional_encoding=True):
        super(Encoder, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding_out = tf.keras.layers.Dense(d_model, activation="relu")
        self.embedding_layers = [tf.keras.layers.Dense(neurons, activation="relu")
                                 for i, neurons in enumerate(embedding_layers)]

        self.enc_layers = [EncoderLayer(d_model, num_heads, dense_neurons, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, positional_encoding, encoder_mask):

        for i in range(len(self.embedding_layers)):
            x = self.embedding_layers[i](x)
        x = self.embedding_out(x)

        if self.use_positional_encoding:
            x += positional_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, encoder_mask)

        return x  # (batch_size, input_seq_len, d_model)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dense_neurons, use_residual=False, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.use_residual = use_residual
        self.mha = MultiHeadAttention(d_model, num_heads, learn_scale=True)
        self.ffn = self.point_wise_feed_forward_network(d_model, dense_neurons)

        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout_1(attn_output, training=training)
        if self.use_residual:
            out1 = self.layer_norm_1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        else:
            out1 = self.layer_norm_1(attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout_2(ffn_output, training=training)

        if self.use_residual:
            out2 = self.layer_norm_2(ffn_output + out1)  # (batch_size, input_seq_len, d_model)
        else:
            out2 = self.layer_norm_2(ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    @staticmethod
    def point_wise_feed_forward_network(d_model, dense_neurons):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dense_neurons, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
