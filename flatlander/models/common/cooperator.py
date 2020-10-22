from typing import List

import tensorflow as tf

from flatlander.models.common.transformer import Encoder


class Cooperator(tf.keras.Model):

    def get_config(self):
        pass

    def __init__(self, num_encoder_layers,
                 d_model,
                 num_heads,
                 encoder_layer_neurons,
                 value_layers: List[int],
                 policy_layers: List[int],
                 action_layers: List[int],
                 n_actions,
                 dropout_rate=0.1,
                 use_cnn_decoding=False,
                 use_positional_encoding=True):
        super(Cooperator, self).__init__()
        if value_layers is None:
            value_layers = [512, 512]

        if policy_layers is None:
            policy_layers = [512, 512]

        if action_layers is None:
            action_layers = [512, 512]

        self.use_positional_encoding = use_positional_encoding
        self.use_cnn_decoding = use_cnn_decoding

        self.encoder = Encoder(num_encoder_layers,
                               d_model,
                               num_heads,
                               encoder_layer_neurons,
                               dropout_rate,
                               use_positional_encoding)

        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)

        self.value_layers = [tf.keras.layers.Dense(neurons, activation="relu")
                             for neurons in value_layers]
        self.policy_layers = [tf.keras.layers.Dense(neurons, activation="relu")
                              for neurons in policy_layers]
        self.action_layers = [tf.keras.layers.Dense(neurons, activation="relu")
                              for neurons in action_layers]

        self.policy = tf.keras.layers.Dense(n_actions)
        self.value = tf.keras.layers.Dense(1)
        self.flatten = tf.keras.layers.Flatten()
        self.concat = tf.keras.layers.Concatenate()

    def call(self, input, train_mode, encoder_mask, other_actions):
        enc_output = self.encoder(input, train_mode, None, encoder_mask=encoder_mask)

        x = self.flatten(enc_output)

        x_a = other_actions
        for i in range(len(self.action_layers)):
            x_a = self.action_layers[i](x_a)

        p_x = self.concat([x, tf.broadcast_to(x_a, (tf.shape(x)[0], tf.shape(x_a)[1]))])
        for i in range(len(self.policy_layers)):
            p_x = self.policy_layers[i](p_x)
        p_x = self.dropout_1(p_x, training=train_mode)
        policy_out = self.policy(p_x)

        v_x = x
        for i in range(len(self.value_layers)):
            v_x = self.value_layers[i](v_x)
        v_x = self.dropout_2(v_x, training=train_mode)
        value_out = self.value(v_x)

        return policy_out, value_out
