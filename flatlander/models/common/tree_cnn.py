from typing import List

import tensorflow as tf

from flatlander.models.common.attention import MultiHeadAttention
from flatlander.models.common.positional_tree_encoding import ShivQuirkPositionalEncoding


class TreeCNN(tf.keras.Model):

    def get_config(self):
        pass

    def __init__(self,
                 value_layers: List[int],
                 policy_layers: List[int],
                 n_actions,
                 dropout_rate=0.1):
        super(TreeCNN, self).__init__()
        if value_layers is None:
            value_layers = [512, 512]

        if policy_layers is None:
            policy_layers = [512, 512]

        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_4 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_5 = tf.keras.layers.Dropout(dropout_rate)

        self.conv_1 = tf.keras.layers.Conv1D(filters=64, activation="relu", kernel_size=3)
        self.conv_2 = tf.keras.layers.Conv1D(filters=64, activation="relu", kernel_size=3)
        self.max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=2)

        self.conv_3 = tf.keras.layers.Conv1D(filters=128, activation="relu", kernel_size=3)
        self.conv_4 = tf.keras.layers.Conv1D(filters=128, activation="relu", kernel_size=3)
        self.max_pool_2 = tf.keras.layers.MaxPool1D(pool_size=2)

        self.conv_5 = tf.keras.layers.Conv1D(filters=256, activation="relu", kernel_size=3)
        self.conv_6 = tf.keras.layers.Conv1D(filters=256, activation="relu", kernel_size=3)
        self.max_pool_3 = tf.keras.layers.MaxPool1D(pool_size=2)

        self.value_layers = [tf.keras.layers.Dense(neurons, activation="relu")
                             for neurons in value_layers]
        self.policy_layers = [tf.keras.layers.Dense(neurons, activation="relu")
                              for neurons in policy_layers]

        self.policy_out = tf.keras.layers.Dense(n_actions)
        self.value_out = tf.keras.layers.Dense(1)

        self.flatten = tf.keras.layers.Flatten()

    def call(self, input, train_mode):

        c_x = self.conv_1(input)
        c_x = self.conv_2(c_x)
        c_x = self.max_pool_1(c_x)
        c_x = self.dropout_1(c_x, training=train_mode)

        c_x = self.conv_3(c_x)
        c_x = self.conv_4(c_x)
        c_x = self.max_pool_2(c_x)

        c_x = self.conv_5(c_x)
        c_x = self.conv_6(c_x)
        c_x = self.max_pool_3(c_x)

        c_x = self.dropout_1(c_x, training=train_mode)
        c_x = self.flatten(c_x)

        p_x = c_x
        for i in range(len(self.policy_layers)):
            p_x = self.policy_layers[i](p_x)
        p_x = self.dropout_3(p_x, training=train_mode)
        policy_out = self.policy_out(p_x)

        v_x = c_x
        for i in range(len(self.value_layers)):
            v_x = self.value_layers[i](v_x)
        v_x = self.dropout_4(v_x, training=train_mode)
        value_out = self.value_out(v_x)

        return policy_out, value_out

