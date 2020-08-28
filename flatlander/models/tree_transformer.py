from typing import List, Callable

import gym
import numpy as np
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from flatlander.models.common.transformer import Transformer


class TreeTransformer(TFModelV2):
    def import_from_h5(self, h5_file):
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        import tensorflow as tf
        tf.executing_eagerly()
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."

        self._options = model_config['custom_options']
        self._mask_unavailable_actions = self._options.get("mask_unavailable_actions", False)
        self._n_features_per_node = self._options.get("n_features_per_node", 11)
        self._tree_depth = self._options.get("tree_depth", 2)
        self.positional_encoding_dim = self.action_space.n * (self._tree_depth+1)
        self._policy_target = tf.expand_dims(np.zeros(shape=self.action_space.n), 0)
        self._baseline = tf.expand_dims([0], 0)
        self._input_shape = (None, self.obs_space.shape[0])

        num_heads = int(self.positional_encoding_dim / (self._tree_depth+1))

        self.transformer = Transformer(num_layers=2,
                                       d_model=self.positional_encoding_dim,
                                       num_heads=num_heads,
                                       dense_neurons=128,
                                       n_features=self._n_features_per_node,
                                       n_actions=self.action_space.n)

        temp_input = tf.expand_dims(tf.random.uniform((21, self._n_features_per_node),
                                                      dtype=tf.int64, minval=-1, maxval=1), 0)
        self.transformer.call(temp_input, self._policy_target,
                              train_mode=False,
                              enc_padding_mask=None,
                              look_ahead_mask=None,
                              dec_padding_mask=None,
                              positional_encoding=np.zeros(self.positional_encoding_dim))
        self.register_variables(self.transformer.variables)

    def forward(self, input_dict, state, seq_lens):
        obs: tf.Tensor = input_dict['obs']
        for sample in zip(obs[0], obs[1]):
            self._policy_target = tf.expand_dims(np.zeros(shape=self.action_space.n), 0)
            input_seq, input_enc = strip_sample()
            self.infer(padded_obs_seq, padded_enc_seq)

        z = self._policy_target
        if self._mask_unavailable_actions:
            inf_mask = tf.maximum(tf.math.log(input_dict['obs']['available_actions']), tf.float32.min)
            z = z + inf_mask
        return z, state

    @staticmethod
    def strip_sample(sample: [tf.Tensor, tf.Tensor]):
        padded_obs_seq = sample[0]
        padded_enc_seq = sample[1]
        filled_enc = tf.fill(dims=(1, 15), value=-np.inf)
        comp = tf.not_equal(padded_enc_seq, filled_enc)
        mask = tf.math.reduce_all(comp, axis=1)
        assert mask.shape[0] == padded_enc_seq.shape[0]
        input_seq = tf.boolean_mask(padded_obs_seq, mask)
        input_enc = tf.boolean_mask(padded_enc_seq, mask)
        return input_seq, input_enc

    def infer(self, x, positional_encoding: np.ndarray) -> np.ndarray:
        policy_target, value_target, _ = self.transformer(x, self._policy_target, train_mode=False,
                                                          enc_padding_mask=None,
                                                          look_ahead_mask=None,
                                                          dec_padding_mask=None,
                                                          positional_encoding=positional_encoding)
        self._baseline = value_target
        self._policy_target = policy_target

    def variables(self, **kwargs):
        return self.transformer.variables

    def value_function(self):
        return self._baseline
