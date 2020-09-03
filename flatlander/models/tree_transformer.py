import gym
import numpy as np
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from flatlander.models.common.transformer import Transformer


class TreeTransformer(TFModelV2):
    def import_from_h5(self, h5_file):
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."

        self._options = model_config['custom_options']
        self._mask_unavailable_actions = self._options.get("mask_unavailable_actions", False)
        self._n_features_per_node = self._options.get("n_features_per_node", 11)
        self._tree_depth = self._options.get("tree_depth", 2)
        self.positional_encoding_dim = self.action_space.n * (self._tree_depth + 1)
        self._policy_target = tf.cast(np.zeros(shape=(100, self.action_space.n)), tf.float32)
        self._baseline = tf.expand_dims([0], 0)

        num_heads = int(self.positional_encoding_dim / (self._tree_depth + 1))

        self.transformer = Transformer(num_layers=2,
                                       d_model=self.positional_encoding_dim,
                                       num_heads=num_heads,
                                       dense_neurons=512,
                                       n_actions=self.action_space.n)

        inp = tf.random.uniform((100, 21, self._n_features_per_node),
                                dtype=tf.float32, minval=-1, maxval=1)
        _, _ = self.transformer.call(inp,
                                     train_mode=False,
                                     positional_encoding=tf.cast(
                                         tf.expand_dims(np.zeros((21, self.positional_encoding_dim)), 0),
                                         dtype=tf.float32))
        self.register_variables(self.transformer.variables)

    def forward(self, input_dict, state, seq_lens):
        """
        To debug use breakpoint with: tf.reduce_any(tf.math.is_inf(padded_obs_seq)).numpy()
        """
        obs: [tf.Tensor, tf.Tensor] = input_dict['obs']
        is_training = False
        if 'is_training' in input_dict.keys():
            is_training = input_dict['is_training']

        padded_obs_seq = tf.cast(obs[0], dtype=tf.float32)
        padded_enc_seq = tf.cast(obs[1], dtype=tf.float32)

        z = self.infer(padded_obs_seq,
                       padded_enc_seq,
                       is_training)

        return z, state

    def infer(self, x, positional_encoding: np.ndarray, is_training) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        positional_encoding = tf.cast(positional_encoding, tf.float32)
        policy_target, value_target = self.transformer(x, train_mode=is_training,
                                                       positional_encoding=positional_encoding)
        self._baseline = tf.reshape(value_target, [-1])
        return policy_target

    @staticmethod
    def strip_single(sample):
        feature_dim = tf.shape(sample)[1]
        filled = tf.fill(dims=(1, feature_dim), value=-np.inf)
        comp = tf.not_equal(sample, filled)
        mask = tf.math.reduce_all(comp, axis=1)
        return sample[mask]

    def variables(self, **kwargs):
        return self.transformer.variables

    def value_function(self):
        return self._baseline
