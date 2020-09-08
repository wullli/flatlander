import logging

import gym
import numpy as np
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from flatlander.models.common.transformer import Transformer


class PositionalTreeTransformer(TFModelV2):
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

        self.positional_encoding_dim = (self.action_space.n - 1) * self._tree_depth
        self._baseline = tf.expand_dims([0], 0)

        self._padded_obs_seq = None
        self._padded_enc_seq = None
        self._z = None

        self._logger = logging.getLogger(PositionalTreeTransformer.__name__)

        num_heads = int(self.positional_encoding_dim / self._tree_depth)

        self.transformer = Transformer(d_model=self.positional_encoding_dim,
                                       num_heads=num_heads,
                                       n_actions=self.action_space.n,
                                       use_positional_encoding=True, **self._options["transformer"])

        self._test_transformer()
        self.register_variables(self.transformer.variables)

    def _test_transformer(self):
        inp = tf.random.uniform((100, 21, self._n_features_per_node),
                                dtype=tf.float32, minval=-1, maxval=1)
        _, _ = self.transformer.call(inp,
                                     train_mode=False,
                                     positional_encoding=tf.cast(
                                         tf.expand_dims(np.zeros((21, self.positional_encoding_dim)), 0),
                                         dtype=tf.float32), encoder_mask=np.zeros((100, 21, 21)))

    def forward(self, input_dict, state, seq_lens):
        """
        To debug use breakpoint with: tf.reduce_any(tf.math.is_inf(padded_obs_seq)).numpy()
        """
        obs: [tf.Tensor, tf.Tensor] = input_dict['obs']
        is_training = False
        if 'is_training' in input_dict.keys():
            is_training = input_dict['is_training']

        self._padded_obs_seq = tf.cast(obs[0], dtype=tf.float32)
        self._padded_enc_seq = tf.cast(obs[1], dtype=tf.float32)

        # ignore unavailable values
        inf = tf.fill(dims=(1, 1, tf.shape(self._padded_obs_seq)[2]), value=-1.)
        encoder_mask = tf.not_equal(self._padded_obs_seq, inf)
        encoder_mask = tf.cast(tf.math.reduce_all(encoder_mask, axis=2), tf.float32)
        obs_shape = tf.shape(self._padded_obs_seq)
        encoder_mask = tf.reshape(encoder_mask, (obs_shape[0], obs_shape[1], 1))
        encoder_mask = tf.broadcast_to(encoder_mask, (obs_shape[0], obs_shape[1], obs_shape[1]))

        self._z = self.infer(self._padded_obs_seq,
                             self._padded_enc_seq,
                             is_training,
                             encoder_mask)

        return self._z, state

    def infer(self, x, positional_encoding: np.ndarray, is_training, encoder_mask) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        positional_encoding = tf.cast(positional_encoding, tf.float32)
        policy_target, value_target = self.transformer(x, train_mode=is_training,
                                                       positional_encoding=positional_encoding,
                                                       encoder_mask=encoder_mask)
        self._baseline = tf.reshape(value_target, [-1])
        return policy_target

    def _traceback(self):
        self._logger.error("policy_out:" + str(self._z.numpy()))
        self._logger.error("obs_seq:" + str(self._padded_obs_seq.numpy()))
        self._logger.error("enc_seq" + str(self._padded_enc_seq.numpy()))

    def variables(self, **kwargs):
        return self.transformer.variables

    def value_function(self):
        return self._baseline

    def __del__(self):
        self._traceback()
