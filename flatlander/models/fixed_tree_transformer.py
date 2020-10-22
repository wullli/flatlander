import logging

import gym
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from flatlander.models.common.transformer import Transformer


class FixedTreeTransformer(TFModelV2):
    def import_from_h5(self, h5_file):
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."

        self._options = model_config['custom_model_config']
        self._baseline = tf.expand_dims([0], 0)

        self._padded_obs_seq = None
        self._z = None

        self._logger = logging.getLogger(FixedTreeTransformer.__name__)

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        self.transformer = Transformer(d_model=self.obs_space.shape[1],
                                       use_positional_encoding=False, **self._options["transformer"])(self.inputs)

        self.policy_out = tf.keras.layers.Dense(action_space.n, activation="relu")
        self.value_out = tf.keras.layers.Dense(1, activation="relu")

        self.model = tf.keras.Model(self.inputs, [self.policy_out, self.value_out])

        self._test_transformer()
        self.register_variables(self.transformer.variables)

    def _test_transformer(self):
        inp = tf.random.uniform((100, self.obs_space.shape[0],
                                 self.obs_space.shape[1]),
                                dtype=tf.float32, minval=-1, maxval=1)
        _, _ = self.transformer.call(inp,
                                     train_mode=False,
                                     encoder_mask=tf.zeros((100, 1, 1, self.obs_space.shape[0])))

    def forward(self, input_dict, state, seq_lens):
        """
        To debug use breakpoint with: tf.reduce_any(tf.equal(encoder_mask, 0.).numpy()
        """
        obs: tf.Tensor = input_dict['obs']
        is_training = False
        if 'is_training' in input_dict.keys():
            is_training = input_dict['is_training']

        self._padded_obs_seq = tf.cast(obs, dtype=tf.float32)

        # ignore unavailable values
        inf = tf.fill(dims=(1, 1, tf.shape(self._padded_obs_seq)[2]), value=-1.)
        encoder_mask = tf.not_equal(self._padded_obs_seq, inf)
        encoder_mask = tf.cast(tf.math.reduce_all(encoder_mask, axis=2), tf.float32)
        obs_shape = tf.shape(self._padded_obs_seq)
        encoder_mask = tf.reshape(encoder_mask, (obs_shape[0], 1, 1, obs_shape[1]))

        self._z = self.infer(self._padded_obs_seq,
                             is_training,
                             encoder_mask)

        return self._z, state

    def infer(self, x, is_training, encoder_mask) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        policy_target, value_target = self.transformer(x, train_mode=is_training,
                                                       encoder_mask=encoder_mask)
        self._baseline = tf.reshape(value_target, [-1])
        return policy_target

    def variables(self, **kwargs):
        return self.transformer.variables

    def value_function(self):
        return self._baseline
