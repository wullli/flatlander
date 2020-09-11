import logging

import gym
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

from flatlander.models.common.cooperator import CooperatorAttention
from flatlander.models.common.transformer import Transformer

tf = try_import_tf()


class CooperativeFtt(TFModelV2):
    def import_from_h5(self, h5_file):
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."

        self._options = model_config['custom_options']
        self._baseline = tf.expand_dims([0], 0)
        self._padded_obs_seq = None
        self._z = None
        self._logger = logging.getLogger(CooperativeFtt.__name__)
        self._real_obs_space = self.obs_space.original_space["obs"]

        self.cooperator = CooperatorAttention(d_model=self._real_obs_space.shape[1] * self._real_obs_space.shape[0],
                                              num_heads=self._real_obs_space.shape[0])
        self.transformer = Transformer(n_actions=self.action_space.n,
                                       d_model=self._real_obs_space.shape[1],
                                       use_positional_encoding=False, **self._options["transformer"])

        self._test_transformer()
        self._test_cooperator()
        self.register_variables(self.variables())

    def forward(self, input_dict, state, seq_lens):
        """
        To debug use breakpoint with: tf.reduce_any(tf.equal(encoder_mask, 0.).numpy()
        """
        obs: tf.Tensor = input_dict['obs']
        is_training = False
        if 'is_training' in input_dict.keys():
            is_training = input_dict['is_training']

        self._padded_obs_seq = tf.cast(obs, dtype=tf.float32)

        weighted_obs = self.cooperator(self._padded_obs_seq)

        # ignore unavailable values
        inf = tf.fill(dims=(1, 1, tf.shape(self._padded_obs_seq)[2]), value=-1.)
        encoder_mask = tf.not_equal(self._padded_obs_seq, inf)
        encoder_mask = tf.cast(tf.math.reduce_all(encoder_mask, axis=2), tf.float32)
        obs_shape = tf.shape(self._padded_obs_seq)
        encoder_mask = tf.reshape(encoder_mask, (obs_shape[0], 1, 1, obs_shape[1]))

        self._z = self.infer(weighted_obs,
                             is_training,
                             encoder_mask)

        return self._z, state

    def infer(self, x, is_training, encoder_mask) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        policy_target, value_target = self.transformer(x, train_mode=is_training,
                                                       encoder_mask=encoder_mask)
        self._baseline = tf.reshape(value_target, [-1])
        return policy_target

    def _test_transformer(self):
        inp = tf.random.uniform((100, self._real_obs_space.shape[0],
                                 self._real_obs_space.shape[1]),
                                dtype=tf.float32, minval=-1, maxval=1)
        _, _ = self.transformer.call(inp,
                                     train_mode=False,
                                     encoder_mask=tf.zeros((100, 1, 1, self._real_obs_space.shape[0])))

    def _test_cooperator(self):
        inp = tf.random.uniform((100, self._real_obs_space.shape[0],
                                 self._real_obs_space.shape[1]),
                                dtype=tf.float32, minval=-1, maxval=1)
        _ = self.cooperator.call(inp, train_mode=False)

    def variables(self, **kwargs):
        return self.transformer.variables + self.cooperator.variables

    def value_function(self):
        return self._baseline
