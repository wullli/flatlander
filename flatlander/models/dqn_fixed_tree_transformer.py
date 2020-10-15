import logging

import gym
import tensorflow as tf
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.framework import TensorType

from flatlander.models.common.transformer import Transformer


class DqnFixedTreeTransformer(DistributionalQTFModel):
    def value_function(self) -> TensorType:
        pass

    def import_from_h5(self, h5_file):
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kw):
        super(DqnFixedTreeTransformer, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."

        self._num_outputs = num_outputs

        self._options = model_config['custom_model_config']
        self._baseline = tf.expand_dims([0], 0)
        self._padded_obs_seq = None
        self._z = None
        self.non_tree_net = None
        self._logger = logging.getLogger(DqnFixedTreeTransformer.__name__)

        if isinstance(obs_space, gym.spaces.Box):
            self._tuple_space = False
            self.transformer = Transformer(out_dim=self._num_outputs, d_model=obs_space.shape[1],
                                           use_positional_encoding=False, **self._options["transformer"])
            self.q_out = tf.keras.layers.Dense(
                num_outputs,
                name="dqn_ttf_out",
                activation=tf.nn.relu,
                kernel_initializer=normc_initializer(1.0))

        elif isinstance(obs_space.original_space, gym.spaces.Tuple):
            self._tuple_space = True
            self.transformer = Transformer(out_dim=self._num_outputs, d_model=obs_space.original_space[0].shape[1],
                                           use_positional_encoding=False, **self._options["transformer"])
            self.q_out = tf.keras.layers.Dense(
                num_outputs,
                name="dqn_ttf_out",
                activation=tf.nn.relu,
                kernel_initializer=normc_initializer(1.0))

            self.non_tree_layer_1 = tf.keras.layers.Dense(128, activation=tf.nn.relu,
                                                          kernel_initializer=normc_initializer(1.0))
            self.non_tree_layer_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu,
                                                          kernel_initializer=normc_initializer(1.0))

        self._test_transformer()
        variables = self.transformer.variables \
                    + self.q_out.variables \
                    + self.q_value_head.variables \
                    + self.state_value_head.variables
        if self._tuple_space:
            variables = variables + self.non_tree_layer_1.variables + self.non_tree_layer_2.variables
        self.register_variables(variables)

    def _test_transformer(self):
        if isinstance(self.obs_space, gym.spaces.Box):
            inp = tf.random.uniform((100, self.obs_space.shape[0],
                                     self.obs_space.shape[1]),
                                    dtype=tf.float32, minval=-1, maxval=1)

            z = self.transformer.call(inp,
                                      train_mode=False,
                                      encoder_mask=tf.zeros((100, 1, 1, self.obs_space.shape[0])))
        elif isinstance(self.obs_space.original_space, gym.spaces.Tuple):

            inp = tf.random.uniform((100, self.obs_space.original_space[0].shape[0],
                                     self.obs_space.original_space[0].shape[1]),
                                    dtype=tf.float32, minval=-1, maxval=1)

            z = self.transformer.call(inp,
                                      train_mode=False,
                                      encoder_mask=tf.zeros((100, 1, 1, self.obs_space.original_space[0].shape[0])))

            z_non_tree = tf.random.uniform((100, self.obs_space.original_space[1].shape[0]),
                                           dtype=tf.float32, minval=-1, maxval=1)

            z_non_tree = self.non_tree_layer_1(z_non_tree)
            z_non_tree = self.non_tree_layer_2(z_non_tree)

            z = tf.concat([z, z_non_tree], axis=1)

        _ = self.q_out(z)

    def forward(self, input_dict, state, seq_lens):
        """
        To debug use breakpoint with: tf.reduce_any(tf.equal(encoder_mask, 0.).numpy()
        """
        obs: tf.Tensor = input_dict['obs']
        is_training = False
        if 'is_training' in input_dict.keys():
            is_training = input_dict['is_training']

        if self._tuple_space:
            self._padded_obs_seq = tf.cast(obs[0], dtype=tf.float32)
        else:
            self._padded_obs_seq = tf.cast(obs, dtype=tf.float32)

        # ignore unavailable values
        inf = tf.fill(dims=(1, 1, tf.shape(self._padded_obs_seq)[2]), value=-1.)
        encoder_mask = tf.not_equal(self._padded_obs_seq, inf)
        encoder_mask = tf.cast(tf.math.reduce_all(encoder_mask, axis=2), tf.float32)
        obs_shape = tf.shape(self._padded_obs_seq)
        encoder_mask = tf.reshape(encoder_mask, (obs_shape[0], 1, 1, obs_shape[1]))

        z_tree = self.infer(self._padded_obs_seq,
                            is_training,
                            encoder_mask)

        if self._tuple_space:
            z_non_tree = tf.cast(obs[1], dtype=tf.float32)

            z_non_tree = self.non_tree_layer_1.call(z_non_tree)
            z_non_tree = self.non_tree_layer_2.call(z_non_tree)

            z = tf.concat([z_tree, z_non_tree], axis=1)
        else:
            z = z_tree

        q_out = self.q_out(z)

        return q_out, state

    def infer(self, x, is_training, encoder_mask) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        out = self.transformer(x, train_mode=is_training,
                               encoder_mask=encoder_mask)
        return out

    def variables(self, **kwargs):
        variables = self.transformer.variables \
                    + self.q_out.variables \
                    + self.q_value_head.variables \
                    + self.state_value_head.variables
        if self._tuple_space:
            variables = variables + self.non_tree_layer_1.variables + self.non_tree_layer_2.variables

        return variables
