import logging

import gym
import tensorflow as tf
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.misc import normc_initializer

from flatlander.models.common.transformer import Transformer

class MyKerasQModel(DistributionalQTFModel):
    """Custom model for DQN."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kw):
        super(MyKerasQModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        # Define the core model layers which will be used by the other
        # output heads of DistributionalQModel
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            128,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(layer_1)
        self.base_model = tf.keras.Model(self.inputs, layer_out)
        self.register_variables(self.base_model.variables)

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state

    def metrics(self):
        return {"foo": tf.constant(42.0)}


class DqnFixedTreeTransformer(DistributionalQTFModel):
    def import_from_h5(self, h5_file):
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kw):
        super(DqnFixedTreeTransformer, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."

        self._num_outputs = num_outputs

        self._options = model_config['custom_options']
        self._baseline = tf.expand_dims([0], 0)
        self._padded_obs_seq = None
        self._z = None

        self._logger = logging.getLogger(DqnFixedTreeTransformer.__name__)

        self.transformer = Transformer(d_model=self.obs_space.shape[1],
                                       use_positional_encoding=False, **self._options["transformer"])

        self.layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="dqn_ttf_out",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.transformer)

        self.model = tf.keras.Model(self.inputs, self.layer_out)

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
