import logging

import gym
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from flatlander.models.common.tree_cnn import TreeCNN


class FixedTreeCnn(TFModelV2):
    def import_from_h5(self, h5_file):
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."

        self._options = model_config['custom_model_config']
        self._baseline = None
        self._padded_obs_seq = None
        self._z = None
        self._logger = logging.getLogger(FixedTreeCnn.__name__)

        self._model = TreeCNN(n_actions=self.action_space.n,
                              **self._options["cnn"])

        self.register_variables(self._model.variables)

    def forward(self, input_dict, state, seq_lens):
        """
        To debug use breakpoint with: tf.reduce_any(tf.equal(encoder_mask, 0.).numpy()
        """
        obs: tf.Tensor = input_dict['obs']
        is_training = False
        if 'is_training' in input_dict.keys():
            is_training = input_dict['is_training']

        self._z = self.infer(tf.cast(obs, dtype=tf.float32),
                             is_training)

        return self._z, state

    def infer(self, x, is_training) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        policy_target, value_target = self._model(x, train_mode=is_training)
        self._baseline = tf.reshape(value_target, [-1])
        return policy_target

    def _traceback(self):
        self._logger.error("policy_out:" + str(self._z.numpy()))
        self._logger.error("obs_seq:" + str(self._padded_obs_seq.numpy()))
        self._logger.error("enc_seq" + str(self._padded_enc_seq.numpy()))

    def variables(self, **kwargs):
        return self._model.variables

    def value_function(self):
        return self._baseline

    def __del__(self):
        self._traceback()
