import gym
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from flatland.core.grid import grid4
from flatlander.models.common.models import NatureCNN, ImpalaCNN
import numpy as np


class GlobalObsModel(TFModelV2):
    def import_from_h5(self, h5_file):
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert isinstance(action_space, gym.spaces.Discrete) or isinstance(action_space, gym.spaces.MultiDiscrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."
        self._action_space = action_space
        self._options = model_config['custom_model_config']
        self.baseline = None
        self._mask_unavailable_actions = self._options.get("mask_unavailable_actions", False)

        if self._mask_unavailable_actions:
            obs_space = obs_space['obs']
        else:
            obs_space = self.obs_space

        observations = tf.keras.layers.Input(shape=obs_space.shape)

        if self._options['architecture'] == 'nature':
            conv_out = NatureCNN(activation_out=True, **self._options.get('architecture_options', {}))(observations)
        elif self._options['architecture'] == 'impala':
            conv_out = ImpalaCNN(activation_out=True, **self._options.get('architecture_options', {}))(observations)
        else:
            raise ValueError(f"Invalid architecture: {self._options['architecture']}.")

        if isinstance(action_space, gym.spaces.Discrete):
            z = tf.keras.layers.Dense(units=action_space.n)(conv_out)
        else:
            z = tf.keras.layers.Dense(units=np.sum(action_space.nvec))(conv_out)
        baseline = tf.keras.layers.Dense(units=1)(conv_out)

        self._model = tf.keras.Model(inputs=observations, outputs=[z, baseline])
        self.register_variables(self._model.variables)
        # self._model.summary()

    def forward(self, input_dict, state, seq_lens):
        # obs = preprocess_obs(input_dict['obs'])
        if self._mask_unavailable_actions:
            obs = input_dict['obs']['obs']
        else:
            obs = input_dict['obs']
        obs = tf.cast(obs, dtype=tf.float32)
        logits, baseline = self._model(obs)
        #if isinstance(self._action_space, gym.spaces.MultiDiscrete):
        #    logits = tf.reshape(logits, (self._action_space.nvec.shape[0], self._action_space.nvec[0]))
        self.baseline = tf.reshape(baseline, [-1])
        if self._mask_unavailable_actions:
            inf_mask = tf.maximum(tf.math.log(input_dict['obs']['available_actions']), tf.float32.min)
            logits = logits + inf_mask
        return logits, state

    def variables(self):
        return self._model.variables

    def value_function(self):
        return self.baseline

