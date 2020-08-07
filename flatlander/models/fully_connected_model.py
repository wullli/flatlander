import gym
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from flatlander.models.common.models import FullyConnected


class FullyConnectedModel(TFModelV2):
    def import_from_h5(self, h5_file):
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."
        self._action_space = action_space
        self._options = model_config['custom_options']
        self._mask_unavailable_actions = self._options.get("mask_unavailable_actions", False)
        self._vf_share_layers = model_config["vf_share_layers"]

        if self._mask_unavailable_actions:
            observations = tf.keras.layers.Input(shape=obs_space.original_space['obs'].shape)
        else:
            observations = tf.keras.layers.Input(shape=obs_space.shape)

        activation = tf.keras.activations.deserialize(self._options['activation'])
        fc_out = FullyConnected(layers=self._options['layers'], activation=activation,
                                layer_norm=self._options['layer_norm'], activation_out=True)(observations)
        logits = tf.keras.layers.Dense(units=action_space.n)(fc_out)
        if not self._vf_share_layers:
            fc_out = FullyConnected(layers=self._options['layers'], activation=activation,
                                    layer_norm=self._options['layer_norm'], activation_out=True)(observations)
        baseline = tf.keras.layers.Dense(units=1)(fc_out)
        self._model = tf.keras.Model(inputs=[observations], outputs=[logits, baseline])
        self.register_variables(self._model.variables)
        self._model.summary()

    def forward(self, input_dict, state, seq_lens):
        if self._mask_unavailable_actions:
            obs = input_dict['obs']['obs']
        else:
            obs = input_dict['obs']
        logits, baseline = self._model(obs)
        self.baseline = tf.reshape(baseline, [-1])
        if self._mask_unavailable_actions:
            inf_mask = tf.maximum(tf.math.log(input_dict['obs']['available_actions']), tf.float32.min)
            logits = logits + inf_mask
        return logits, state

    def value_function(self):
        return self.baseline
