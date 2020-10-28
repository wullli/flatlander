import gym
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from flatlander.models.common.models import NatureCNN, ImpalaCNN


class MetaObsModel(TFModelV2):
    def import_from_h5(self, h5_file):
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space,
                         action_space, num_outputs,
                         model_config, name)
        self._options = model_config['custom_model_config']

        obs_space = obs_space.original_space

        observations = [tf.keras.layers.Input(shape=o.shape) for o in obs_space]
        conv_inp = observations[0]
        dense_inp = observations[1]

        if self._options['architecture'] == 'nature':
            conv_out = NatureCNN(activation_out=True, **self._options['architecture_options'])(conv_inp)
        elif self._options['architecture'] == 'impala':
            conv_out = ImpalaCNN(activation_out=True, **self._options['architecture_options'])(conv_inp)
        else:
            raise ValueError(f"Invalid architecture: {self._options['architecture']}.")

        dense_out = tf.keras.layers.Dense(16, activation='relu')(dense_inp)
        dense_out = tf.keras.layers.Dense(16, activation='relu')(dense_out)
        out = tf.keras.layers.Concatenate()([conv_out, dense_out])

        logits = tf.keras.layers.Dense(units=2)(out)
        baseline = tf.keras.layers.Dense(units=1)(out)
        self._model = tf.keras.Model(inputs=[conv_inp, dense_inp], outputs=[logits, baseline])
        self.register_variables(self._model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        logits, baseline = self._model(obs)
        self.baseline = tf.reshape(baseline, [-1])
        return logits, state

    def value_function(self):
        return self.baseline
