import logging

import gym
import numpy as np
from ray.rllib import TFPolicy, Policy
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule
from ray.rllib.utils import override, DeveloperAPI, try_import_tf
from ray.tune import register_trainable

from flatlander.models.common.tree_transformer import Transformer

tf = try_import_tf()


class FixedTreeTransformer(TFModelV2):
    def import_from_h5(self, h5_file):
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."

        self._options = model_config['custom_options']
        self._n_features_per_node = self._options.get("n_features_per_node", 11)
        self._tree_depth = self._options.get("tree_depth", 2)
        self._baseline = tf.expand_dims([0], 0)

        self._padded_obs_seq = None
        self._z = None

        self._logger = logging.getLogger(FixedTreeTransformer.__name__)

        self.transformer = Transformer(n_actions=self.action_space.n,
                                       d_model=self._n_features_per_node,
                                       use_positional_encoding=False, **self._options["transformer"])

        self._test_transformer()
        self.register_variables(self.transformer.variables)

    def _test_transformer(self):
        inp = tf.random.uniform((100, 21, self._n_features_per_node),
                                dtype=tf.float32, minval=-1, maxval=1)
        _, _ = self.transformer.call(inp,
                                     train_mode=False,
                                     encoder_mask=np.zeros((100, 1, 1, 21)))

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


@DeveloperAPI
class TransformerLearningRateSchedule:
    """Mixin for TFPolicy that adds a learning rate schedule."""

    @DeveloperAPI
    def __init__(self, d_model, warmup_steps):
        self.cur_lr = tf.get_variable("lr", initializer=0.001, trainable=False)
        self.cur_step = tf.get_variable("cur_step", initializer=1.0, trainable=False)
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(TransformerLearningRateSchedule, self).on_global_var_update(global_vars)
        self.cur_step.load(float(global_vars["timestep"]), session=self._sess)
        arg1 = tf.math.rsqrt(self.cur_step)
        arg2 = self.cur_step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        self.cur_lr = lr

    @override(TFPolicy)
    def optimizer(self):
        return tf.train.AdamOptimizer(self.cur_lr, beta1=0.9, beta2=0.98, epsilon=1e-9)


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    warmup_steps = config["model"]["custom_options"].get("warmup_steps", 10000)
    TransformerLearningRateSchedule.__init__(policy,
                                             config["model"]["custom_options"]["n_features_per_node"],
                                             warmup_steps)


TTFPPOPolicy = PPOTFPolicy.with_updates(
    name="TTFPPOPolicy",
    before_loss_init=setup_mixins,
    mixins=[
        TransformerLearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])

register_trainable(
    "TTFPPO",
    PPOTrainer.with_updates(
        name="CCPPOTrainer", get_policy_class=lambda c: TTFPPOPolicy
    ),
)
