import math

from ray.rllib import Policy, TFPolicy
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy, ComputeTDErrorMixin
from ray.rllib.agents.dqn.simple_q_tf_policy import TargetNetworkMixin
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import ValueNetworkMixin, PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import KLCoeffMixin
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.utils import override, DeveloperAPI, try_import_tf
from ray.tune import register_trainable
import numpy as np

from flatlander.alg.dynapex import DYNAPEX_DEFAULT_CONFIG, apex_execution_plan

tf = try_import_tf()


@DeveloperAPI
class TransformerLearningRateSchedule:
    """Mixin for TFPolicy that adds a learning rate schedule with warmup."""

    @DeveloperAPI
    def __init__(self, d_model, warmup_steps):
        self.cur_lr = tf.get_variable("lr", initializer=0.00005, trainable=False)
        self.cur_step = 1.0
        self.d_model_size = d_model
        self.warmup_steps = warmup_steps

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(TransformerLearningRateSchedule, self).on_global_var_update(global_vars)
        self.cur_step = global_vars["timestep"] + 1
        arg1 = np.reciprocal(np.sqrt(self.cur_step))
        arg2 = self.cur_step * (self.warmup_steps ** -1.5)
        lr = np.reciprocal(np.sqrt(self.d_model_size)) * min(arg1, arg2)
        self.cur_lr.load(lr * 0.05, session=self._sess)

    @override(TFPolicy)
    def optimizer(self):
        return tf.train.AdamOptimizer(self.cur_lr, beta1=0.9, beta2=0.98, epsilon=1e-9)


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    warmup_steps = config["model"]["custom_options"].get("warmup_steps", 100000)
    TransformerLearningRateSchedule.__init__(policy,
                                             config["model"]["custom_options"]["transformer"]["num_heads"],
                                             warmup_steps)


TTFPPOPolicy = PPOTFPolicy.with_updates(
    name="TTFPPOPolicy",
    before_loss_init=setup_mixins,
    mixins=[
        TransformerLearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])

TTFPPOPolicyInfer = PPOTFPolicy.with_updates(
    name="TTFPPOPolicyInfer",
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])


def setup_early_mixins(policy, obs_space, action_space, config):
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def setup_mid_mixins(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy)
    warmup_steps = config["model"]["custom_options"].get("warmup_steps", 100000)
    TransformerLearningRateSchedule.__init__(policy,
                                             config["model"]["custom_options"]["transformer"]["num_heads"],
                                             warmup_steps)


def setup_late_mixins(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, obs_space, action_space, config)


DQNTFWarmupPolicy = DQNTFPolicy.with_updates(
    name="DQNTFWarmupPolicy",
    before_init=setup_early_mixins,
    before_loss_init=setup_mid_mixins,
    after_init=setup_late_mixins,
    obs_include_prev_action_reward=False,
    mixins=[
        TransformerLearningRateSchedule,
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ])

ApexTrainer = DQNTrainer.with_updates(
    name="DYNAPEXWarmup",
    default_policy=DQNTFWarmupPolicy,
    default_config=DYNAPEX_DEFAULT_CONFIG,
    execution_plan=apex_execution_plan)

register_trainable(
    "APEXWarmup",
    DQNTrainer.with_updates(
        name="DYNAPEXWarmup",
        default_config=DYNAPEX_DEFAULT_CONFIG,
        default_policy=DQNTFWarmupPolicy,
        execution_plan=apex_execution_plan)
)

register_trainable(
    "TTFPPO",
    PPOTrainer.with_updates(
        name="TTFPPOTrainer", get_policy_class=lambda c: TTFPPOPolicy
    ),
)
