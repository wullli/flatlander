from ray.rllib import Policy, TFPolicy
from ray.rllib.agents.dqn.apex import ApexTrainer
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy, ComputeTDErrorMixin
from ray.rllib.agents.dqn.simple_q_tf_policy import TargetNetworkMixin
from ray.rllib.utils import try_import_tf, override, DeveloperAPI
from ray.tune import register_trainable

tf1, tf, tfv = try_import_tf()
import numpy as np


@DeveloperAPI
class TransformerLearningRateSchedule:
    """Mixin for TFPolicy that adds a learning rate schedule with warmup."""

    @DeveloperAPI
    def __init__(self, d_model, warmup_steps):
        self.cur_lr = tf1.get_variable("lr", initializer=0.00005, trainable=False)
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
        return tf1.train.AdamOptimizer(self.cur_lr, beta1=0.9, beta2=0.98, epsilon=1e-9)


def setup_mid_mixins(policy: Policy, _1, _2, config) -> None:
    ComputeTDErrorMixin.__init__(policy)
    warmup_steps = config["model"]["custom_model_config"].get("warmup_steps", 100000)
    TransformerLearningRateSchedule.__init__(policy,
                                             config["model"]["custom_options"]["transformer"]["num_heads"],
                                             warmup_steps)


WarmupDQNTFPolicy = DQNTFPolicy.with_updates(
    name="WarmupDQNTFPolicy",
    before_loss_init=setup_mid_mixins,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        TransformerLearningRateSchedule,
    ])

ApexWarmupTrainer = ApexTrainer.with_updates(
    name="APEXWarmup",
    default_policy=WarmupDQNTFPolicy,
    get_policy_class=lambda c: WarmupDQNTFPolicy
)

register_trainable(
    "APEXWarmup",
    ApexWarmupTrainer
)
