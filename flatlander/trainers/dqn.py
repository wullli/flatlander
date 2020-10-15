from ray.rllib import Policy
from ray.rllib.agents.dqn.apex import ApexTrainer
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy, ComputeTDErrorMixin
from ray.rllib.agents.dqn.simple_q_tf_policy import TargetNetworkMixin
from ray.tune import register_trainable

from flatlander.trainers.ppo import TransformerLearningRateSchedule


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
    default_policy=WarmupDQNTFPolicy
)

register_trainable(
    "APEXWarmup",
    ApexWarmupTrainer
)
