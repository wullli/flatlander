from ray.rllib import Policy, TFPolicy
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import ValueNetworkMixin, PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import KLCoeffMixin
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule
from ray.rllib.utils import override, DeveloperAPI, try_import_tf
from ray.tune import register_trainable

tf = try_import_tf()


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
        self.cur_lr = tf.multiply(lr, 0.1)

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
        name="TTFPPOTrainer", get_policy_class=lambda c: TTFPPOPolicy
    ),
)
