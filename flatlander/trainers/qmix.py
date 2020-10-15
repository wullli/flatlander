from ray.rllib.agents.dqn.apex import apex_execution_plan
from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer
from ray.rllib.agents.qmix import DEFAULT_CONFIG
from ray.rllib.agents.qmix.qmix_policy import QMixTorchPolicy
from ray.tune import register_trainable
from ray.tune.utils import merge_dicts

QMIX_APEX_DEFAULT_CONFIG = merge_dicts(
    DEFAULT_CONFIG,  # see also the options in dqn.py, which are also supported
    {
        "optimizer": {
            "max_weight_sync_delay": 400,
            "num_replay_buffer_shards": 4,
            "debug": False
        },
        "n_step": 3,
        "num_gpus": 1,
        "num_workers": 32,
        "buffer_size": 2000000,
        "learning_starts": 50000,
        "train_batch_size": 512,
        "rollout_fragment_length": 50,
        "target_network_update_freq": 500000,
        "timesteps_per_iteration": 1000,
        "exploration_config": {"type": "PerWorkerEpsilonGreedy"},
        "worker_side_prioritization": True,
        "min_iter_time_s": 30,
        "training_intensity": None,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "final_prioritized_replay_beta": 0.4,
        "prioritized_replay_beta_annealing_timesteps": 20000,
        "prioritized_replay_eps": 1e-6,
    },
)

QMixTrainer = GenericOffPolicyTrainer.with_updates(
    name="QMIXApex",
    default_config=QMIX_APEX_DEFAULT_CONFIG,
    default_policy=QMixTorchPolicy,
    get_policy_class=None,
    execution_plan=apex_execution_plan)

register_trainable(
    "QMIXApex",
    QMixTrainer
)
