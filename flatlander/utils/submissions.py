import os

import numpy as np
from ray.rllib.agents import sac, ppo, dqn


def _get_tune_time_schedule():
    agent_nums = np.array(list(range(100)))
    times = np.ones(100)
    times[:6] *= 60
    times[6:18] *= 2 * 60
    times[18:100] *= 4 * 60 + 30

    return dict(zip(agent_nums, times))


TUNE_TIME_SCHEDULE = _get_tune_time_schedule()


def get_tune_time(n_agents):
    if n_agents <= 100:
        return TUNE_TIME_SCHEDULE[n_agents]
    else:
        return 4 * 60


agent_map = {"sac": sac.SACTrainer,
             "ppo": ppo.PPOTrainer,
             "dqn": dqn.DQNTrainer,
             "apex": dqn.ApexTrainer}

SUBMISSIONS = {
    "apex_dqn_1": {
        "checkpoint_path": os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         "model_checkpoints/apex_dqn_small_v0/checkpoint_119/checkpoint-119")),
        "agent": agent_map["apex"]
    },
    "sac_small_v0": {
        "checkpoint_path": os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         "model_checkpoints/sac_small_v0/checkpoint-51089/checkpoint-51089")),
        "agent": agent_map["sac"]
    },
    "ttf_1": {
        "checkpoint_path": os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         "model_checkpoints/tree_tf_1/checkpoint_5860_scaling/checkpoint-5860")),
        "agent": agent_map["sac"]
    }}

RUN = SUBMISSIONS["apex_dqn_1"]
CURRENT_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'current_env.pkl'))
