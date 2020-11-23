import os

import numpy as np
from ray.rllib.agents import sac, ppo, dqn, impala


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


AGENT_MAP = {"sac": sac.SACTrainer,
             "ppo": ppo.PPOTrainer,
             "dqn": dqn.DQNTrainer,
             "impala": impala.ImpalaTrainer,
             "apex": dqn.ApexTrainer}

SUBMISSIONS = {
    "ato": {
        "checkpoint_paths": os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         f"model_checkpoints/ato_baseline/checkpoint_119/checkpoint-119")),
        "agent": "apex"
    },
    "apto": {
        "checkpoint_paths": os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         f"model_checkpoints/apto/checkpoint_86/checkpoint-86")),
        "agent": "apex"
    },
    "rlps-tcpr": {
        "checkpoint_paths": os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         f"model_checkpoints/rlps-tcpr/checkpoint_27/checkpoint-27")),

        "agent": "apex"
    },
    "rlps-tcpr-2": {
        "checkpoint_paths": os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         f"model_checkpoints/rlps-tcpr/retrained/checkpoint-291")),
        "agent": "apex"
    },
    "rlpr-tcpr": {
        "checkpoint_paths": os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         f"model_checkpoints/rlpr-tcpr/checkpoint_76/checkpoint-76")),
        "agent": "ppo"
    },
    "rlpr-tcpr-2": {
        "checkpoint_paths": os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         f"model_checkpoints/rlpr-tcpr/checkpoint_1496/checkpoint-1496")),
        "agent": "ppo"
    }

}

RUN = SUBMISSIONS["rlpr-tcpr"]
CURRENT_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/current_env.pkl'))
