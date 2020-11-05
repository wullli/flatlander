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
    "apex_dqn_1": {
        "checkpoint_paths": {n_agents: os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         f"model_checkpoints/apex_dqn_small_v0/{n_agents}_agents/checkpoint_119/checkpoint-119"))
            for n_agents in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18]},
        "agent": "apex"
    },
    "apex_dqn_robust": {
        "checkpoint_paths": {n_agents: os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         f"model_checkpoints/apex_dqn_robust/checkpoint_27/checkpoint-27"))
            for n_agents in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18]},
        "agent": "apex"
    },
    "impala_priorization": {
        "checkpoint_paths": {n_agents: os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "submissions",
                         f"model_checkpoints/priorization/checkpoint_801/checkpoint-801"))
            for n_agents in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18]},
        "agent": "sac"
    }

}

RUN = SUBMISSIONS["impala_priorization"]
CURRENT_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/current_env.pkl'))
