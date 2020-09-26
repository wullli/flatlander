import os

from ray.rllib.agents import sac, ppo, dqn

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
FINETUNE_BUDGET_S = 60 * 4
CURRENT_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'current_env.pkl'))
