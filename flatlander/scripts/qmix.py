"""The two-step game from QMIX: https://arxiv.org/pdf/1803.11485.pdf
Configurations you can try:
    - normal policy gradients (PG)
    - contrib/MADDPG
    - QMIX
See also: centralized_critic.py for centralized critic PPO on this game.
"""

import argparse
from gym.spaces import Tuple, MultiDiscrete, Dict, Discrete

import ray
from ray import tune
from ray.rllib.agents.qmix.qmix_policy import ENV_STATE
from ray.rllib.examples.twostep_game import TwoStepGame
from ray.tune import register_env, grid_search

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="QMIX")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--stop-reward", type=float, default=7.0)
parser.add_argument("--stop-timesteps", type=int, default=50000)

if __name__ == "__main__":
    args = parser.parse_args()

    grouping = {
        "group_1": [0, 1],
    }
    obs_space = Tuple([
        Dict({
            "obs": MultiDiscrete([2, 2, 2, 3]),
            ENV_STATE: MultiDiscrete([2, 2, 2])
        }),
        Dict({
            "obs": MultiDiscrete([2, 2, 2, 3]),
            ENV_STATE: MultiDiscrete([2, 2, 2])
        }),
    ])
    act_space = Tuple([
        TwoStepGame.action_space,
        TwoStepGame.action_space,
    ])
    register_env(
        "grouped_twostep",
        lambda config: TwoStepGame(config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

    config = {
        "rollout_fragment_length": 4,
        "train_batch_size": 32,
        "exploration_config": {
            "epsilon_timesteps": 5000,
            "final_epsilon": 0.05,
        },
        "num_workers": 0,
        "mixer": grid_search([None, "qmix", "vdn"]),
        "env_config": {
            "separate_state_space": True,
            "one_hot_state_encoding": True
        },
    }
    group = True

    ray.init(num_cpus=args.num_cpus or None)

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
    }

    config = dict(config, **{
        "env": "grouped_twostep",
    })

    results = tune.run(args.run, stop=stop, config=config, verbose=1)

    ray.shutdown()
