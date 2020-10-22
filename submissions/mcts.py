import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool

from flatlander.envs.observations.path_obs import PathObservationBuilder
from flatlander.mcts.mcts import MonteCarloTreeSearch

env = RailEnv(width=25, height=25,
              rail_generator=sparse_rail_generator(),
              number_of_agents=5,
              obs_builder_object=PathObservationBuilder())

mcts = MonteCarloTreeSearch(2, epsilon=1, rollout_depth=10000)

obs, _ = env.reset()

env_renderer = RenderTool(env)
env_renderer.render_env(show=True, frames=True, show_observations=False)
done = {"__all__": False}

episode_return = 0
while not done["__all__"]:
    action = mcts.get_best_actions(env=env, obs=obs)
    obs, all_rewards, done, _ = env.step(action)
    episode_return += np.sum(list(all_rewards.values()))
    env_renderer.render_env(show=True, frames=True, show_observations=False)
    print("Rewards: ", all_rewards, "  [done=", done, "]")

print("Episode return:", episode_return)
