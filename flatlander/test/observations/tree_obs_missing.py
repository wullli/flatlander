import time

import numpy as np
from flatland.envs.schedule_generators import sparse_schedule_generator

from flatland.envs.malfunction_generators import NoMalfunctionGen, ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatland.core.grid.grid4_utils import get_new_position

from flatland.envs.agent_utils import RailAgentStatus

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator, complex_rail_generator, sparse_rail_generator
from flatland.utils.rendertools import RenderTool

rail_generator = sparse_rail_generator(
    seed=0,
    max_num_cities=4,
    grid_mode=False,
    max_rails_between_cities=2,
    max_rails_in_city=2
)


malfunction_generator = ParamMalfunctionGen(
    MalfunctionParameters(malfunction_rate=10, min_duration=20, max_duration=50))

speed_ratio_map = None
speed_ratio_map = {1: 1}
schedule_generator = sparse_schedule_generator(speed_ratio_map)

n_agents = 5
env = RailEnv(
    width=25,
    height=25,
    rail_generator=rail_generator,
    schedule_generator=schedule_generator,
    number_of_agents=n_agents,
    malfunction_generator=malfunction_generator,
    obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(max_depth=20)),
    remove_agents_at_target=False,
    random_seed=0,
)

env_renderer = None

for _ in range(100):
    if env_renderer is not None:
        env_renderer.close_window()
    obs, _ = env.reset()
    done = {"__all__": False}
    while not done["__all__"]:
        action = {}
        obs, all_rewards, done, _ = env.step(action)
        print("Rewards: ", all_rewards, "  [done=", done, "]")
        print("Observations: ", obs)
        assert len(obs.keys()) == n_agents
