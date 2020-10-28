from typing import Optional, List, Dict

import gym
import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatlander.envs.observations import Observation, register_obs


@register_obs("meta")
class MetaObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._builder = MetaObservationBuilder(config['height'], config['width'], config['max_t'])

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        obs_shape = self._builder.observation_shape
        return gym.spaces.Tuple([
            gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32),  # own density map
            gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),  # own distance to target & nr agents at start
        ])


class MetaObservationBuilder(ObservationBuilder):

    def __init__(self, height, width, max_t=20):
        super().__init__()
        self._height = height
        self._width = width
        self._encode = lambda t: np.exp(-t / np.sqrt(max_t))
        self._predictor = ShortestPathPredictorForRailEnv(max_t)

    @property
    def observation_shape(self):
        return self._height, self._width, 3

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """
        get density maps for agents and compose the observation with agent's and other's density maps
        """
        if self.env._elapsed_steps == 0:
            self._predictions = self._predictor.get()
            distance_map = self.env.distance_map.get()
            nan_inf_mask = ((distance_map != np.inf) * (np.abs(np.isnan(distance_map) - 1))).astype(np.bool)
            max_distance = np.max(distance_map[nan_inf_mask])
            density_maps = dict()
            for handle in handles:
                density_maps[handle] = self.get(handle)
            obs = dict()
            for handle in handles:
                stacked_obs = np.zeros(shape=self.observation_shape, dtype=np.float32)
                agent = self.env.agents[handle]
                init_pos = agent.initial_position
                init_dir = agent.initial_direction
                nr_agents_same_start = len([a.handle for a in self.env.agents
                                            if a.initial_position == agent.initial_position])
                init_pos_map = np.zeros(shape=(self._height, self._width), dtype=np.float32)
                init_pos_map[init_pos] = 1
                other_dens_maps = [density_maps[key] for key in density_maps if key != handle]
                others_density = np.mean(np.array(other_dens_maps), axis=0)
                distance = distance_map[handle][init_pos + (init_dir,)]
                distance = max_distance if (
                        distance == np.inf or np.isnan(distance)) else distance

                stacked_obs[:, :, 0] = density_maps[handle]
                stacked_obs[:, :, 1] = others_density
                stacked_obs[:, :, 2] = init_pos_map

                obs[handle] = (stacked_obs,
                               np.array([distance / max_distance, nr_agents_same_start / len(self.env.agents)]))
            return obs
        else:
            return {h: np.zeros(1) for h in handles}

    def get(self, handle: int = 0):
        """
        compute density map for agent: a value is asigned to every cell along the shortest path between
        the agent and its target based on the distance to the agent, i.e. the number of time steps the
        agent needs to reach the cell, encoding the time information.
        """
        density_map = np.zeros(shape=(self._height, self._width), dtype=np.float32)
        if self._predictions[handle] is not None:
            for t, prediction in enumerate(self._predictions[handle]):
                p = tuple(np.array(prediction[1:3]).astype(int))
                density_map[p] = self._encode(t)
        return density_map

    def set_env(self, env: Environment):
        self.env: RailEnv = env
        self._predictor.set_env(self.env)

    def reset(self):
        pass
