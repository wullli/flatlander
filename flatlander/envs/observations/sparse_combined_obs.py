from typing import Optional, List

import gym

from flatland.core.env_observation_builder import ObservationBuilder
from flatlander.envs.observations import Observation, register_obs, make_obs


@register_obs("sparse_combined")
class CombinedObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.interval = 10
        if config.get('interval', None) is not None:
            self.interval = config['interval']
            del config['interval']
        self._observations = [
            make_obs(obs_name, config[obs_name]) for obs_name in config.keys()
        ]
        self._builder = CombinedObsForRailEnv([
            o._builder for o in self._observations
        ], interval=self.interval)

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        space = []
        for o in self._observations:
            space.append(o.observation_space())
        return gym.spaces.Tuple(space)


class CombinedObsForRailEnv(ObservationBuilder):

    def __init__(self, builders: [ObservationBuilder], interval=10):
        super().__init__()
        self._builders = builders
        self.interval = interval

    def reset(self):
        for b in self._builders:
            b.reset()

    def get(self, handle: int = 0):
        return None

    def get_many(self, handles: Optional[List[int]] = None):
        if self.env._elapsed_steps % self.interval == 0:
            obs = {h: [] for h in handles}
            for b in self._builders:
                sub_obs = b.get_many(handles)
                for h in handles:
                    obs[h].append(sub_obs[h])
            return {h: tuple(o) for h, o in obs.items()}
        else:
            return {h: ([], []) for h in handles}

    def set_env(self, env):
        for b in self._builders:
            b.set_env(env)
        self.env = env
