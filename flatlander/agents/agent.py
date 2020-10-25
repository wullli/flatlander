from abc import abstractmethod, ABC

from flatland.envs.rail_env import RailEnv


class Agent(ABC):

    @abstractmethod
    def compute_actions(self, observation_dict: dict, env: RailEnv):
        raise NotImplementedError()

    @abstractmethod
    def compute_action(self, obs, env: RailEnv, handle):
        raise NotImplementedError()
