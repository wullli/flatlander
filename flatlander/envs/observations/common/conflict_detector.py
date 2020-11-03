from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

from flatland.envs.rail_env import RailEnv


class ConflictDetector(ABC):
    def __init__(self):
        self.rail_env: Optional[RailEnv] = None

    @abstractmethod
    def detect_conflicts(self, handles=None, positions=None, directions=None) -> (defaultdict, defaultdict):
        raise NotImplementedError()

    def set_env(self, rail_env: RailEnv):
        self.rail_env = rail_env
