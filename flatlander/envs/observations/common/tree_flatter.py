from abc import ABC, abstractmethod
from typing import Any


class TreeFlattener(ABC):

    @abstractmethod
    def flatten(self, root: Any, **kwargs):
        raise NotImplementedError("This method needs to be implemented in a subclass")