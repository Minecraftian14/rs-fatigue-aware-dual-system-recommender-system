import gymnasium as gym
from abc import ABC, abstractmethod


class ModelBlueprint(ABC):
    def __init__(self, env_name: str = None):
        self.env_name = env_name

    def create_model(self, env: gym.Env = None):
        if self.env_name is not None:
            assert env.__class__.__name__ == self.env_name
        return self._create_model(env)

    @abstractmethod
    def _create_model(self, env: gym.Env = None): ...
