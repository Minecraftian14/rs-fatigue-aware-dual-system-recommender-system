import gymnasium as gym
from stable_baselines3 import DQN

from .blueprint_model import ModelBlueprint


class ABCNameMeModel(ModelBlueprint):

    def __init__(self):
        super().__init__('ABCNameMeEnv')

    def _create_model(self, env: gym.Env):
        return DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=10000,
            learning_starts=100,
            batch_size=32,
        )
