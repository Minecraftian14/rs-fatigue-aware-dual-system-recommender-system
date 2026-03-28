import gymnasium as gym
from stable_baselines3 import DQN

from ...models.blueprint_model import ModelFactoryBlueprint


class SEWADQNModelFactory(ModelFactoryBlueprint):
    """
    A Simple Experiment With A DQN Model
    """

    def __init__(self):
        super().__init__('SEWADQNEnvironment')

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
