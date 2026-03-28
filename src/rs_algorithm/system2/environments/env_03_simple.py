import gymnasium as gym
from gymnasium import spaces
import numpy as np

from data_loaders import DataModel
from ...models.user_models.blueprint_user_model import UserModel
from ...models.user_models.user_01_dpr import DPRUserModel


class SimpleEnvironment(gym.Env):
    """
    State = [preference, fatigue, memory]
    Action = Next Item
    Reward = f(click, fatigue_penalty)
    Termination = fatigue > threshold OR too many no-clicks
    """

    def __init__(self, data: DataModel, user: UserModel, max_steps=50):
        super().__init__()

        self.item_embeds = data.get_items_embeds()
        self.n_items, self.d_item_embeds = data.get_n_items(), data.get_d_item_embeds()

        self.max_steps = max_steps
        self.step_count = 0

        self.user = user

        self.action_space = spaces.Discrete(self.n_items)

        # observation = [user_pref + fatigue]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.d_item_embeds + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.user.reset(seed=seed)
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        item = self.item_embeds[action][None, :]

        clicked, fatigue = self.user.react(None, None, item)
        reward = float(clicked) - fatigue

        self.step_count += 1

        terminated = fatigue >= self.user.mental_capacity
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "clicked": clicked,
            "fatigue": fatigue,
        }

    def _get_obs(self):
        fatigue = self.user.fatigue or self.user.fatigue_buildup / self.user.mental_capacity
        return np.hstack([self.user.preference, [[fatigue]]])
