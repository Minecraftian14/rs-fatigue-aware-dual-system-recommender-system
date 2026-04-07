from datetime import datetime, timedelta
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

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

    def __init__(self,
                 data: DataModel,
                 user: UserModel,
                 max_steps=50,
                 history_length=10,
                 ):
        super().__init__()

        self.item_embeds = data.get_items_embeds()
        self.n_items, self.d_item_embeds = data.get_n_items(), data.get_d_item_embeds()
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(self.n_items)
        # observation = [user_pref + fatigue]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(history_length * self.d_item_embeds + history_length,),
            dtype=np.float32
        )

        self.user = user
        self.step_count = 0
        self.today = datetime.now()
        self.history = deque(maxlen=history_length)

        for _ in range(history_length):
            self.history.append((
                np.zeros((self.d_item_embeds,)),
                0.0,
            ))

    def _get_obs(self):
        items, rewards = zip(*self.history)
        items, rewards = np.asarray(items), np.asarray(rewards)
        return np.concatenate([
            items.flatten(),  # (history_length * d_item_embeds,)
            rewards  # (history_length,)
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.user.reset(seed=seed)
        self.step_count = 0
        self.today = datetime.now()

        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0

        if self.user.can_interact_today():
            item = self.item_embeds[action]

            clicked = self.user.react(self.today, None, item[None, :])
            user_state = self.user.get_internal_representation()

            # Add click reward
            reward += 1.0 * clicked
            # Add fatigue penalty
            if 'fatigue' in user_state: reward -= 1.0 * user_state['fatigue']
            if 'is_fatigued' in user_state: reward -= 2.0 * user_state['is_fatigued']  # Some big number
            # Add discontentment penalty
            if 'discontent' in user_state: reward -= 1.0 * user_state['discontent']
            if 'is_discontent' in user_state: reward -= 10.0 * user_state['is_discontent']  # Some big number

            self.history.append((item, reward))

        else:
            reward = -0.1  # Ambient pressure, though not really required because of the heavy is_fatigued penalty
            self.today += timedelta(days=1)
            user_state = self.user.get_internal_representation()

            self.history.append((np.zeros((self.d_item_embeds,)), reward))

        terminated = self.user.is_gone_for_good()

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, user_state
