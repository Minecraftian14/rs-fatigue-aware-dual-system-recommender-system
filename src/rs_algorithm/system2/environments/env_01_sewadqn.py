import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SEWADQNEnvironment(gym.Env):
    """
    A Simple Experiment With A DQN Model
    """

    def __init__(self, num_items=100, embedding_dim=10, max_steps=20):
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_steps = max_steps

        # Action = recommend item ID
        self.action_space = spaces.Discrete(num_items)

        # State = user embedding + last item embedding
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(embedding_dim * 2,), dtype=np.float32
        )

        # Random item embeddings
        self.item_embeddings = np.random.normal(size=(num_items, embedding_dim))

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 🧠 USER MODEL
        self.user_pref = np.random.normal(size=self.embedding_dim)

        self.current_step = 0
        self.last_item = np.zeros(self.embedding_dim)

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.user_pref, self.last_item]).astype(np.float32)

    def step(self, action):
        item_vec = self.item_embeddings[action]

        # 🎯 Reward = similarity (user likes similar items)
        reward = np.dot(self.user_pref, item_vec)

        # Optional: squash reward
        reward = float(np.tanh(reward))

        # Update last item
        self.last_item = item_vec

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_obs(), reward, done, False, {}
