import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from data_loaders import DataModel
from .blueprint_user_model import UserModel
from miscellaneous import *


class DPRUserModel(UserModel):
    """
    Simple Dot Product-Based Relevance Model
    """

    def __init__(self,
                 data: DataModel,
                 mental_capacity=10,
                 similarity_penalty=1,
                 dissimilarity_reward=1.1,
                 ):
        super(DPRUserModel, self).__init__()
        self.data = data
        self.mental_capacity = mental_capacity
        self.similarity_penalty = similarity_penalty
        self.dissimilarity_reward = dissimilarity_reward
        n_items = self.data.get_n_items()
        self.preference = self.data.get_items_embeds(random.randint(0, n_items - 1))[None, :]
        self.fatigue_buildup: float = None
        self.fatigue: float = None
        self.reset()

    def reset(self, seed=None):
        self.fatigue_buildup = 0

    def react(self, time_step, device, item):
        relevance = cosine_similarity(self.preference, item).squeeze()
        relevance = relevance / 2 + 0.5  # Snap to [0, 1] range for probability

        self.fatigue_buildup += lerp(-self.dissimilarity_reward, self.similarity_penalty, relevance)
        self.fatigue_buildup = clip(self.fatigue_buildup, 0, self.mental_capacity)

        self.fatigue = self.fatigue_buildup / self.mental_capacity
        click_probability = lerp(relevance, 0, self.fatigue)
        clicked = random.random() < click_probability

        return clicked.squeeze()


class DPRIMUserModel(UserModel):
    """
    Simple Dot Product-Based Relevance Model with Item Memory
    """

    def __init__(self,
                 data: DataModel,
                 max_memory=5,
                 ):
        super(DPRIMUserModel, self).__init__()
        n_items = data.get_n_items()
        self.preference = data.get_items_embeds(random.randint(0, n_items - 1))
        self.max_memory = max_memory
        self.fatigue = 0
        self.memory = []

    def react(self, time_step, device, item):
        relevance = cosine_similarity(self.preference, item)
        relevance = relevance / 2 + 1  # Snap to [0, 1] range for probability

        if self.memory:
            sims = [np.dot(item, h) for h in self.memory]
            repetition = max(sims)
        else:
            repetition = 0

        fatigue_penalty = self.fatigue * repetition

        score = 3 * relevance - fatigue_penalty
        prob = 1 / (1 + np.exp(-score))

        clicked = np.random.rand() < prob

        # update state
        self.memory.append(item)
        if len(self.memory) > self.max_memory: self.memory.pop(0)
        self.fatigue = 0.9 * self.fatigue + repetition

        return clicked
