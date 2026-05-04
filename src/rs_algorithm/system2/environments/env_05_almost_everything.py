from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import gymnasium as gym
from faiss import Index
from gymnasium.core import ActType, ObsType
from typing import SupportsFloat, Any

import torch
from torch.nn import functional as F

from data_loaders import KuaiRecData
from miscellaneous import get_relevant_items, get_semi_relevant_items, get_random_items
from ...models import AdaptiveRepresentation, TwoTowerRecommender, ThreeTowerRecommenderV2
from ...models.user_models.user_02_sim import SimUserModel, norm_dot_product_based_relavance_calculation


def create_norm_press_preference(data_model: KuaiRecData, user_model: AdaptiveRepresentation):
    with torch.no_grad():
        uids = data_model.ratings['user_id'].unique()
        user_model.eval()
        uem, _ = user_model(
            torch.tensor(uids, dtype=torch.int64),
            torch.tensor(data_model.get_user_features(uids), dtype=torch.float32))
    user_embed_mean = uem.mean(dim=1).mean().numpy()
    user_embed_std = uem.std(dim=1).mean().numpy()

    def norm_press_preference(preference):
        preference = (preference - preference.mean()) / preference.std()
        return preference * user_embed_std + user_embed_mean

    return norm_press_preference


class SSREnvironment(gym.Env):
    def __init__(self,

                 data_model: KuaiRecData,
                 d_embed: int,
                 user_model: AdaptiveRepresentation,
                 item_model: AdaptiveRepresentation,
                 s2p_recommender: ThreeTowerRecommenderV2,
                 index: Index,
                 idx2iid: dict[int, int],

                 norm_press_preference=None,

                 user_id=None,

                 n_action=(70, 20, 10),
                 max_steps=1000,
                 max_history_len=100,
                 metrics_window=20

                 ):
        """

        :param n_action: The distribution of actions (relevant item, semi relevant item, random item)
        :param max_steps:
        :param max_history_len:
        :param metrics_window:
        """

        self.data_model = data_model
        self.d_embed = d_embed
        self.user_model = user_model
        self.item_model = item_model
        self.s2p_recommender = s2p_recommender
        self.index = index
        self.idx2iid = idx2iid
        if norm_press_preference is None: norm_press_preference = create_norm_press_preference(data_model, user_model)
        self.norm_press_preference = norm_press_preference
        self.user_id = user_id

        self.n_items = data_model.ratings['item_id'].nunique()

        self.n_action = n_action
        self.max_steps = max_steps
        self.max_history_len = max_history_len
        self.metrics_window = metrics_window

        self.step_count = 0

        self.action_space = gym.spaces.Discrete(np.sum(n_action))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(d_embed + 1 + 4,),  # embeds, bias and 4 internal state monitorable metrics
            dtype=np.float32
        )

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        uid = self.user_id or self.data_model.ratings['user_id'].sample(1).iloc[0]
        uft = self.data_model.get_user_features(uid)
        uid = torch.tensor([uid], dtype=torch.int64)
        uft = torch.tensor(uft[None, :], dtype=torch.float32)
        with torch.no_grad():
            self.user_model.eval()
            uem, ubs = self.user_model(uid, uft)
            uem, ubs = uem.cpu().numpy(), ubs.cpu().numpy()
        self.user_state = {'uid': uid, 'uft': uft}
        self.history_state = {'hid': [], 'hcl': []}

        self.user = SimUserModel(
            uem, ubs,
            relevance_calculator=norm_dot_product_based_relavance_calculation,
            preference_update_manager=self.norm_press_preference,
        )

        self.today = datetime.now()

        self.recent_click_rate = 0
        self.similarity_of_recent_items = 0
        self.diversity_score = 0
        self.session_length = 0

        return self._get_obs(), {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        #
        if not self.user.can_interact_today(self.today):
            self.today += timedelta(days=1)
            self.session_length = 0
            return self._get_obs(), -1, False, False, self.get_internal_representation()

        uid, uft = self.user_state['uid'], self.user_state['uft']
        hid, hcl = self.history_state['hid'], self.history_state['hcl']
        hft = torch.tensor(self.data_model.get_item_features(hid)[None], dtype=torch.float32)
        hid = torch.tensor([hid], dtype=torch.int64)
        hcl = torch.tensor([hcl], dtype=torch.float32)
        msk = torch.tensor([[True] * hft.shape[1]], dtype=torch.bool)
        lns = torch.tensor([hft.shape[1]], dtype=torch.int64)
        with torch.no_grad():
            pem, (_, pbs) = self.s2p_recommender(uid, uft, hid, hft, hcl, msk, lns)
            pem, pbs = pem.cpu().numpy(), pbs.cpu().numpy()

        n_relevant_item_ids, n_semi_relevant_item_ids, n_random_item_ids = self.n_action
        relevant_item_ids = get_relevant_items(self.index, self.idx2iid, pem, n_relevant_item_ids)
        semi_relevant_item_ids = get_semi_relevant_items(self.index, self.idx2iid, pem, n_semi_relevant_item_ids)
        random_item_ids = get_random_items(self.n_items, self.idx2iid, n_random_item_ids)
        item_ids = np.concatenate((relevant_item_ids, semi_relevant_item_ids, random_item_ids))

        iid = item_ids[action]
        ift = self.data_model.get_item_features(iid)
        with torch.no_grad():
            self.item_model.eval()
            iem, ibs = self.item_model(
                torch.tensor([iid], dtype=torch.int64),
                torch.tensor(ift[None, :], dtype=torch.float32))
            iem, ibs = iem.cpu().numpy(), ibs.cpu().numpy()

        did_i_click = self.user.react(self.today, None, iem, ibs)

        self.history_state['hid'].append(iid)
        self.history_state['hcl'].append(did_i_click)
        if len(self.history_state['hid']) > self.max_history_len:
            self.history_state['hid'].pop(0)
            self.history_state['hcl'].pop(0)

        reward = (
                + 2 * float(did_i_click)
                - 8 * (not self.user.can_interact_today(self.today))
                - 100 * self.user.is_gone_for_good()
                - 6 * self.user.fatigue ** 0.25
                - 2 * self.user.discontent ** 0.25
                # - 0.5 * self.similarity_of_recent_items
        )
        # TODO: Add another penalty if the user is fatigued
        # TODO: Add another reward for diversity
        # TODO: Add another penalty for repetition

        self.step_count += 1

        if hcl.shape[1] > 0:
            self.recent_click_rate = hcl[:, -self.metrics_window:].mean().item()
        if hcl.shape[1] > 1:
            hem, _ = self.item_model(hid[0, -self.metrics_window:], hft[0, -self.metrics_window:])
            self.similarity_of_recent_items = (F.normalize(hem[:-1], dim=1) * F.normalize(hem[1:], dim=1)).sum(dim=1).mean().item()
            self.diversity_score = 1 - self.similarity_of_recent_items

        self.session_length += 1

        terminated = self.user.is_gone_for_good()
        truncated = self.step_count >= self.max_steps

        return self._get_obs(pem, pbs), reward, terminated, truncated, self.get_internal_representation(iem, ibs)

    def _get_obs(self, pem=None, pbs=None):
        if pem is None: pem = self.user._state_vector['preference']
        if pbs is None: pbs = self.user._state_vector['preference_bias']
        return np.hstack([
            pem.flatten(),
            np.asarray([
                (pbs or [0.0])[0],
                self.recent_click_rate,
                self.similarity_of_recent_items,
                self.diversity_score,
                self.session_length,
            ], dtype=np.float32),
        ])

    def get_internal_representation(self, iem=None, ibs=None):
        return {
            'user_state': self.user.get_internal_representation(),
            'step_state': {
                'iem': iem,
                'ibs': ibs,
                'recent_click_rate': self.recent_click_rate,
                'similarity_of_recent_items': self.similarity_of_recent_items,
                'diversity_score': self.diversity_score,
                'session_length': self.session_length,
            },
        }
