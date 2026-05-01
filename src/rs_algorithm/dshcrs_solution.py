import os

import faiss
import numpy as np
import stable_baselines3 as sb
import torch
from stable_baselines3.common.env_util import make_vec_env

from data_loaders import KuaiRecData
from dl_trainer import Trainer
from miscellaneous import get_relevant_items
from .baselines import Solution
from .models import AdaptiveRepresentation
from .models import TwoTowerRecommender, ThreeTowerRecommenderV2, ConcatInteractionHistoryRepresentationWithUserEmbedInitialization
from .system2.environments.env_05_almost_everything import SSREnvironment, create_norm_press_preference


class DSHCRSSolution(Solution):
    def __init__(self, data_model: KuaiRecData, root=os.path.join('temp', 'checkpoints')):
        super().__init__(data_model)
        self.root = root
        self.live_env = {}

    def prepare(self):
        d_embed = 128
        n_users, d_user_feats = self.data_model.ratings['user_id'].nunique(), self.data_model.user_features.shape[1]
        n_items, d_item_feats = self.data_model.ratings['item_id'].nunique(), self.data_model.item_features.shape[1]

        # LOAD THE EMBEDDINGS
        uid2idx = {v: i for i, v in enumerate(np.unique(self.data_model.ratings['user_id']))}
        user_model = AdaptiveRepresentation(n_users, d_user_feats, d_hidden=512, d_embed=d_embed, id2idx=uid2idx)
        iid2idx = {v: i for i, v in enumerate(np.unique(self.data_model.ratings['item_id']))}
        item_model = AdaptiveRepresentation(n_items, d_item_feats, d_hidden=512, d_embed=d_embed, id2idx=iid2idx)
        dot_embed_model = TwoTowerRecommender(user_model, item_model)
        Trainer(dot_embed_model, None, model_dir=self.root).load_model('model_id_trained_embedder_v3_128.pt')

        # LOAD THE SYSTEM 1
        for mod in (user_model, item_model):
            for par in mod.parameters():
                par.requires_grad = False

        history_model = ConcatInteractionHistoryRepresentationWithUserEmbedInitialization(d_embed, 512)
        s1recommender = ThreeTowerRecommenderV2(d_embed, user_model, item_model, history_model)
        Trainer(s1recommender, None, model_dir=self.root).load_model('model_recommender_v6.pt')

        # LOAD THE ANN
        index = faiss.IndexFlatIP(d_embed)
        idx2iid = {}
        with torch.no_grad():
            for idx, iid in enumerate(self.data_model.ratings['item_id'].unique()):
                item_model.eval()
                iem, *_ = item_model(
                    torch.tensor([iid], dtype=torch.int64),
                    torch.tensor(self.data_model.get_item_features(iid)[None, :], dtype=torch.float32))
                index.add(iem.numpy())
                idx2iid[idx] = iid

        # LOAD THE SYSTEM 2
        norm_press_preference = create_norm_press_preference(self.data_model, user_model)

        def create_env(user_id=None):
            return SSREnvironment(self.data_model, d_embed, user_model, item_model, s1recommender, index, idx2iid, norm_press_preference, user_id)

        s2recommender = sb.DQN.load(os.path.join(self.root, 'dqn_system_2'), make_vec_env(create_env, n_envs=8))

        self.d_embed, self.n_users, self.n_items, self.d_user_feats, self.d_item_feats = d_embed, n_users, n_items, d_user_feats, d_item_feats
        self.uid2idx, self.user_model, self.iid2idx, self.item_model, self.dot_embed_model = uid2idx, user_model, iid2idx, item_model, dot_embed_model
        self.history_model, self.s1recommender = history_model, s1recommender
        self.index, self.idx2iid, self.norm_press_preference, self.create_env, self.s2recommender = index, idx2iid, norm_press_preference, create_env, s2recommender

    def get_recommendations(self, user_ids: list[int], top_k=10, filter_already_liked_items=False) -> list[list[int]]:
        envs = list(map(self._get_live_env, user_ids))
        recs = []
        for i in range(len(envs)):
            env, obs = envs[i]
            action, _ = self.s2recommender.predict(obs, deterministic=True)
            obs, *_, info = env.step(action)
            envs[i] = (env, obs)
            iem = info['step_state']['iem']
            recs.append(get_relevant_items(env.index, env.idx2iid, iem, top_k))
        return recs

    def _get_live_env(self, user_id: int):
        if user_id not in self.live_env:
            env = self.create_env(user_id)
            self.live_env[user_id] = (env, env.reset()[0])
        return self.live_env[user_id]
