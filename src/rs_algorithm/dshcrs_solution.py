import os

import faiss
import numpy as np
import stable_baselines3 as sb
from stable_baselines3.common.env_util import make_vec_env

import torch
import torch.nn.functional as F

from data_loaders import KuaiRecData
from dl_trainer import Trainer
from miscellaneous import get_relevant_items, get_semi_relevant_items, get_random_items
from .baselines import Solution
from .models import AdaptiveRepresentation
from .models import TwoTowerRecommender, ThreeTowerRecommenderV2, ConcatInteractionHistoryRepresentationWithUserEmbedInitialization
from .system2.environments.env_05_almost_everything import SSREnvironment, create_norm_press_preference


class EmbedDotSolution(Solution):
    def __init__(self,

                 data_model: KuaiRecData,
                 root=os.path.join('temp', 'checkpoints'),

                 d_embed=128,
                 d_hidden=512,

                 embed_model_name='model_id_trained_embedder_v3_128.pt',

                 ):
        super().__init__(data_model)
        self.root = root
        self.history_state = {}

        n_users, d_user_feats = self.data_model.ratings['user_id'].nunique(), self.data_model.user_features.shape[1]
        n_items, d_item_feats = self.data_model.ratings['item_id'].nunique(), self.data_model.item_features.shape[1]

        # LOAD THE EMBEDDINGS
        uid2idx = {v: i for i, v in enumerate(np.unique(self.data_model.ratings['user_id']))}
        user_model = AdaptiveRepresentation(n_users, d_user_feats, d_hidden=d_hidden, d_embed=d_embed, id2idx=uid2idx)
        iid2idx = {v: i for i, v in enumerate(np.unique(self.data_model.ratings['item_id']))}
        item_model = AdaptiveRepresentation(n_items, d_item_feats, d_hidden=d_hidden, d_embed=d_embed, id2idx=iid2idx)
        dot_embed_model = TwoTowerRecommender(user_model, item_model)
        Trainer(dot_embed_model, None, model_dir=self.root).load_model(embed_model_name)

        # LOAD THE ANN
        index = faiss.IndexFlatIP(d_embed)
        idx2iid = {}
        for idx, iid in enumerate(self.data_model.ratings['item_id'].unique()):
            iem, _ = item_model.embed(iid, self.data_model.get_item_features(iid))
            index.add(iem)
            idx2iid[idx] = iid

        self.d_embed, self.n_users, self.n_items, self.d_user_feats, self.d_item_feats = d_embed, n_users, n_items, d_user_feats, d_item_feats
        self.uid2idx, self.user_model, self.iid2idx, self.item_model, self.dot_embed_model = uid2idx, user_model, iid2idx, item_model, dot_embed_model
        self.index, self.idx2iid = index, idx2iid

    def get_recommendations(self, user_ids: list[int], top_k=10, filter_already_liked_items=False, last_item_id=None, did_user_click=None) -> list[list[int]]:
        uem, _ = self.user_model.embed(user_ids, self.data_model.get_user_features(user_ids))
        return get_relevant_items(self.index, self.idx2iid, uem, top_k).reshape(len(user_ids), top_k).tolist()


class HistoryModelS1Solution(EmbedDotSolution):

    def __init__(self,

                 data_model: KuaiRecData,
                 root=os.path.join('temp', 'checkpoints'),

                 d_embed=128,
                 d_hidden=512,
                 metrics_window=10,

                 embed_model_name='model_id_trained_embedder_v3_128.pt',
                 s1_model_name='model_recommender_v6.pt',

                 ):
        super().__init__(data_model, root, d_embed, d_hidden, embed_model_name)
        self.metrics_window = metrics_window

        # LOAD THE SYSTEM 1
        history_model = ConcatInteractionHistoryRepresentationWithUserEmbedInitialization(d_embed, d_hidden)
        s1recommender = ThreeTowerRecommenderV2(d_embed, self.user_model, self.item_model, history_model)
        Trainer(s1recommender, None, model_dir=self.root).load_model(s1_model_name)

        self.history_model, self.s1recommender = history_model, s1recommender

        self.history_state = {}

    def reset_for_evaluation(self):
        self.history_state = {}

    def get_recommendations(self, user_ids: list[int], top_k=10, filter_already_liked_items=False, last_item_id=None, did_user_click=None) -> list[list[int]]:
        uid, uft = user_ids, self.data_model.get_user_features(user_ids)
        hid, hcl = self._get_history_state(user_ids, last_item_id, did_user_click)
        hft = [self.data_model.get_item_features(x) for x in hid]
        hem, (_, _) = self.s1recommender.embed(uid, uft, hid, hft, hcl)
        self._update_history_state(user_ids, hem)
        return get_relevant_items(self.index, self.idx2iid, hem, top_k).reshape(len(user_ids), top_k).tolist()

    def _get_history_state(self, user_ids: list[int], last_item_id: list[int], did_user_click: list[int]) -> tuple[list[int], list[int]]:
        if last_item_id is None: last_item_id = [None] * len(user_ids)
        if did_user_click is None: did_user_click = [None] * len(user_ids)
        hid, hcl = [], []
        for uid, lii, duc in zip(user_ids, last_item_id, did_user_click):
            if uid not in self.history_state:
                self.history_state[uid] = {
                    'hid': [],
                    'hcl': [],
                    'hem': [],
                    'recent_click_rate': 0.0,
                    'similarity_of_recent_items': 0.0,
                    'diversity_score': 1.0,
                    'session_length': 0.0,
                }

            history_state = self.history_state[uid]

            if lii is not None and duc is not None:
                history_state['hid'].append(lii)
                history_state['hcl'].append(duc)

            hid.append(history_state['hid'])
            hcl.append(history_state['hcl'])

        return hid, hcl

    def _update_history_state(self, user_ids: list[int], hem:np.ndarray):
        for idx, uid in enumerate(user_ids):
            history_state = self.history_state[uid]
            history_state['hem'].append(hem[idx])
            pem = history_state['hem']

            history_state['recent_click_rate'] = np.mean(history_state['hcl'][-self.metrics_window:]) if len(history_state['hcl']) > 0 else 0.0
            # TODO: Debug empty slice with mean issue
            history_state['similarity_of_recent_items'] = ((pem[:-1] / np.linalg.norm(pem[:-1], axis=1, keepdims=True)) * (pem[1:] / np.linalg.norm(pem[1:], axis=1, keepdims=True))).sum(axis=1).mean() if len(pem) >= 2 else 0.0
            history_state['diversity_score'] = 1.0 - history_state['similarity_of_recent_items']
            history_state['session_length'] = 1.0 + history_state['session_length']


class RLModelS2Solution(HistoryModelS1Solution):

    def __init__(self,

                 data_model: KuaiRecData,
                 root=os.path.join('temp', 'checkpoints'),

                 d_embed=128,
                 d_hidden=512,
                 metrics_window=10,
                 n_action=(70, 20, 10),

                 embed_model_name='model_id_trained_embedder_v3_128.pt',
                 s1_model_name='model_recommender_v6.pt',
                 s2_model_name='ppo_system_2_v3',

                 ):
        super().__init__(data_model, root, d_embed, d_hidden, metrics_window, embed_model_name, s1_model_name)
        self.n_action = n_action

        # LOAD THE SYSTEM 2
        norm_press_preference = create_norm_press_preference(self.data_model, self.user_model)

        def create_env(user_id=None):
            return SSREnvironment(self.data_model, d_embed, self.user_model, self.item_model, self.s1recommender, self.index, self.idx2iid, norm_press_preference, user_id)

        # s2recommender = sb.DQN.load(os.path.join(self.root, s2_model_name), make_vec_env(create_env, n_envs=8))
        s2recommender = sb.PPO.load(os.path.join(self.root, s2_model_name), make_vec_env(create_env, n_envs=8))

        self.norm_press_preference, self.create_env, self.s2recommender = norm_press_preference, create_env, s2recommender

    def get_recommendations(self, user_ids: list[int], top_k=10, filter_already_liked_items=False, last_item_id=None, did_user_click=None) -> list[list[int]]:
        uid, uft = user_ids, self.data_model.get_user_features(user_ids)
        hid, hcl = self._get_history_state(user_ids, last_item_id, did_user_click)
        hft = [self.data_model.get_item_features(x) for x in hid]
        hem, (_, hbs) = self.s1recommender.embed(uid, uft, hid, hft, hcl)
        self._update_history_state(user_ids, hem)

        n_relevant_item_ids, n_semi_relevant_item_ids, n_random_item_ids = self.n_action
        relevant_item_ids = get_relevant_items(self.index, self.idx2iid, hem, n_relevant_item_ids)
        semi_relevant_item_ids = get_semi_relevant_items(self.index, self.idx2iid, hem, n_semi_relevant_item_ids)
        random_item_ids = get_random_items(self.n_items, self.idx2iid, n_random_item_ids)
        item_ids = np.concatenate((relevant_item_ids, semi_relevant_item_ids, random_item_ids))

        obses = self._create_obses(uid, hem, hbs)

        try:
            actions = [self.s2recommender.predict(obs, deterministic=True) for obs in obses]
        except Exception as e:
            raise e

        iid = [item_ids[action[0]] for action in actions]
        iem, _ = self.item_model.embed(iid, self.data_model.get_item_features(iid))

        return get_relevant_items(self.index, self.idx2iid, iem, top_k).reshape(len(user_ids), top_k).tolist()

    def _create_obses(self, user_ids: list[int], hem, hbs):

        obs = []
        for idx, uid in enumerate(user_ids):
            if hem.shape[1] == 0:
                pem, pbs = self.user_model.embed(uid, self.data_model.get_user_features(uid))
            else:
                pem, pbs = hem[idx], hbs[idx]
            hs = self.history_state[uid]
            obs.append(np.hstack([
                pem.flatten(),
                np.asarray([
                    pbs.flatten()[0],
                    hs['recent_click_rate'],
                    hs['similarity_of_recent_items'],
                    hs['diversity_score'],
                    hs['session_length'],
                ], dtype=np.float32),
            ]))
        return obs


class DSHCRSSolution(Solution):
    def __init__(self, data_model: KuaiRecData, root=os.path.join('temp', 'checkpoints')):
        super().__init__(data_model)
        self.root = root
        self.live_env = {}
        self.history_state = {}

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

    def get_recommendations(self, user_ids: list[int], top_k=10, filter_already_liked_items=False, last_item_id=None, did_user_click=None) -> list[list[int]]:
        results = []

        for uid in user_ids:
            uft = self.data_model.get_user_features(uid)
            history_state = self._get_history_state(uid)
            if last_item_id is not None:
                history_state[0].append(last_item_id)
                history_state[1].append(did_user_click)
            hid, hcl = history_state[0], history_state[1]
            hft = torch.tensor(self.data_model.get_item_features(hid)[None], dtype=torch.float32)
            hid = torch.tensor([hid], dtype=torch.int64)
            hcl = torch.tensor([hcl], dtype=torch.float32)
            msk = torch.tensor([[True] * hft.shape[1]], dtype=torch.bool)
            lns = torch.tensor([hft.shape[1]], dtype=torch.int64)
            with torch.no_grad():
                pem, (_, pbs) = self.s1recommender(
                    torch.tensor([uid], dtype=torch.int64),
                    torch.tensor(uft, dtype=torch.float32),
                    hid, hft, hcl, msk, lns)
                pem, pbs = pem.cpu().numpy(), pbs.cpu().numpy()

            n_relevant_item_ids, n_semi_relevant_item_ids, n_random_item_ids = (70, 20, 10)
            relevant_item_ids = get_relevant_items(self.index, self.idx2iid, pem, n_relevant_item_ids)
            semi_relevant_item_ids = get_semi_relevant_items(self.index, self.idx2iid, pem, n_semi_relevant_item_ids)
            random_item_ids = get_random_items(self.n_items, self.idx2iid, n_random_item_ids)
            item_ids = np.concatenate((relevant_item_ids, semi_relevant_item_ids, random_item_ids))

            obs = self._create_obs(uid)
            action = self.s2recommender.predict(obs, deterministic=True)
            iid = item_ids[action]
            ift = self.data_model.get_item_features(iid)
            with torch.no_grad():
                self.item_model.eval()
                iem, ibs = self.item_model(
                    torch.tensor([iid], dtype=torch.int64),
                    torch.tensor(ift[None, :], dtype=torch.float32))
                iem, _ = iem.cpu().numpy(), ibs.cpu().numpy()

            recs = get_relevant_items(self.index, self.idx2iid, pem, top_k)
            results.append(recs)

            if hcl.shape[1] > 0:
                self.recent_click_rate = hcl[:, -self.metrics_window:].mean().item()
            if hcl.shape[1] > 1:
                hem, _ = self.item_model(hid[0, -self.metrics_window:], hft[0, -self.metrics_window:])
                self.similarity_of_recent_items = (F.normalize(hem[:-1], dim=1) * F.normalize(hem[1:], dim=1)).sum(dim=1).mean().item()
                self.diversity_score = 1 - self.similarity_of_recent_items

        return results

    def _get_history_state(self, user_id: int):
        if user_id not in self.history_state:
            self.history_state[user_id] = ([], [])
        return self.history_state[user_id]

    def _create_obs(self, user_id: int):
        with torch.no_grad():
            self.user_model.eval()
            uem, ubs = self.user_model(
                torch.tensor([user_id], dtype=torch.int64),
                torch.tensor(self.data_model.get_user_features(user_id), dtype=torch.float32))
        return np.hstack([
            uem.flatten(),
            np.asarray([
                ubs.flatten()[0],
                0.0,  # TOTO: Fill in for these values self.recent_click_rate,
                0.0,  # TOTO: Fill in for these values self.similarity_of_recent_items,
                0.0,  # TOTO: Fill in for these values self.diversity_score,
                0.0,  # TOTO: Fill in for these values self.session_length,
            ], dtype=np.float32),
        ])

    # def _get_live_env(self, user_id: int):
    #     if user_id not in self.live_env:
    #         env = self.create_env(user_id)
    #         self.live_env[user_id] = (env, env.reset()[0])
    #     return self.live_env[user_id]

    def reset_for_evaluation(self):
        pass
        # for env in self.live_env.values(): env.reset()
