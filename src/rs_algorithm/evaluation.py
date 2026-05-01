import os
from datetime import datetime, timedelta

import numpy as np
from collections.abc import Callable

import faiss
from sklearn.model_selection import train_test_split

import torch

from data_loaders import KuaiRecData
from dl_trainer import Trainer
from miscellaneous import get_relevant_items
from .system2.environments.env_05_almost_everything import create_norm_press_preference
from .models import AdaptiveRepresentation, TwoTowerRecommender
from .baselines import Solution
from .models.user_models.user_02_sim import SimUserModel, norm_dot_product_based_relavance_calculation


def aggregate(recs_items: list[list[int]], true_items: list[set[int]], function: Callable[[set[int], set[int]], float]):
    """
    Supposed to be mappings from user (index used as pseudo-index) to a list of items.
    Both arrays zipped correspond to users 1-1.
    """
    return np.mean([function(recs, true) for recs, true in zip(recs_items, true_items)])


def _validate_inputs(recs: list[list[int]], rel: list[set[int]]):
    assert len(recs) == len(rel), "recs and rel must have same length"
    for i, (r, t) in enumerate(zip(recs, rel)):
        assert len(r) > 0, f"recs must not be empty. Found empty at {i}"
        assert len(t) > 0, f"rel must not be empty. Found empty at {i}"


def _sample_top_k_as_set(recs: list[int], top_k: int) -> set[int]:
    if len(recs) <= top_k: return set(recs)
    result = set()
    for r in recs:
        result.add(r)
        if len(result) == top_k: break
    return result


def _sample_top_k_as_list(recs: list[int], top_k: int) -> list[int]:
    if len(recs) <= top_k: return list(dict.fromkeys(recs))
    seen = set()
    result = []
    for r in recs:
        if r in seen: continue
        result.append(r), seen.add(r)
        if len(result) == top_k: break
    return result


def hit_rate_at_k(recs: list[list[int]], rel: list[set[int]], top_k=10):
    _validate_inputs(recs, rel)
    top_recs = [set(r[:top_k]) for r in recs]
    hits = [len(tr & rl) > 0 for tr, rl in zip(top_recs, rel)]
    return np.mean(hits)


def precision_at_k(recs: list[list[int]], rel: list[set[int]], top_k=10):
    _validate_inputs(recs, rel)
    top_recs = [_sample_top_k_as_set(r, top_k) for r in recs]
    hits = [len(tr & rl) / len(tr) for tr, rl in zip(top_recs, rel)]
    return np.mean(hits)


def recall_at_k(recs: list[list[int]], rel: list[set[int]], top_k=10):
    _validate_inputs(recs, rel)
    top_recs = [_sample_top_k_as_set(r, top_k) for r in recs]
    hits = [len(tr & rl) / min(top_k, len(rl)) for tr, rl in zip(top_recs, rel)]
    return np.mean(hits)


def ndcg_at_k(recs: list[list[int]], rel: list[set[int]], top_k=10):
    _validate_inputs(recs, rel)

    top_recs = [_sample_top_k_as_list(r, top_k) for r in recs]

    def dcg(tr, rl):
        return sum(
            (1.0 / np.log2(i + 2)) if item in rl else 0.0
            for i, item in enumerate(tr)
        )

    def idcg(rl, k):
        ideal_hits = min(len(rl), k)
        return sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    scores = []
    for tr, rl in zip(top_recs, rel):
        dcg_val = dcg(tr, rl)
        idcg_val = idcg(rl, len(tr))
        scores.append(dcg_val / idcg_val if idcg_val > 0 else 0.0)

    return np.mean(scores)


def mrr_at_k(recs: list[list[int]], rel: list[set[int]], top_k=10):
    _validate_inputs(recs, rel)

    top_recs = [_sample_top_k_as_list(r, top_k) for r in recs]

    scores = []
    for tr, rl in zip(top_recs, rel):
        rr = 0.0
        for i, item in enumerate(tr):
            if item in rl:
                rr = 1.0 / (i + 1)
                break
        scores.append(rr)

    return np.mean(scores)


class Evaluator:
    def __init__(self, test_size=100, top_k=10):
        self.test_size = test_size
        self.top_k = top_k

    def evaluate(self, solution: Solution, top_k=None):
        top_k = top_k or self.top_k

        user_ids = solution.sample_random_users(self.test_size)
        train_true_items = solution.get_true_items(user_ids, solution.train_int)
        train_recs_items = solution.get_recommendations(user_ids, top_k, filter_already_liked_items=False)
        test_true_items = solution.get_true_items(user_ids, solution.test_int)
        test_recs_items = solution.get_recommendations(user_ids, top_k, filter_already_liked_items=True)

        return {
            "train": {
                "hit_rate_at_k": hit_rate_at_k(train_recs_items, train_true_items, top_k),
                "precision_at_k": precision_at_k(train_recs_items, train_true_items, top_k),
                "recall_at_k": recall_at_k(train_recs_items, train_true_items, top_k),
                "ndcg_at_k": ndcg_at_k(train_recs_items, train_true_items, top_k),
                "mrr_at_k": mrr_at_k(train_recs_items, train_true_items, top_k),
            },
            "test": {
                "hit_rate_at_k": hit_rate_at_k(test_recs_items, test_true_items, top_k),
                "precision_at_k": precision_at_k(test_recs_items, test_true_items, top_k),
                "recall_at_k": recall_at_k(test_recs_items, test_true_items, top_k),
                "ndcg_at_k": ndcg_at_k(test_recs_items, test_true_items, top_k),
                "mrr_at_k": mrr_at_k(test_recs_items, test_true_items, top_k),
            },
        }


class SimulationEvaluator:
    def __init__(self,
                 data_model: KuaiRecData, root=os.path.join('temp', 'checkpoints'),
                 n_sample_users=100, n_time_stamps=20,
                 top_k=10
                 ):
        #
        self.data_model = data_model
        self.root = root
        self.n_sample_users = n_sample_users
        self.n_time_stamps = n_time_stamps
        self.top_k = top_k

        self.d_embed = 128
        self.n_users, self.d_user_feats = data_model.ratings['user_id'].nunique(), data_model.user_features.shape[1]
        self.n_items, self.d_item_feats = data_model.ratings['item_id'].nunique(), data_model.item_features.shape[1]

        self.uid2idx = {v: i for i, v in enumerate(np.unique(self.data_model.ratings['user_id']))}
        self.user_model = AdaptiveRepresentation(self.n_users, self.d_user_feats, d_hidden=512, d_embed=self.d_embed, id2idx=self.uid2idx)
        self.iid2idx = {v: i for i, v in enumerate(np.unique(self.data_model.ratings['item_id']))}
        self.item_model = AdaptiveRepresentation(self.n_items, self.d_item_feats, d_hidden=512, d_embed=self.d_embed, id2idx=self.iid2idx)
        dot_embed_model = TwoTowerRecommender(self.user_model, self.item_model)
        Trainer(dot_embed_model, None, model_dir=self.root).load_model('model_id_trained_embedder_v3_128.pt')

        self.index = faiss.IndexFlatIP(self.d_embed)
        self.idx2iid = {}
        with torch.no_grad():
            for idx, iid in enumerate(self.data_model.ratings['item_id'].unique()):
                self.item_model.eval()
                iem, *_ = self.item_model(
                    torch.tensor([iid], dtype=torch.int64),
                    torch.tensor(self.data_model.get_item_features(iid)[None, :], dtype=torch.float32))
                self.index.add(iem.numpy())
                self.idx2iid[idx] = iid

        self.sample_user_ids = np.random.choice(data_model.ratings['user_id'].unique(), replace=False, size=n_sample_users)
        self.sample_users = self._build_users(self.sample_user_ids)

    def evaluate_simulation(self, solution: Solution, n_time_stamps=None, top_k=None):
        n_time_stamps = n_time_stamps or self.n_time_stamps
        top_k = top_k or self.top_k

        results = []

        solution.reset_for_evaluation()
        last_item_id = None
        did_user_click = None

        for uid in self.sample_user_ids:
            user = self.sample_users[uid]
            user.reset()  # Safety
            user_results = {'hit_rate_at_k': [], 'precision_at_k': [], 'recall_at_k': [], 'ndcg_at_k': [], 'mrr_at_k': []}

            for t in range(n_time_stamps):
                recs = solution.get_recommendations([uid], top_k, last_item_id=[last_item_id], did_user_click=[did_user_click])
                rel = [self.get_true_items(user)]

                user_results['hit_rate_at_k'].append(hit_rate_at_k(recs, rel, top_k))
                user_results['precision_at_k'].append(precision_at_k(recs, rel, top_k))
                user_results['recall_at_k'].append(recall_at_k(recs, rel, top_k))
                user_results['ndcg_at_k'].append(ndcg_at_k(recs, rel, top_k))
                user_results['mrr_at_k'].append(mrr_at_k(recs, rel, top_k))

                iid = recs[0][0]
                with torch.no_grad():
                    self.item_model.eval()
                    iem, ibs = self.item_model(
                        torch.tensor([iid], dtype=torch.int64),
                        torch.tensor(self.data_model.get_item_features(iid), dtype=torch.float32))
                    iem, ibs = iem.cpu().numpy(), ibs.cpu().numpy()
                last_item_id = iid
                did_user_click = user.react(datetime.now() + timedelta(days=t + 1), None, iem, ibs)

            results.append(user_results)

            user.reset()  # Manners

        return results

    @torch.no_grad()
    def _build_users(self, user_ids) -> dict[int, SimUserModel]:
        users: dict[int, SimUserModel] = {}

        norm_press_preference = create_norm_press_preference(self.data_model, self.user_model)

        for uid in user_ids:
            self.user_model.eval()
            uem, ubs = self.user_model(
                torch.tensor([uid], dtype=torch.int64),
                torch.tensor(self.data_model.get_user_features(uid), dtype=torch.float32), )
            users[uid] = SimUserModel(
                uem.cpu().numpy(), ubs.cpu().numpy(),
                relevance_calculator=norm_dot_product_based_relavance_calculation,
                preference_update_manager=norm_press_preference,
                good_item_discontent_weight=0.0,
                bad_item_discontent_weight=0.0,
            )

        return users

    def get_true_items(self, user: SimUserModel) -> set[int]:
        scores, indices = self.index.search(user.preference, 1000)
        return {self.idx2iid[i] for s, i in zip(scores.flatten(), indices.flatten()) if s > 1000000}  # TUNE THIS
        # _, indices = self.index.search(user.preference, 1000)
        # return set(np.vectorize(self.idx2iid.get)(indices[0, :int(len(indices[0]) * 0.1)]))
