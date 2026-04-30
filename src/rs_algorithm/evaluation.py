import numpy as np
from collections.abc import Callable

from .baselines import Solution


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
    hits = [len(tr & rl) / len(rl) for tr, rl in zip(top_recs, rel)]
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
