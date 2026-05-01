from abc import ABC, abstractmethod

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.recommender_base import RecommenderBase
from sklearn.model_selection import train_test_split

from data_loaders import KuaiRecData
from miscellaneous import build_sparse_matrix


class Solution(ABC):

    def __init__(self, data_model: KuaiRecData):
        self.data_model = data_model
        self.interactions = self.data_model.ratings[['user_id', 'item_id', 'rating']]
        self.median = self.interactions['rating'].median()

        self.train_int, self.test_int = train_test_split(
            self.interactions,
            test_size=0.2, stratify=self.interactions['user_id'])

        self.n_users = self.interactions['user_id'].max() + 1
        self.n_items = self.interactions['item_id'].max() + 1
        self.known_users = self.interactions['user_id'].unique()
        self.known_items = self.interactions['item_id'].unique()

    def prepare(self): pass

    def sample_random_users(self, test_size: int) -> list[int]:
        return np.random.choice(self.known_users, test_size).tolist()

    def get_true_items(self, user_ids: list[int], data=None) -> list[set[int]]:
        ratings = self.train_int if data is None else data
        ratings = ratings[(ratings['user_id'].isin(user_ids))
                          & (ratings['rating'] >= self.median)]
        ratings = (ratings.groupby("user_id")["item_id"]
                   .apply(set).to_dict())
        # .apply(dict.fromkeys).to_dict())
        return [ratings.get(u, {}) for u in user_ids]

    @abstractmethod
    def get_recommendations(self, user_ids: list[int], top_k=10, filter_already_liked_items=False, last_item_id=None, did_user_click=None) -> list[list[int]]: pass

    def reset_for_evaluation(self): pass


class IdealSolutionBaseline(Solution):
    def __init__(self, data_model: KuaiRecData):
        super().__init__(data_model)

    def get_recommendations(self, user_ids: list[int], top_k=10, filter_already_liked_items=False, last_item_id=None, did_user_click=None) -> list[list[int]]:
        data = self.test_int if filter_already_liked_items else self.train_int
        return [list(x)[:top_k] for x in self.get_true_items(user_ids, data=data)]


class ImplicitBaseline(Solution):
    def __init__(self, data_model: KuaiRecData, model: RecommenderBase):
        super().__init__(data_model)

        user_ids = self.train_int['user_id'].to_numpy(dtype=np.int64)
        item_ids = self.train_int['item_id'].to_numpy(dtype=np.int64)
        ratings = self.train_int['rating'].to_numpy(dtype=np.float32)

        self.train_data = build_sparse_matrix(user_ids, item_ids, ratings, self.n_users, self.n_items)
        self.model = model
        self.model.fit(self.train_data)

    def _get_recommendations(self, user_id: int, top_k, filter_already_liked_items) -> list[int]:
        user_items = self.train_data[user_id]

        item_ids, scores = self.model.recommend(
            userid=int(user_id),
            user_items=user_items,
            N=top_k,
            filter_already_liked_items=filter_already_liked_items
        )

        return item_ids.tolist()

    def get_recommendations(self, user_ids: list[int], top_k=10, filter_already_liked_items=False, last_item_id=None, did_user_click=None) -> list[list[int]]:
        return [self._get_recommendations(u, top_k, filter_already_liked_items) for u in user_ids]


class AlternatingLeastSquaresBaseline(Solution):
    def __init__(self, data_model: KuaiRecData, factors=64, regularization=0.01, iterations=15):
        super().__init__(data_model)

        user_ids = self.train_int['user_id'].to_numpy(dtype=np.int64)
        item_ids = self.train_int['item_id'].to_numpy(dtype=np.int64)
        ratings = self.train_int['rating'].to_numpy(dtype=np.float32)

        self.train_data = build_sparse_matrix(user_ids, item_ids, ratings, self.n_users, self.n_items)

        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )
        self.model.fit(self.train_data)

    def _get_recommendations(self, user_id: int, top_k, filter_already_liked_items) -> list[int]:
        user_items = self.train_data[user_id]

        item_ids, scores = self.model.recommend(
            userid=int(user_id),
            user_items=user_items,
            N=top_k,
            filter_already_liked_items=filter_already_liked_items
        )

        return item_ids.tolist()

    def get_recommendations(self, user_ids: list[int], top_k=10, filter_already_liked_items=False, last_item_id=None, did_user_click=None) -> list[list[int]]:
        return [self._get_recommendations(u, top_k, filter_already_liked_items) for u in user_ids]


class BayesianPersonalizedRankingBaseline(ImplicitBaseline):
    def __init__(self, data_model: KuaiRecData, factors=64, learning_rate=0.01, regularization=0.01, iterations=50):
        super().__init__(data_model, BayesianPersonalizedRanking(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations
        ))
