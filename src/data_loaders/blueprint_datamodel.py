from abc import ABC

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    def __init__(self, data_model: 'DataModel', ratings=None):
        self.data_model = data_model
        self.ratings: pd.DataFrame = ratings if ratings is not None else data_model.ratings
        self.ratings = self.ratings.sample(frac=1)
        self.median = self.ratings['rating'].median()
        self.item_lookup = self.ratings.groupby('user_id')['item_id'].apply(list).to_dict()
        self.rating_lookup = self.ratings.groupby('user_id')['rating'].apply(list).to_dict()
        self.item_lookup = {k: np.array(v, dtype=np.int64) for k, v in self.item_lookup.items()}
        self.rating_lookup = {k: np.array(v, dtype=np.float32) for k, v in self.rating_lookup.items()}

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx: int):
        user_id = self.ratings.iloc[idx]['user_id'].astype(np.int64)
        a_item_id = self.ratings.iloc[idx]['item_id'].astype(np.int64)
        a_rating = self.ratings.iloc[idx]['rating'].astype(np.float32)

        user_ratings = self.rating_lookup[user_id]
        user_items = self.item_lookup[user_id]

        if a_rating > self.median:
            mask = user_ratings <= self.median
        else:
            mask = user_ratings > self.median

        if mask.sum() > 0:
            user_ratings = user_ratings[mask]
            choice = np.random.randint(len(user_ratings))
            b_item_id = user_items[mask][choice]
            b_rating = user_ratings[choice]
        else:
            while True:
                choice = np.random.randint(self.data_model.n_items)
                if choice not in user_items: break
                # Might select an item which the user has not seen but liked anyways
            b_item_id = choice
            b_rating = np.array(0.0) if a_rating > self.median else np.array(2 * self.median)

        pos_item = a_item_id if a_rating > b_rating else b_item_id
        neg_item = a_item_id if a_rating <= b_rating else b_item_id
        pos_rating = max(a_rating, b_rating)
        neg_rating = min(a_rating, b_rating)

        return [[user_id, self.data_model.get_user_features(user_id),
                 pos_item, self.data_model.get_item_features(pos_item),
                 neg_item, self.data_model.get_item_features(neg_item)],
                [pos_rating, neg_rating]]


class DataModel(ABC):

    def __init__(self):
        self.user_features: pd.DataFrame = None
        self.item_features: pd.DataFrame = None
        self.ratings: pd.DataFrame = None
        self.n_users: int = None
        self.d_user_feats: int = None
        self.n_items: int = None
        self.d_item_feats: int = None

    def get_n_items(self) -> int:
        pass

    def get_d_item_embeds(self) -> int:
        pass

    def get_items_embeds(self, item_idx: int | slice | None = None) -> np.ndarray:
        pass

    def get_bpr_dataset(self, ratings=None):
        if ratings is None: ratings = self.ratings
        return BPRDataset(self, ratings)

    def get_user_features(self, user_idx: int | slice | None = None, use_pd=False):
        if user_idx is None: result = self.user_features
        else: result = self.user_features.iloc[user_idx]
        if use_pd: return result
        return result.astype(np.float32).to_numpy()

    def get_item_features(self, item_idx: int | slice | None = None, use_pd=False):
        if item_idx is None: result = self.item_features
        else: result = self.item_features.iloc[item_idx]
        if use_pd: return result
        return result.astype(np.float32).to_numpy()
