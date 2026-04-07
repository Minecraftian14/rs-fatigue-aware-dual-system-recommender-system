from abc import ABC
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pickle as pi
from tqdm import tqdm

from torch.utils.data import Dataset


def _build_tree(sample):
    if isinstance(sample, tuple):
        return tuple(_build_tree(x) for x in sample)
    return []


def _append_tree(tree, sample):
    if isinstance(sample, tuple):
        for y, x in zip(tree, sample): _append_tree(y, x)
    else: tree.append(sample)


def _save_tree(tree, name, path):
    if isinstance(tree, tuple):
        for i, y in enumerate(tree): _save_tree(y, f"{name}.t{i}", path)
    else:
        array = np.asarray(tree)
        np.save(os.path.join(path, name), array)


def _load_tree(tree, name, path):
    if isinstance(tree, tuple):
        return tuple(_load_tree(y, f"{name}.t{i}", path) for i, y in enumerate(tree))
    else: return np.load(os.path.join(path, f"{name}.npy"), mmap_mode='r')


def _tree_size(tree):
    if isinstance(tree, tuple):
        for y in tree: return _tree_size(y)
        return None
    elif isinstance(tree, list): return len(tree)
    else: return tree.shape[0]


def _sample_tree(tree, idx):
    if isinstance(tree, tuple):
        return tuple(_sample_tree(y, idx) for y in tree)
    else: return tree[idx]


def commit_dataset(dataset: Dataset, name: str, path: str = os.path.join('temp', 'commited_datasets')):
    sample = dataset[0]
    tree = _build_tree(sample)
    with open(os.path.join(path, f"{name}.tree"), "wb") as f: pi.dump(tree, f)
    for sample in tqdm(dataset):
        _append_tree(tree, sample)
    _save_tree(tree, name, path)


class CommitedDataset(Dataset):
    def __init__(self, name: str, path: str = os.path.join('temp', 'commited_datasets')):
        with open(os.path.join(path, f"{name}.tree"), "rb") as f: tree = pi.load(f)
        self.tree = _load_tree(tree, name, path)
        self.size = _tree_size(self.tree)

    def __getitem__(self, idx):
        return _sample_tree(self.tree, idx)

    def __len__(self):
        return self.size


class SimpleDataset(Dataset):
    def __init__(self, data_model: 'DataModel', ratings=None):
        self.data_model = data_model
        self.ratings: pd.DataFrame = ratings if ratings is not None else data_model.ratings
        self.ratings = self.ratings.sample(frac=1)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx: int):
        user_id = self.ratings.iloc[idx]['user_id'].astype(np.int64)
        return user_id, self.data_model.get_user_features(user_id)


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

        return ((user_id, self.data_model.get_user_features(user_id),
                 pos_item, self.data_model.get_item_features(pos_item),
                 neg_item, self.data_model.get_item_features(neg_item)),
                (pos_rating, neg_rating))


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

    def get_dataset(self, ratings=None):
        if ratings is None: ratings = self.ratings
        return SimpleDataset(self, ratings)

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
