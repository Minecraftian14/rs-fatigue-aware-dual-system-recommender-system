from abc import ABC
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pickle as pi
from tqdm import tqdm

from torch.utils.data import Dataset


# BASE IMPLEMENTATION

def _build_tree(sample):
    if isinstance(sample, tuple):
        return tuple(_build_tree(x) for x in sample)
    return []


def _append_tree(tree, sample):
    if isinstance(sample, tuple):
        for y, x in zip(tree, sample): _append_tree(y, x)
    else: tree.append(sample)


def _save_tree(tree, name, path):
    # print(type(tree), name)

    # if name == "VVICDataset.t0.t4":
    #     print()
    #     print(type(tree[0]))

    if isinstance(tree, tuple):
        for i, y in enumerate(tree): _save_tree(y, f"{name}.t{i}", path)
    else:
        array = np.asarray(tree)
        np.save(os.path.join(path, name), array)


def _load_tree(tree, name, path):
    if isinstance(tree, tuple):
        return tuple(_load_tree(y, f"{name}.t{i}", path) for i, y in enumerate(tree))
    else:
        # try:
        return np.load(os.path.join(path, f"{name}.npy"), mmap_mode='r')
        # except ValueError:
        #     return np.load(os.path.join(path, f"{name}.npy"), allow_pickle=True)


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
    for _, sample in zip(tqdm(range(len(dataset))), dataset):
        _append_tree(tree, sample)
    _save_tree(tree, name, path)


class CommitedDataset(Dataset):
    def __init__(self, name: str, path: str = os.path.join('temp', 'commited_datasets')):
        with open(os.path.join(path, f"{name}.tree"), "rb") as f: tree = pi.load(f)

        if isinstance(tree, tuple) and tree[0] == 'jagged':
            _, skeleton = tree
            self.tree = _load_jagged_tree(skeleton, name, path)
        else:
            self.tree = _load_tree(tree, name, path)

        self.size = _tree_size(self.tree)

    def __getitem__(self, idx):
        return _sample_tree(self.tree, idx)

    def __len__(self):
        return self.size

# JAGGED IMPLEMENTATION

class JaggedArray:
    """A variable-length sequence store backed by a flat memmap + an offsets array.

    Layout:
        offsets : int64 array of shape (N+1,) — offsets[i]:offsets[i+1] is sample i
        data    : flat memmap of shape (total_elements, *inner_shape)
    """
    def __init__(self, data: np.ndarray, offsets: np.ndarray):
        self.data = data
        self.offsets = offsets

    def __getitem__(self, idx):
        return self.data[self.offsets[idx]:self.offsets[idx + 1]]

    def __len__(self):
        return len(self.offsets) - 1

    @property
    def shape(self):
        return (len(self),)


# --- Pass 1 helpers: collect per-leaf lengths ---

def _jagged_build_lengths_tree(sample):
    if isinstance(sample, tuple):
        return tuple(_jagged_build_lengths_tree(x) for x in sample)
    return []


def _jagged_collect_lengths(lengths_tree, sample):
    if isinstance(sample, tuple):
        for lt, x in zip(lengths_tree, sample):
            _jagged_collect_lengths(lt, x)
    else:
        arr = np.asarray(sample)
        lengths_tree.append(arr.shape[0] if arr.ndim > 0 else 1)


def _jagged_build_info_tree(sample):
    """Capture (dtype, inner_shape) from the first sample at each leaf."""
    if isinstance(sample, tuple):
        return tuple(_jagged_build_info_tree(x) for x in sample)
    arr = np.asarray(sample)
    return (arr.dtype, arr.shape[1:])


def _jagged_is_variable(lengths: list) -> bool:
    return len(set(lengths)) > 1


# --- Allocation: pick the right storage per leaf ---

def _jagged_allocate_tree(lengths_tree, info_tree, size, name, path):
    if isinstance(lengths_tree, tuple):
        return tuple(
            _jagged_allocate_tree(lt, it, size, f"{name}.t{i}", path)
            for i, (lt, it) in enumerate(zip(lengths_tree, info_tree))
        )

    dtype, inner_shape = info_tree
    npy_path = os.path.join(path, f"{name}.npy")

    if _jagged_is_variable(lengths_tree):
        lengths = np.array(lengths_tree, dtype=np.int64)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        np.save(os.path.join(path, f"{name}.offsets.npy"), offsets)
        total = int(offsets[-1])
        shape = (total, *inner_shape) if inner_shape else (total,)
        data = np.lib.format.open_memmap(npy_path, mode='w+', dtype=dtype, shape=shape)
        return JaggedArray(data, offsets)
    else:
        L = lengths_tree[0]
        shape = (size, L, *inner_shape)  # ← always include L, no special case
        return np.lib.format.open_memmap(npy_path, mode='w+', dtype=dtype, shape=shape)

# --- Pass 2 helpers: stream samples to disk ---

def _jagged_write_tree(tree, sample, idx):
    if isinstance(sample, tuple):
        for subtree, x in zip(tree, sample):
            _jagged_write_tree(subtree, x, idx)
    elif isinstance(tree, JaggedArray):
        s, e = tree.offsets[idx], tree.offsets[idx + 1]
        tree.data[s:e] = np.asarray(sample)
    else:
        tree[idx] = np.asarray(sample)


# --- Load helpers ---

def _jagged_load_tree(skeleton, name, path):
    """Reconstruct tree from disk; detects jagged leaves by presence of .offsets.npy."""
    if isinstance(skeleton, tuple):
        return tuple(
            _jagged_load_tree(s, f"{name}.t{i}", path)
            for i, s in enumerate(skeleton)
        )
    data = np.load(os.path.join(path, f"{name}.npy"), mmap_mode='r')
    offsets_path = os.path.join(path, f"{name}.offsets.npy")
    if os.path.exists(offsets_path):
        offsets = np.load(offsets_path)
        return JaggedArray(data, offsets)
    return data


def _jagged_tree_size(tree):
    if isinstance(tree, tuple):
        for y in tree:
            return _jagged_tree_size(y)
        return None
    else:
        return len(tree)


def _jagged_sample_tree(tree, idx):
    if isinstance(tree, tuple):
        return tuple(_jagged_sample_tree(y, idx) for y in tree)
    else:
        return tree[idx]


# --- Public API ---

def commit_dataset_jagged(dataset: Dataset, name: str, path: str = os.path.join('temp', 'commited_datasets')):
    """Commit a mixed fixed/variable-length dataset to disk with O(1) RAM usage.

    Two passes are made over the dataset:
        Pass 1 — collect per-leaf lengths (cheap; no data stored in RAM).
        Pass 2 — stream samples directly into pre-allocated memmap files.

    Fixed-size leaves  → rectangular memmap  (N, L, ...)
    Variable-size leaves → JaggedArray        flat buffer + offsets
    """
    os.makedirs(path, exist_ok=True)
    first = dataset[0]
    size = len(dataset)

    # Pass 1: lengths only
    lengths_tree = _jagged_build_lengths_tree(first)
    print("Pass 1/2: collecting lengths...")
    for _, sample in zip(tqdm(range(len(dataset))), dataset):
        _jagged_collect_lengths(lengths_tree, sample)

    # Save skeleton so JaggedCommittedDataset can reconstruct the tree shape
    skeleton = _jagged_build_lengths_tree(first)   # structure only, no data
    with open(os.path.join(path, f"{name}.tree"), "wb") as f:
        pi.dump(('jagged', skeleton), f)

    # Allocate storage — each leaf independently becomes memmap or JaggedArray
    info_tree = _jagged_build_info_tree(first)
    tree = _jagged_allocate_tree(lengths_tree, info_tree, size, name, path)

    # Pass 2: write to disk
    print("Pass 2/2: writing data...")
    for idx, sample in zip(tqdm(range(len(dataset))), dataset):
        _jagged_write_tree(tree, sample, idx)


class JaggedCommittedDataset(Dataset):
    """Load a dataset committed with commit_dataset_jagged (mixed fixed/variable leaves)."""
    def __init__(self, name: str, path: str = os.path.join('temp', 'commited_datasets')):
        with open(os.path.join(path, f"{name}.tree"), "rb") as f:
            tag, skeleton = pi.load(f)
        assert tag == 'jagged', f"Expected 'jagged' dataset, got '{tag}'. Use CommittedDataset instead."
        self.tree = _jagged_load_tree(skeleton, name, path)
        self.size = _jagged_tree_size(self.tree)

    def __getitem__(self, idx):
        return _jagged_sample_tree(self.tree, idx)

    def __len__(self):
        return self.size

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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


class NegativeSamplingDataset(Dataset):
    def __init__(self, data_model: 'DataModel', num_negatives=4):
        self.data_model = data_model



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
