import numpy as np
from .math_helpers import random_perpendicular


def get_relevant_items(index, idx2iid, vector, n=7):
    indices = index.search(vector, n)[1]
    return np.vectorize(idx2iid.get)(indices.flatten())


def get_semi_relevant_items(index, idx2iid, vector, n=3):
    indices = index.search(vector, 1500 + n)[1]
    return np.vectorize(idx2iid.get)(indices[:, 1500:].flatten())


def get_irrelevant_items(index, idx2iid, vector, n=7):
    vector = random_perpendicular(vector)
    return get_relevant_items(index, idx2iid, vector, n)


def get_random_items(n_items, idx2iid, n=2):
    indices = np.random.randint(0, n_items, size=n)
    return np.vectorize(idx2iid.get)(indices)
