import numpy as np
import scipy.sparse as sp


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lerp(a, b, f):
    return a + (b - a) * f


def clip(a, m=0, M=1):
    return min(max(a, m), M)


def angle(a: np.ndarray, b: np.ndarray):
    a, b = a.flatten(), b.flatten()
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def random_axis(dims, count):
    if count > dims: raise ValueError("Cannot have more than n orthogonal vectors in n-dimensional space.")

    vectors = []

    for _ in range(count):
        v = np.random.randn(dims)

        # Subtract projections onto all previous vectors
        for u in vectors:  v -= np.dot(v, u) * u

        # Retry if vector is too small (numerical issue)
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return random_axis(dims, count)

        v = v / norm
        vectors.append(v)

    return vectors


def build_sparse_matrix(position_x, position_y, values, size_x, size_y):
    return sp.csr_matrix(
        (values, (position_x, position_y)),
        shape=(size_x, size_y),
        dtype=np.float32
    )
