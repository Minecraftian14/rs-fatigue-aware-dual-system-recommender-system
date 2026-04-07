import os
from collections import defaultdict

import numpy as np
import pandas as pd


def smart_encode(df: pd.DataFrame, ignore_cols: set[str] = None, force_ohe: set[str] = None, index=None):
    if ignore_cols is None: ignore_cols = set()
    if force_ohe is None: force_ohe = set()

    df = df.copy()
    result = pd.DataFrame(index=df.index if index is None else index)

    for col in [col for col in df.columns if col not in ignore_cols]:
        if col not in force_ohe:
            # # If Boolean
            # if pd.api.types.is_bool_dtype(df[col]):
            #     result[col] = df[col].astype(int)
            #     continue

            # If Numerical
            if pd.api.types.is_numeric_dtype(df[col]):
                result[col] = df[col]
                continue

        unique_vals = df[col].dropna().unique()

        # If Binary
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: True, unique_vals[1]: False}
            result[f"{col}_is_{unique_vals[0]}"] = df[col].map(mapping)

        # If Categorical
        else:
            dummies = pd.get_dummies(df[col], prefix=col)
            result = pd.concat([result, dummies], axis=1)

    return result


def balanced_subset(df: pd.DataFrame, target_size: int, user_id='user_id', item_id='item_id', seed: int = 42):
    np.random.seed(seed)

    users = df[user_id].unique()
    items = df[item_id].unique()

    U = len(users)
    I = len(items)

    k_u = max(1, target_size // U)
    k_i = max(1, target_size // I)

    print(f"k_u (per user): {k_u}, k_i (per item): {k_i}")

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    user_deg = defaultdict(int)
    item_deg = defaultdict(int)

    selected_indices = []

    for idx, row in df_shuffled.iterrows():
        u = row[user_id]
        i = row[item_id]

        if user_deg[u] < k_u and item_deg[i] < k_i:
            selected_indices.append(idx)
            user_deg[u] += 1
            item_deg[i] += 1

        if len(selected_indices) >= target_size:
            break

    subset = df_shuffled.loc[selected_indices].copy()

    missing_users = set(users) - set(subset[user_id])
    if missing_users:
        print(f"Fixing {len(missing_users)} missing users")
        for u in missing_users:
            row = df[df[user_id] == u].sample(1, random_state=seed)
            subset = pd.concat([subset, row], ignore_index=True)

    missing_items = set(items) - set(subset[item_id])
    if missing_items:
        print(f"Fixing {len(missing_items)} missing items")
        for i in missing_items:
            row = df[df[item_id] == i].sample(1, random_state=seed)
            subset = pd.concat([subset, row], ignore_index=True)

    subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)

    return subset


def keyed_cache(name, fn):
    file = os.path.join('temp', 'keyed_cache', f"{name}.parquet")
    if os.path.exists(file): return pd.read_parquet(file)
    result = fn()
    result.to_parquet(file)
    return result
