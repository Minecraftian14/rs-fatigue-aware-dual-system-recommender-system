import numpy as np
import os
from math import isclose

from sklearn.metrics.pairwise import cosine_similarity

from data_loaders import DataModel, MovieLensData100k, SyntheticData, KuaiRecData, commit_dataset, CommitedDataset


def test_stuff():
    print("Ahoy")
    data_model = MovieLensData100k()
    print("Ahoy")


def test_dm_synthetic_data():
    data: DataModel = SyntheticData()

    n_items = data.get_n_items()
    item_0 = data.get_items_embeds(0)
    items = data.get_items_embeds(slice(0, n_items))

    dps = cosine_similarity(item_0 * items)

    print(dps.min(), dps.max())
    assert dps.min() < -0.8  # And some dissimilar ones
    assert dps.max() > +0.8  # There are some similar items


def test_commited_datasets():
    # data_model = KuaiRecData(initial_specimen='smaller')
    # dataset = data_model.get_bpr_dataset()
    # commit_dataset(dataset, 'KuaiRecData_smaller')
    dataset = CommitedDataset('KuaiRecData_smaller')
    assert len(dataset) == 998100
