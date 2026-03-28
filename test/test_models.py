import torch
from torch import nn
import torch.nn.functional as F

from rs_algorithm.models import GenericRepresentation


def test_generic_representation():
    user_ids = torch.randint(0, 100, (100,), dtype=torch.long)
    user_feats = torch.randn(100, 10)

    id_only = GenericRepresentation(use_ids=True, n_entities=100)
    user_repr = id_only(user_ids)
    assert user_repr.shape == (100, 128)

    feat_only = GenericRepresentation(use_feats=True, d_feat=10)
    user_repr = feat_only(user_feats)
    assert user_repr.shape == (100, 128)

    both = GenericRepresentation(use_ids=True, use_feats=True, n_entities=100, d_feat=10)
    user_repr = both((user_ids, user_feats))
    assert user_repr.shape == (100, 256)
