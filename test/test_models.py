import torch
from torch import nn
import torch.nn.functional as F

from rs_algorithm.models import GenericRepresentation, AdaptiveRepresentation, SequenceRepresentation, AttentionPooling


def test_generic_representation():
    user_ids = torch.randint(0, 100, (100,), dtype=torch.long)
    user_feats = torch.randn(100, 10)

    id_only = GenericRepresentation(use_ids=True, n_entities=100)
    with torch.no_grad():
        user_repr = id_only(user_ids)
        assert user_repr.shape == (100, 128)

    feat_only = GenericRepresentation(use_feats=True, d_feat=10)
    with torch.no_grad():
        user_repr = feat_only(user_feats)
        assert user_repr.shape == (100, 128)

    both = GenericRepresentation(use_ids=True, use_feats=True, n_entities=100, d_feat=10)
    with torch.no_grad():
        user_repr = both((user_ids, user_feats))
        assert user_repr.shape == (100, 256)


def test_adaptive_representation():
    user_ids = torch.randint(0, 100, (100,), dtype=torch.long)
    user_feats = torch.randn(100, 10)

    representation = AdaptiveRepresentation(n_entities=100, d_feat=10)

    with torch.no_grad():
        user_repr, user_bias = representation(user_ids, user_feats)
        assert user_repr.shape == (100, 64)
        assert user_bias.shape == (100,)

        user_repr, user_bias = representation(None, user_feats)
        assert user_repr.shape == (100, 64)
        assert user_bias is None

        representation.training = True
        user_repr, user_bias = representation.forward(user_ids, user_feats, drop_id_prob=0.5)
        assert user_repr.shape == (100, 64)
        assert user_bias.shape == (100,)


def test_attention_pooling():
    sequence = torch.randn(100, 18, 64)  # (batch, seq_len, d_feat)
    mask = torch.ones(100, 18)  # (batch, seq_len, d_feat)
    pooling = AttentionPooling(64)
    with torch.no_grad():
        output = pooling(sequence, mask)
        assert output.shape == (100, 64)


def test_sequence_representation():
    sequence = torch.randn(100, 18, 64)  # (batch, seq_len, d_feat)
    mask = torch.ones(100, 18)

    for model in ('lstm', 'gru', 'attention', 'mean'):
        representation = SequenceRepresentation(model)
        with torch.no_grad():
            output = representation(sequence, mask)
            assert output.shape == (100, 64)
