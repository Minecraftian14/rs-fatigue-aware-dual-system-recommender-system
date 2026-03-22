import gymnasium as gym
import torch
from torch import nn

from .blueprint_model import ModelBlueprint


class _SuperEmbedding(nn.Module):
    def __init__(self,
                 use_ids=False, n_entities=None, d_id_embed=128,
                 use_feats=False, d_feat=None, d_feat_embed=128,
                 ):
        super(_SuperEmbedding, self).__init__()
        assert use_ids or use_feats

        if use_ids:
            self.id_embed = nn.Embedding(
                num_embeddings=n_entities,
                embedding_dim=d_id_embed,
            )
        else:
            self.id_embed = None

        if use_feats:
            self.feat_embed = nn.Sequential(
                nn.Linear(d_feat, d_feat_embed),
                nn.ReLU(),
                nn.Linear(d_feat_embed, d_feat_embed),
            )
        else:
            self.feat_embed = None

    def forward(self, *something):
        if len(something) == 1:
            if self.id_embed is not None:
                return self.id_embed(something[0]).squeeze(1)
            else:
                return self.feat_embed(something[0])
        ids, feats = something
        ids, feats = self.id_embed(ids), self.feat_embed(feats)
        return torch.cat([ids, feats], dim=-1)


class _NowThatsSomeModel(nn.Module):
    def __init__(self,
                 n_users, d_item_feat,
                 ):
        super(_NowThatsSomeModel, self).__init__()
        self.user_embed = _SuperEmbedding(use_ids=True, n_entities=n_users)
        self.item_embed = _SuperEmbedding(use_feats=True, d_feat=d_item_feat)

    def forward(self, user_ids, item_feats):
        # Expected inputs: user_ids [=] (batch_size, 1 (index)), item_ids [=] (batch_size, d_item_feat)
        user_embeds = self.user_embed(user_ids)
        item_embeds = self.item_embed(item_feats)
        # Expected embeds: user_embeds [=] (batch_size, 128), item_embeds [=] (batch_size, 128)
        return torch.sigmoid((user_embeds * item_embeds).sum(-1).unsqueeze(-1))
        # Expected outputs: return [=] (batch_size, 1)


class NowThatsSomeModel(ModelBlueprint):
    def __init__(self, n_users, d_item_feat):
        super().__init__()
        self.n_users = n_users
        self.d_item_feat = d_item_feat

    def _create_model(self, env: gym.Env = None):
        return _NowThatsSomeModel(self.n_users, self.d_item_feat)
