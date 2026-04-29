from typing import overload, Self

import torch
from torch import nn, dtype, Tensor
import torch.nn.functional as F
from torch._prims_common import DeviceLikeType

from . import InteractionHistoryRepresentation
from .entity_representation import EntityEmbedding, SequenceEmbedding


class TwoTowerRecommender(nn.Module):

    def __init__(self, user_model: EntityEmbedding, item_model: EntityEmbedding):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.use_bias = True

    def to(self, *args, **kwargs):
        self.user_model = self.user_model.to(*args, **kwargs)
        self.item_model = self.item_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def prepare_for_feature_based_embeds_training(self):
        self.use_bias = False
        # TODO: Assuming that the models are AdaptiveRepresentation
        # self.user_model.adaptor[2].p = 0.2
        # self.item_model.adaptor[2].p = 0.2
        for param in self.user_model.adaptor.parameters(): param.requires_grad = True
        for param in self.item_model.adaptor.parameters(): param.requires_grad = True

    def prepare_for_id_based_embeds_training(self):
        self.use_bias = True
        # TODO: Assuming that the models are AdaptiveRepresentation
        # self.user_model.adaptor[2].p = 0.0
        # self.item_model.adaptor[2].p = 0.0
        for param in self.user_model.adaptor.parameters(): param.requires_grad = False
        for param in self.item_model.adaptor.parameters(): param.requires_grad = False

    def forward(self, user_ids, user_feats, pos_item_ids, pos_item_feats, neg_item_ids=None, neg_item_feats=None):
        user_emb, user_bias, *user_extras = self.user_model(user_ids, user_feats, use_bias=self.use_bias, return_alpha=True)

        pos_item_emb, pos_item_bias, *pos_item_extras = self.item_model(pos_item_ids, pos_item_feats, use_bias=self.use_bias, return_alpha=True)

        user_emb = F.normalize(user_emb, dim=1)
        pos_item_emb = F.normalize(pos_item_emb, dim=1)
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)

        if self.use_bias and user_bias is not None and pos_item_bias is not None:
            pos_scores = pos_scores + user_bias + pos_item_bias

        if neg_item_ids is None:
            return pos_scores, tuple(zip(user_extras, pos_item_extras))

        neg_item_emb, neg_item_bias, *neg_item_extras = self.item_model(neg_item_ids, neg_item_feats, use_bias=self.use_bias, return_alpha=True)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)
        if self.use_bias: neg_scores = neg_scores + user_bias + neg_item_bias

        return pos_scores, neg_scores, tuple(zip(user_extras, pos_item_extras, neg_item_extras))


class ThreeTowerRecommender(nn.Module):
    """
    A recommender based on user embeddings and item history
    """

    def __init__(self, d_embed,
                 user_model: EntityEmbedding,
                 item_model: EntityEmbedding,
                 history_model: InteractionHistoryRepresentation,
                 ):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.history_model = history_model
        self.alpha = nn.Sequential(
            nn.Linear(d_embed, 1),
            nn.Sigmoid(),
        )

    def to(self, *args, **kwargs):
        self.user_model = self.user_model.to(*args, **kwargs)
        self.item_model = self.item_model.to(*args, **kwargs)
        self.history_model = self.history_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, user_ids, user_feats, item_hist_ids, item_hist_feats, item_hist_clks, mask, lens):
        # print("user_ids, user_feats, item_hist_ids, item_hist_feats, item_hist_clks, mask")
        # print(user_ids.shape, user_feats.shape, item_hist_ids.shape, item_hist_feats.shape, item_hist_clks.shape, mask.shape)
        if user_ids.ndim == 2: user_ids = user_ids.squeeze(-1)
        user_emb, user_bias = self.user_model(user_ids, user_feats)
        # print("user_emb, user_bias", user_emb.shape, user_bias.shape)

        if len(item_hist_feats.shape) == 3:
            item_hist_ids, item_hist_feats = item_hist_ids.flatten(), item_hist_feats.flatten(0, 1)
        hist_embs, hist_biases = self.item_model(item_hist_ids, item_hist_feats)

        hist_embs = hist_embs.reshape((mask.shape[0], -1, hist_embs.shape[-1]))
        # hist_embs = torch.cat((hist_embs, item_hist_clks.unsqueeze(-1)), dim=-1)
        # hist_bias = hist_biases.reshape((mask.shape[0], -1, 1))
        hist_bias = hist_biases.reshape((mask.shape[0], -1))

        hist_embs, hist_bias = self.history_model(hist_embs, hist_bias, item_hist_clks, mask, lens)

        alpha = self.alpha(user_emb)

        return alpha * user_emb + (1 - alpha) * hist_embs, (user_bias, hist_bias)


class ThreeTowerRecommenderV2(nn.Module):
    """
    A recommender based on user embeddings and item history
    """

    def __init__(self, d_embed,
                 user_model: EntityEmbedding,
                 item_model: EntityEmbedding,
                 history_model: InteractionHistoryRepresentation,
                 ):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.history_model = history_model
        self.alpha = nn.Sequential(
            nn.Linear(d_embed, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_ids, user_feats, item_hist_ids, item_hist_feats, item_hist_clks, mask, lens):
        # print("user_ids, user_feats, item_hist_ids, item_hist_feats, item_hist_clks, mask")
        # print(user_ids.shape, user_feats.shape, item_hist_ids.shape, item_hist_feats.shape, item_hist_clks.shape, mask.shape)
        if user_ids.ndim == 2: user_ids = user_ids.squeeze(-1)
        user_emb, user_bias = self.user_model(user_ids, user_feats)
        # print("user_emb, user_bias", user_emb.shape, user_bias.shape)

        if item_hist_ids.shape[1] == 0:
            return user_emb, (user_bias, user_bias)

        if len(item_hist_feats.shape) == 3:
            item_hist_ids, item_hist_feats = item_hist_ids.flatten(), item_hist_feats.flatten(0, 1)
        hist_embs, hist_biases = self.item_model(item_hist_ids, item_hist_feats)

        hist_embs = hist_embs.reshape((mask.shape[0], -1, hist_embs.shape[-1]))
        # hist_embs = torch.cat((hist_embs, item_hist_clks.unsqueeze(-1)), dim=-1)
        # hist_bias = hist_biases.reshape((mask.shape[0], -1, 1))
        hist_bias = hist_biases.reshape((mask.shape[0], -1))

        hist_embs, hist_bias = self.history_model(user_emb, user_bias, hist_embs, hist_bias, item_hist_clks, mask, lens)

        # alpha = self.alpha(user_emb)
        # return alpha * user_emb + (1 - alpha) * hist_embs, (user_bias, hist_bias)

        return hist_embs, (user_bias, hist_bias)
