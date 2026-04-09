import torch
from torch import nn
import torch.nn.functional as F

from .entity_representation import EntityEmbedding, SequenceEmbedding


class RegressionBasedBPEModel(nn.Module):

    def __init__(self, user_model: EntityEmbedding, item_model: EntityEmbedding):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model

    def forward(self, user_ids, user_feats, pos_item_ids, pos_item_feats, neg_item_ids, neg_item_feats):
        user_emb, user_bias, user_id_embeds = self.user_model(user_ids, user_feats, return_alpha=True)
        pos_item_emb, pos_item_bias, pos_item_id_embeds = self.item_model(pos_item_ids, pos_item_feats, return_alpha=True)
        neg_item_emb, neg_item_bias, neg_item_id_embeds = self.item_model(neg_item_ids, neg_item_feats, return_alpha=True)

        pos_scores = (user_emb * pos_item_emb).sum(dim=1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)

        if user_bias is not None:
            pos_scores += user_bias
            neg_scores += user_bias
        if pos_item_bias is not None:
            pos_scores += pos_item_bias
            neg_scores += neg_item_bias

        return pos_scores, neg_scores, (user_id_embeds, pos_item_id_embeds, neg_item_id_embeds)


class ThreeTowerRecommender(nn.Module):
    """
    A recommender based on user embeddings and item history
    """

    def __init__(self, d_embed,
                 user_model: EntityEmbedding,
                 item_model: EntityEmbedding,
                 history_model: SequenceEmbedding,
                 ):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.history_model = history_model
        self.alpha = nn.Sequential(
            nn.Linear(d_embed, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_ids, user_feats, item_hist_ids, item_hist_feats, mask):
        user_emb, user_bias = self.user_model(user_ids, user_feats)

        if len(item_hist_feats.shape) == 3:
            item_hist_ids, item_hist_feats = item_hist_ids.flatten(), item_hist_feats.flatten(0, 1)
        hist_embs, hist_biases = self.item_model(item_hist_ids, item_hist_feats)

        hist_embs = hist_embs.reshape((mask.shape[0], -1, hist_embs.shape[-1]))
        hist_bias = hist_biases.reshape((mask.shape[0], -1, 1))

        hist_emb = self.history_model(hist_embs, mask)

        alpha = self.alpha(user_emb)

        return alpha * user_emb + (1 - alpha) * hist_emb, (user_bias, hist_bias)
