import torch
from torch import nn
import torch.nn.functional as F


class RegressionBasedBPEModel(nn.Module):

    def __init__(self, user_model: nn.Module, item_model: nn.Module):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model

    def forward(self, user_ids, user_feats, pos_item_ids, pos_item_feats, neg_item_ids, neg_item_feats):
        user_emb, user_bias = self.user_model(user_ids, user_feats)
        pos_item_emb, pos_item_bias = self.item_model(pos_item_ids, pos_item_feats)
        neg_item_emb, neg_item_bias = self.item_model(neg_item_ids, neg_item_feats)

        pos_scores = (user_emb * pos_item_emb).sum(dim=1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)

        if user_bias is not None:
            pos_scores += user_bias
            neg_scores += user_bias
        if pos_item_bias is not None:
            pos_scores += pos_item_bias
            neg_scores += neg_item_bias

        return pos_scores, neg_scores
