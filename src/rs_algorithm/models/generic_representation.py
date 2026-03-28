import torch
from torch import nn
import torch.nn.functional as F


class GenericRepresentation(nn.Module):
    def __init__(self,
                 use_ids=False, n_entities=None, d_id_embed=128,
                 use_feats=False, d_feat=None, d_feat_embed=128,
                 ):
        super(GenericRepresentation, self).__init__()
        assert use_ids or use_feats

        if use_ids:
            self.id_embed = nn.Sequential(
                nn.Embedding(
                    num_embeddings=n_entities,
                    embedding_dim=d_id_embed,
                ),
                nn.Flatten(),
                nn.Linear(d_id_embed, d_id_embed),
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

        if use_ids and use_feats:
            self.__forward__ = self.forward_both
        elif use_ids:
            self.__forward__ = self.forward_id_only
        elif use_feats:
            self.__forward__ = self.forward_feat_only
        else:
            raise Exception("No inputs to the model")

    def forward(self, something):
        return self.__forward__(something)

    def forward_id_only(self, ids):
        return self.id_embed(ids)

    def forward_feat_only(self, feats):
        return self.feat_embed(feats)

    def forward_both(self, batch):
        ids, feats = batch
        ids, feats = self.id_embed(ids), self.feat_embed(feats)
        return torch.cat([ids, feats], dim=-1)
