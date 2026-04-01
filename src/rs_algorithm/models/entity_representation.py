import torch
from torch import nn
import torch.nn.functional as F


class GenericRepresentation(nn.Module):
    def __init__(self,
                 use_ids=False, n_entities=None, d_id_embed=128,
                 use_feats=False, d_feat=None, d_feat_embed=128,
                 aggregation=lambda x, y: torch.cat([x, y], dim=-1),
                 ):
        super(GenericRepresentation, self).__init__()
        assert use_ids or use_feats
        self.aggregation = aggregation

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
        return self.aggregation(ids, feats)


class AdaptiveRepresentation(nn.Module):
    def __init__(self, n_entities, d_feat, d_hidden=128, d_embed=64, dropout=0.2, id2idx=None):
        super(AdaptiveRepresentation, self).__init__()

        if id2idx is None: id2idx = {i: i for i in range(n_entities)}
        self.id2idx = id2idx
        self.idx2id = {v: k for k, v in id2idx.items()}
        assert len(self.id2idx) == len(self.idx2id)
        self.lookup = torch.zeros(max(id2idx.keys()) + 1, dtype=torch.int64)
        for k, v in id2idx.items(): self.lookup[k] = v

        self.embedder = nn.Sequential(
            nn.Embedding(num_embeddings=n_entities, embedding_dim=d_embed),
            nn.Flatten(),
            nn.LayerNorm(d_embed),
        )
        self.adaptor = nn.Sequential(
            nn.Linear(d_feat, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_embed),
            nn.LayerNorm(d_embed),
        )
        self.alpha = nn.Sequential(
            nn.Linear(d_feat, 1),
            nn.Sigmoid(),
        )
        self.bias = nn.Embedding(n_entities, 1)

    def forward(self, ids, feats, drop_id_prob=0.0):
        feat_embeds = self.adaptor(feats)

        if ids is None: return feat_embeds, None
        ids = self.lookup[ids]

        id_embeds = self.embedder(ids)
        id_bias = self.bias(ids).squeeze(-1)

        if self.training and drop_id_prob > 0:
            mask = (torch.rand(ids.shape[0], device=ids.device) > drop_id_prob).float().unsqueeze(1)
            id_embeds = id_embeds * mask
            id_bias = id_bias * mask

        alpha = self.alpha(feats)
        fusion = alpha * id_embeds + (1 - alpha) * feat_embeds
        # fusion = alpha * id_embeds + feat_embeds

        return fusion, id_bias
