import torch
import torch.nn.functional as F
from torch import nn

from .other_models import AttentionPooling


class EntityEmbedding(nn.Module):
    def __init__(self):
        super(EntityEmbedding, self).__init__()


class SequenceEmbedding(nn.Module):
    def __init__(self):
        super(SequenceEmbedding, self).__init__()


class GenericRepresentation(EntityEmbedding):
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


class AdaptiveRepresentation(EntityEmbedding):
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
            # nn.LayerNorm(d_embed),
        )
        self.adaptor = nn.Sequential(
            nn.Linear(d_feat, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_embed),
            # nn.LayerNorm(d_embed),
        )
        self.alpha = nn.Sequential(
            nn.Linear(d_feat, 1),
            nn.Sigmoid(),
        )
        self.bias = nn.Embedding(n_entities, 1)

    def to(self, *args):
        self.lookup = self.lookup.to(*args)
        return super().to(*args)

    def forward(self, ids, feats, drop_id_prob=0.0, use_bias=True, return_id_embeds=False, return_alpha=False):
        feat_embeds = self.adaptor(feats)
        if feat_embeds.ndim == 1: feat_embeds = feat_embeds.unsqueeze(0)

        if not use_bias or ids is None:
            result = [feat_embeds, None]
            if return_id_embeds: result.append(None)
            if return_alpha: result.append(None)
            return tuple(result)

        ids = self.lookup[ids.cpu()]

        id_embeds = self.embedder(ids)
        id_bias = self.bias(ids).squeeze(-1) if use_bias else None

        if self.training and drop_id_prob > 0:
            mask = (torch.rand(ids.shape[0], device=ids.device) > drop_id_prob).float().unsqueeze(1)
            id_embeds = id_embeds * mask
            if use_bias: id_bias = id_bias * mask

        alpha = self.alpha(feats)
        fusion = alpha * id_embeds + (1 - alpha) * feat_embeds
        if use_bias: id_bias = alpha.squeeze(-1) * id_bias
        # fusion = alpha * id_embeds + feat_embeds
        # fusion = alpha * (id_embeds + id_bias) + (1 - alpha) * feat_embeds
        # fusion = alpha * (id_embeds + id_bias) + feat_embeds

        result = [fusion, id_bias]
        if return_id_embeds: result.append(alpha * id_embeds)
        if return_alpha: result.append(alpha)
        return tuple(result)

    @torch.no_grad()
    def embed(self, ids, feats):
        self.eval()
        if ids is not None and not hasattr(ids, '__len__'): ids = [ids]
        device = next(self.parameters()).device
        iem, ibs = self.forward(
            torch.tensor(ids, dtype=torch.int64, device=device),
            torch.tensor(feats, dtype=torch.float32, device=device))
        return iem.cpu().numpy(), ibs.cpu().numpy()

    @torch.no_grad()
    def update_embeds(self, uid, uft, new_uem, new_ubs):
        self.eval()
        if uid is not None and not hasattr(uid, '__len__'): uid = [uid]

        device = next(self.parameters()).device
        uid, uft = (torch.tensor(uid, dtype=torch.int64, device=device),
                    torch.tensor(uft, dtype=torch.float32, device=device))
        new_uem, new_ubs = (torch.tensor(new_uem, dtype=torch.float32, device=device),
                            torch.tensor(new_ubs, dtype=torch.float32, device=device))

        uem_f = self.adaptor(uft)
        if uem_f.ndim == 1: uem_f = uem_f.unsqueeze(0)

        uid = self.lookup[uid.cpu()]
        alpha = self.alpha(uft)

        self.embedder[0].weight[uid] = (new_uem - (1 - alpha) * uem_f) / alpha
        self.bias.weight[uid] = new_ubs.unsqueeze(1) / alpha


class LSTMRepresentation(nn.Module):
    def __init__(self, d_embed_in, d_embed_out):
        super(LSTMRepresentation, self).__init__()
        self.lstm = nn.LSTM(input_size=d_embed_in, hidden_size=d_embed_out, num_layers=1, batch_first=True)
        self.post = nn.Sequential(
            nn.Flatten(0, 1),
            nn.LayerNorm(d_embed_out),
        )

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        outputs, (hidden, cell) = self.lstm(packed)
        assert hidden.shape[0] == 1
        return self.post(hidden)


class GRURepresentation(nn.Module):
    def __init__(self, d_embed_in, d_embed_out):
        super(GRURepresentation, self).__init__()
        self.gru = nn.GRU(input_size=d_embed_in, hidden_size=d_embed_out, num_layers=1, batch_first=True)
        self.post = nn.Sequential(
            nn.Flatten(0, 1),
            nn.LayerNorm(d_embed_out),
        )

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        outputs, hidden = self.gru(packed)
        assert hidden.shape[0] == 1
        return self.post(hidden)


class AttentionRepresentation(nn.Module):
    def __init__(self, d_embed_in, d_embed_out):
        super(AttentionRepresentation, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=d_embed_in, nhead=1, dim_feedforward=d_embed_out, batch_first=True)
        self.proj = nn.Linear(d_embed_in, d_embed_out)
        self.pool = AttentionPooling(d_embed_out)
        self.norm = nn.LayerNorm(d_embed_out)

    def forward(self, x, mask):
        key_padding_mask = (mask == 0)  # [B, T]
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(self.pool(self.proj(x), mask))


class MeanRepresentation(nn.Module):
    def __init__(self, d_embed_in, d_embed_out):
        super(MeanRepresentation, self).__init__()
        self.norm = nn.LayerNorm(d_embed_in)
        self.d_embed_out = d_embed_out

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1)  # [Batch Size, Seq Lens, 1]
        x_masked = x * mask
        sum_x = x_masked.sum(dim=1)  # [B, D]
        lengths = mask.sum(dim=1)  # [B, 1]
        mean_x = sum_x / lengths.clamp(min=1e-6)
        return self.norm(mean_x)[:, :self.d_embed_out]


class SequenceRepresentation(SequenceEmbedding):
    def __init__(self, model: str | nn.Module, d_embed_in=64, d_embed_out=64):
        super(SequenceRepresentation, self).__init__()
        self.name = self.__class__.__name__
        if isinstance(model, str):
            self.name = self.name + "_" + model
            match model:
                case 'lstm':
                    self.model = LSTMRepresentation(d_embed_in, d_embed_out)
                case 'gru':
                    self.model = GRURepresentation(d_embed_in, d_embed_out)
                case 'attention':
                    self.model = AttentionRepresentation(d_embed_in, d_embed_out)
                case 'mean':
                    self.model = MeanRepresentation(d_embed_in, d_embed_out)
                case _:
                    raise Exception(f"Unknown model type: {model}")
        else:
            self.model = model

    def forward(self, x, mask):
        return self.model(x, mask)

    def _get_name(self):
        return self.name


class InteractionHistoryRepresentation(nn.Module):
    def __init__(self,
                 hem_embed: SequenceEmbedding,
                 hbs_embed: SequenceEmbedding,
                 ):
        super(InteractionHistoryRepresentation, self).__init__()
        self.hem_embed = hem_embed
        self.hbs_embed = hbs_embed
        self.hcl_reweight = nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.hcl_linear = nn.Linear(16, 1)

    def forward(self, hes, hbs, hcs, mask):
        # print('hes', hes.shape, 'hbs', hbs.shape, 'hcs', hcs.shape, 'mask', mask.shape)
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(hcs.unsqueeze(-1), lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.hcl_reweight(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=hes.shape[1])
        weights = F.sigmoid(self.hcl_linear(outputs))
        hem = self.hem_embed(hes * weights, mask)
        hbs = self.hbs_embed(hbs.unsqueeze(-1), mask)
        # hbs = self.hbs_embed(hbs.unsqueeze(-1) * weights, mask)
        return hem, hbs


class ConcatInteractionHistoryRepresentation(nn.Module):
    def __init__(self, d_embed: int):
        super(ConcatInteractionHistoryRepresentation, self).__init__()
        self.d_embed = d_embed
        self.contraction = nn.GRU(
            input_size=d_embed + 1 + 1,  # d_embed for embeddings, 1 for bias and 1 for click history
            hidden_size=d_embed, num_layers=1, batch_first=True)

    def forward(self, hes, hbs, hcs, mask, lengths):
        # print('hes', hes.shape, 'hbs', hbs.shape, 'hcs', hcs.shape, 'mask', mask.shape)
        # hes torch.Size([1, 3, 1024]) hbs torch.Size([1, 3]) hcs torch.Size([1, 3]) mask torch.Size([1, 3])
        if hbs is None: hbs = torch.zeros_like(hcs)
        agg_data = torch.cat([hes, hbs.unsqueeze(-1), hcs.unsqueeze(-1)], dim=2)
        packed = nn.utils.rnn.pack_padded_sequence(agg_data, lengths, enforce_sorted=False, batch_first=True)
        _, hidden = self.contraction(packed)
        return hidden.squeeze(0), None


class ConcatInteractionHistoryRepresentationWithUserEmbedInitialization(nn.Module):
    def __init__(self, d_embed: int, d_hidden: int):
        super(ConcatInteractionHistoryRepresentationWithUserEmbedInitialization, self).__init__()
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.hidden = nn.Linear(d_embed + 1, d_hidden)
        self.state = nn.Linear(d_embed + 1, d_hidden)
        self.contraction = nn.LSTM(
            input_size=d_embed + 1 + 1,  # d_embed for embeddings, 1 for bias and 1 for click history
            hidden_size=d_hidden, num_layers=1, batch_first=True)
        self.proj = nn.Linear(d_hidden, d_embed + 1)

    def forward(self, uem, ubs, hes, hbs, hcs, mask, lengths):
        # print('hes', hes.shape, 'hbs', hbs.shape, 'hcs', hcs.shape, 'mask', mask.shape)
        # hes torch.Size([1, 3, 1024]) hbs torch.Size([1, 3]) hcs torch.Size([1, 3]) mask torch.Size([1, 3])
        if ubs is None: ubs = torch.zeros_like(hcs)
        if hbs is None: hbs = torch.zeros_like(hcs)

        user_repr = torch.cat([uem, ubs.unsqueeze(-1)], dim=1)
        h0, c0 = self.hidden(user_repr), self.state(user_repr)
        h0, c0 = h0.unsqueeze(0).repeat(self.contraction.num_layers, 1, 1), c0.unsqueeze(0).repeat(self.contraction.num_layers, 1, 1)

        iih_repr = torch.cat([hes, hbs.unsqueeze(-1), hcs.unsqueeze(-1)], dim=2)
        packed = nn.utils.rnn.pack_padded_sequence(iih_repr, lengths.cpu(), enforce_sorted=False, batch_first=True)

        _, hidden = self.contraction(packed, (h0, c0))

        pref_repr = self.proj(hidden[0].squeeze(0))
        pem, pbs = pref_repr[:, :-1], pref_repr[:, -1]

        return pem, pbs
