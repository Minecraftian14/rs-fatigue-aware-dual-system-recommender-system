import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, d_feat):
        super(AttentionPooling, self).__init__()
        self.score = nn.Linear(d_feat, 1)

    def forward(self, x, mask):
        scores = self.score(x).squeeze(-1)
        all_masked = (mask == 0).all(dim=1, keepdim=True)  # [B, 1]
        safe_mask = mask.clone()
        safe_mask[all_masked.squeeze(1)] = 1
        scores = scores.masked_fill(safe_mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)
