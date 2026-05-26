"""
Common layer building blocks for DeMUL:
  - LinearLayer : Linear + LayerNorm + Dropout + optional activation
  - NetVLAD     : NetVLAD pooling (used in QueryWeightEncoder)
  - ConvSE      : 1-D Conv score predictor (start/end scoring head)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_logits(target, mask, eps=-1e4):
    """Set masked positions to a very negative value (pre-softmax masking)."""
    return target * mask + (1 - mask) * eps


# ---------------------------------------------------------------------------
# LinearLayer
# ---------------------------------------------------------------------------

class LinearLayer(nn.Module):
    """Linear projection with LayerNorm, Dropout, and optional activation."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1,
                 relu=True, tanh=False):
        super().__init__()
        self.relu = relu
        self.tanh = tanh
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz),
        )

    def forward(self, x):
        """x: (N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        if self.tanh:
            x = torch.tanh(x)
        return x


# ---------------------------------------------------------------------------
# NetVLAD
# ---------------------------------------------------------------------------

class NetVLAD(nn.Module):
    """NetVLAD pooling that aggregates a variable-length sequence into a
    fixed-size descriptor via soft cluster assignment.

    out_dim = cluster_size * feature_size
    """

    def __init__(self, cluster_size, feature_size, add_norm=True):
        super().__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters  = nn.Parameter(
            (1 / math.sqrt(feature_size)) * torch.randn(feature_size,
                                                         cluster_size))
        self.clusters2 = nn.Parameter(
            (1 / math.sqrt(feature_size)) * torch.randn(1, feature_size,
                                                         cluster_size))
        self.add_norm = add_norm
        self.LayerNorm = nn.LayerNorm(cluster_size)
        self.out_dim = cluster_size * feature_size

    def forward(self, x):
        # x: (B, L, D)
        max_sample = x.size(1)
        x = x.view(-1, self.feature_size)               # (B*L, D)
        assignment = torch.matmul(x, self.clusters)     # (B*L, K)
        if self.add_norm:
            assignment = self.LayerNorm(assignment)
        assignment = F.softmax(assignment, dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # (B, L, K)

        a_sum = assignment.sum(-2, keepdim=True)         # (B, 1, K)
        a = a_sum * self.clusters2                       # (B, D, K)

        assignment = assignment.transpose(1, 2)          # (B, K, L)
        x = x.view(-1, max_sample, self.feature_size)   # (B, L, D)
        vlad = torch.matmul(assignment, x)               # (B, K, D)
        vlad = vlad.transpose(1, 2)                      # (B, D, K)
        vlad = vlad - a

        vlad = F.normalize(vlad)                         # intra L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)                         # L2 norm
        return vlad                                      # (B, K*D)


# ---------------------------------------------------------------------------
# ConvSE  –  1-D convolutional score predictor
# ---------------------------------------------------------------------------

class ConvSE(nn.Module):
    """Two-layer 1-D Conv scorer used by the Moment Localization Head.

    Config must provide:
        config.conv_cfg_1 : dict  – kwargs for the first  nn.Conv1d
        config.conv_cfg_2 : dict  – kwargs for the second nn.Conv1d
    """

    def __init__(self, config):
        super().__init__()
        self.clip_score_predictor = nn.Sequential(
            nn.Conv1d(**config.conv_cfg_1),
            nn.ReLU(),
            nn.Conv1d(**config.conv_cfg_2),
        )

    def forward(self, contextual_qal_features, video_mask):
        """
        Args:
            contextual_qal_features : (B, C, T)
            video_mask              : (B, T)   int/bool, 1 = valid
        Returns:
            score : (B, T)  masked logits
        """
        score = self.clip_score_predictor(
            contextual_qal_features).squeeze(1)          # (B, T)
        score = mask_logits(score, video_mask)
        return score
