"""
Moment Localization Head for DeMUL.

Uses two 2-layer Bidirectional GRUs (for start and end positions respectively)
followed by a two-layer 1-D Conv scorer (ConvSE) for each boundary.

Input  : (B, T, d)  – contextualized video clip features
Output : start_scores (B, T), end_scores (B, T)  – unnormalized logits
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.layers import ConvSE


class MomentLocalizationHead(nn.Module):
    """Temporal boundary predictor.

    Args:
        config              : EasyDict with moment_localization_config
                              (conv_cfg_1 and conv_cfg_2 for ConvSE)
        base_bert_layer_config : bert_config (kept for compatibility, unused)
        hidden_dim          : feature dimension d
    """

    def __init__(self, config, base_bert_layer_config, hidden_dim):
        super().__init__()
        # Two separate BiGRUs for start and end boundaries.
        # Weights are kept independent; only the forward pass is merged.
        self.begin_GRU = nn.GRU(
            hidden_dim, hidden_dim // 2, num_layers=2,
            bidirectional=True, batch_first=True)
        self.end_GRU = nn.GRU(
            hidden_dim, hidden_dim // 2, num_layers=2,
            bidirectional=True, batch_first=True)

        # Convolutional scorers
        self.begin_score = ConvSE(config)
        self.end_score   = ConvSE(config)

    @staticmethod
    def _gru_packed(gru, G, lengths, T):
        """Run a BiGRU with pack/unpad to skip computation on padded positions."""
        packed = pack_padded_sequence(
            G, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = gru(packed)
        feat, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=T)
        return feat                                         # (B, T, d)

    def forward(self, G, video_mask):
        """
        Args:
            G          : (B, T, d)   contextualized clip features
            video_mask : (B, T)      1 = valid clip, 0 = padding
        Returns:
            begin_score_distribution : (B, T)  start logits
            end_score_distribution   : (B, T)  end logits
        """
        T = G.size(1)
        # Sequence lengths per sample — move to CPU as required by pack_padded_sequence.
        lengths = video_mask.sum(dim=1).long().cpu()

        # pack_padded_sequence skips padded timesteps, reducing GRU FLOPs
        # proportional to the average padding ratio.
        begin_feat = self._gru_packed(self.begin_GRU, G, lengths, T)
        end_feat   = self._gru_packed(self.end_GRU,   G, lengths, T)

        # ConvSE expects channel-first: (B, d, T)
        begin_scores = self.begin_score(begin_feat.transpose(1, 2), video_mask)
        end_scores   = self.end_score(end_feat.transpose(1, 2),     video_mask)

        return begin_scores, end_scores
