"""
DeMUL – Dual-modality Encoder for Moment reterival with Uncertainty Learning.

Architecture overview (forward pass):
  1. BidVideoQueryEncoder
       project + embed visual/subtitle/query → windowed self-attn + cross-modal attn
  2. QueryWeightEncoder (MoE gate)
       NetVLAD(query) → sigmoid weights per modality
  3. Contextual QDF Refinement (2 × TransformerBlock)
       further refine each modality's features with query guidance
  4. MoE fusion
       weighted sum of visual and subtitle features
  5. Missing-modality fallback
       replace fused positions with the available single-modality feature
  6. MomentLocalizationHead (× 3 during training)
       BiGRU + ConvSE → start/end distributions for visual / text / fusion

Training loss = lw_st_ed * (L_visual + L_text + L_fusion)
where each L = CE(start) + CE(end) with shared-normalization labels.
"""

import logging
import torch
import torch.nn as nn

from model.encoder import BidVideoQueryEncoder, QueryWeightEncoder
from model.transformer import TransformerBlock
from model.head import MomentLocalizationHead

logger = logging.getLogger(__name__)


def _replace_missing_modal_features(fusion_feature, visual_feature,
                                    sub_feature, video_mask, sub_mask):
    """For clips where one modality is entirely absent (mask = 0),
    replace the fused representation with the available single-modality one.

    Args:
        fusion_feature : (B, T, d)
        visual_feature : (B, T, d)
        sub_feature    : (B, T, d)
        video_mask     : (B, T) bool – True where visual is valid
        sub_mask       : (B, T) bool – True where subtitle is valid
    Returns:
        fusion_feature : (B, T, d)  (potentially patched)
    """
    visual_missing = (~video_mask) & sub_mask    # positions with only subtitle
    sub_missing    = video_mask & (~sub_mask)    # positions with only visual
    fusion_feature = torch.where(
        visual_missing.unsqueeze(-1), sub_feature, fusion_feature)
    fusion_feature = torch.where(
        sub_missing.unsqueeze(-1), visual_feature, fusion_feature)
    return fusion_feature


class DeMUL(nn.Module):
    """DeMUL main model.

    Args:
        config            : EasyDict loaded from model_config.json
        visual_dim        : dimension of visual clip features (e.g. 4352)
        text_dim          : dimension of subtitle clip features (e.g. 768)
        query_dim         : dimension of query features (e.g. 768)
        hidden_dim        : shared hidden dimension (e.g. 384)
        video_len         : max number of clips per video (e.g. 100)
        ctx_mode          : modality specification string, e.g. "visual_sub"
        lw_st_ed          : loss weight for moment CE loss
        lw_video_ce       : loss weight for video CE loss (unused in base DeMUL)
        similarity_measure: currently unused ("general")
        use_debug         : enable DEBUG logging
    """

    def __init__(self, config,
                 visual_dim=4352, text_dim=768,
                 query_dim=768, hidden_dim=384,
                 video_len=100, ctx_mode="visual_sub",
                 lw_st_ed=0.01, lw_video_ce=0.05,
                 similarity_measure="general",
                 use_debug=False):
        super().__init__()
        self.config = config
        self.lw_st_ed = lw_st_ed
        self.lw_video_ce = lw_video_ce
        self.similarity_measure = similarity_measure
        self.video_modality = ctx_mode.split("_")
        logger.info("video modality: %s", self.video_modality)

        # ── 1. Backbone encoder ──────────────────────────────────────────────
        self.encoder = BidVideoQueryEncoder(
            config, video_modality=self.video_modality,
            visual_dim=visual_dim, text_dim=text_dim,
            query_dim=query_dim, hidden_dim=hidden_dim,
            win_size=5, num_head=8, use_rel_pe=True)

        # ── 2. MoE query-weight encoder ─────────────────────────────────────
        if len(self.video_modality) > 1:
            self.query_weight = QueryWeightEncoder(
                config.netvlad_config,
                video_modality=self.video_modality)

        # ── 3. Contextual QDF refinement (one block per modality) ────────────
        self.qdf_visual = TransformerBlock(
            n_embd=hidden_dim, n_head=8,
            mha_win_size=5, use_rel_pe=True, use_cross_modal=True)
        self.qdf_text = TransformerBlock(
            n_embd=hidden_dim, n_head=8,
            mha_win_size=5, use_rel_pe=True, use_cross_modal=True)

        # ── 4. Moment localization head (shared across modalities) ───────────
        self.moment_head = MomentLocalizationHead(
            config.moment_localization_config,
            config.bert_config,
            hidden_dim)

        self.temporal_criterion = nn.CrossEntropyLoss(reduction="mean")

        if use_debug:
            logger.setLevel(logging.DEBUG)

        self._reset_parameters()

    # ── parameter initialisation ─────────────────────────────────────────────

    def _reset_parameters(self):
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0,
                                      std=self.config.initializer_range)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.Conv1d):
                m.reset_parameters()
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        self.apply(_init)

    # ── MoE weighted sum ─────────────────────────────────────────────────────

    def _compute_fusion(self, feature_dict, moe_weights=None):
        """Compute weighted (or simple average) fusion of modality features.

        Vectorised over modalities to avoid a Python loop per forward pass.
        """
        feats = torch.stack(
            [feature_dict[mod] for mod in self.video_modality], dim=0)  # (M, ...)
        if moe_weights is not None:
            # weights: (M, B) → reshape to broadcast over the spatial/temporal dims
            weights = torch.stack(
                [moe_weights[mod] for mod in self.video_modality], dim=0)  # (M, B)
            # feats is (M, B, ...) — add singleton dims after B for broadcasting
            extra_dims = feats.dim() - 2                      # number of dims after B
            w = weights.view(*weights.shape, *([1] * extra_dims))  # (M, B, 1, ...)
            return (feats * w).sum(0)
        else:
            return feats.mean(0)

    # ── shared-normalization CE loss ─────────────────────────────────────────

    def _moment_ce_loss(self, begin_dist, end_dist, st_ed_indices):
        """Compute start + end cross-entropy with shared normalization."""
        bs, shared_n, video_len = begin_dist.size()
        begin_dist = begin_dist.view(bs, -1)
        end_dist   = end_dist.view(bs, -1)
        loss = (self.temporal_criterion(begin_dist, st_ed_indices[:, 0])
                + self.temporal_criterion(end_dist,   st_ed_indices[:, 1]))
        return loss

    # ── forward pass ─────────────────────────────────────────────────────────

    def _encode(self, batch):
        """Run encoder + QDF refinement, return per-modality & fused features."""
        video_feat, query_feat = self.encoder(batch)

        # sizes
        query_batch = query_feat.size(0)
        video_batch, video_len = video_feat["visual"].size()[:2]
        shared_video_num = video_batch // query_batch

        # expand query to match video batch
        q_exp      = torch.repeat_interleave(query_feat, shared_video_num, dim=0)
        q_mask_exp = torch.repeat_interleave(
            batch["query"]["feat_mask"], shared_video_num, dim=0)

        video_mask = batch["visual"]["feat_mask"]
        sub_mask   = batch["sub"]["feat_mask"]
        fused_mask = video_mask | sub_mask

        # contextual QDF refinement
        video_feat["visual"] = self.qdf_visual(
            video_feat["visual"].transpose(-1, -2),
            video_mask.unsqueeze(1),
            q_exp.transpose(-1, -2),
            q_mask_exp.unsqueeze(1),
        )[0].transpose(-1, -2)

        video_feat["sub"] = self.qdf_text(
            video_feat["sub"].transpose(-1, -2),
            sub_mask.unsqueeze(1),
            q_exp.transpose(-1, -2),
            q_mask_exp.unsqueeze(1),
        )[0].transpose(-1, -2)

        # MoE fusion
        if len(self.video_modality) > 1:
            moe_w = self.query_weight(q_exp)
            qdf_fusion = self._compute_fusion(video_feat, moe_w)
        else:
            qdf_fusion = self._compute_fusion(video_feat)

        # missing-modality fallback
        qdf_fusion = _replace_missing_modal_features(
            fusion_feature=qdf_fusion,
            visual_feature=video_feat["visual"],
            sub_feature=video_feat["sub"],
            video_mask=video_mask.bool(),
            sub_mask=sub_mask.bool(),
        )

        return (video_feat["visual"], video_feat["sub"], qdf_fusion,
                video_mask, sub_mask, fused_mask,
                query_batch, video_len, shared_video_num)

    def get_pred_from_raw_query(self, batch, is_eval=True):
        """Run the full forward pass and return score distributions.

        During eval, only the fused distribution is returned.
        During training, per-modality distributions are also returned.
        """
        (vis_feat, txt_feat, fus_feat,
         video_mask, sub_mask, fused_mask,
         query_batch, video_len, shared_video_num) = self._encode(batch)

        def _predict(feat, mask):
            bs_dist, ed_dist = self.moment_head(feat, mask)
            bs_dist = bs_dist.view(query_batch, shared_video_num, video_len)
            ed_dist = ed_dist.view(query_batch, shared_video_num, video_len)
            return bs_dist, ed_dist

        if is_eval:
            bs_fus, ed_fus = _predict(fus_feat, fused_mask)
            return None, bs_fus, ed_fus
        else:
            bs_vis, ed_vis = _predict(vis_feat, video_mask)
            bs_txt, ed_txt = _predict(txt_feat, sub_mask)
            bs_fus, ed_fus = _predict(fus_feat, fused_mask)
            return (None,
                    bs_vis, ed_vis,
                    bs_txt, ed_txt,
                    bs_fus, ed_fus)

    def forward(self, batch):
        """Training forward pass.

        Returns:
            loss       : scalar total loss
            loss_dict  : dict with individual loss values (floats)
        """
        (_, bs_vis, ed_vis,
         bs_txt, ed_txt,
         bs_fus, ed_fus) = self.get_pred_from_raw_query(batch, is_eval=False)

        st_ed = batch["st_ed_indices"]
        L_vis = self.lw_st_ed * self._moment_ce_loss(bs_vis, ed_vis, st_ed)
        L_txt = self.lw_st_ed * self._moment_ce_loss(bs_txt, ed_txt, st_ed)
        L_fus = self.lw_st_ed * self._moment_ce_loss(bs_fus, ed_fus, st_ed)

        loss = L_vis + L_txt + L_fus
        return loss, {
            "visual_ce_loss": float(L_vis),
            "text_ce_loss":   float(L_txt),
            "fusion_ce_loss": float(L_fus),
            "loss_overall":   float(loss),
        }
