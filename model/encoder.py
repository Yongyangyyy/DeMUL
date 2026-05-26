"""
Encoder modules for DeMUL:
  - BidVideoQueryEncoder : projects & encodes visual, subtitle, and query
                           features using TransformerBlocks with cross-modal
                           attention (query guides each video modality).
  - QueryWeightEncoder   : predicts per-modality MoE fusion weights from
                           the query via NetVLAD + FC + Sigmoid.
"""

import torch
import torch.nn as nn

from model.transformer import TransformerBlock
from model.layers import LinearLayer, NetVLAD


class BidVideoQueryEncoder(nn.Module):
    """Backbone encoder for DeMUL.

    Steps:
    1. Project each modality (visual, subtitle, query) to hidden_dim.
    2. Add positional and token-type embeddings.
    3. Self-encode the query with a standard TransformerBlock.
    4. Self-encode each video modality with a *windowed* TransformerBlock
       and simultaneously attend to the query via cross-modal attention.

    Args:
        config       : model config (EasyDict), must contain bert_config
        video_modality : list[str], e.g. ["visual", "sub"]
        visual_dim   : dimension of visual features (default 4352)
        text_dim     : dimension of subtitle features (default 768)
        query_dim    : dimension of query features (default 768)
        hidden_dim   : shared hidden dimension (default 384)
        win_size     : local attention window size (default 5)
        num_head     : number of attention heads (default 8)
        use_rel_pe   : whether to use relative position encoding
    """

    def __init__(self, config, video_modality,
                 visual_dim=4352, text_dim=768,
                 query_dim=768, hidden_dim=384,
                 win_size=5, num_head=8, use_rel_pe=True):
        super().__init__()

        bert_cfg = config.bert_config

        # input projections
        self.visual_proj = LinearLayer(in_hsz=visual_dim, out_hsz=hidden_dim)
        self.text_proj   = LinearLayer(in_hsz=text_dim,   out_hsz=hidden_dim)
        self.query_proj  = LinearLayer(in_hsz=query_dim,  out_hsz=hidden_dim)

        # shared embeddings
        self.position_embeddings    = nn.Embedding(
            bert_cfg.max_position_embeddings, hidden_dim)
        self.token_type_embeddings  = nn.Embedding(
            bert_cfg.type_vocab_size, hidden_dim)

        # per-modality layer norms & dropouts
        p = bert_cfg.hidden_dropout_prob
        self.LayerNormQuery  = nn.LayerNorm(hidden_dim, eps=bert_cfg.layer_norm_eps)
        self.LayerNormVisual = nn.LayerNorm(hidden_dim, eps=bert_cfg.layer_norm_eps)
        self.LayerNormText   = nn.LayerNorm(hidden_dim, eps=bert_cfg.layer_norm_eps)
        self.dropoutQuery    = nn.Dropout(p)
        self.dropoutVisual   = nn.Dropout(p)
        self.dropoutText     = nn.Dropout(p)

        # TransformerBlocks
        # query: standard full self-attention (no cross-modal needed)
        self.queryEncoder  = TransformerBlock(
            n_embd=hidden_dim, n_head=num_head)
        # video modalities: windowed self-attention + cross-modal (query-guided)
        self.visualEncoder = TransformerBlock(
            n_embd=hidden_dim, n_head=num_head,
            mha_win_size=win_size, use_rel_pe=use_rel_pe,
            use_cross_modal=True)
        self.textEncoder   = TransformerBlock(
            n_embd=hidden_dim, n_head=num_head,
            mha_win_size=win_size, use_rel_pe=use_rel_pe,
            use_cross_modal=True)

    def forward(self, batch):
        """
        Args:
            batch: dict with keys "query", "visual", "sub", each containing
                   "feat", "feat_mask", "feat_pos_id", "feat_token_id".
                   visual/sub shapes: (B, N, T, D); query shape: (B, L_q, D_q).
        Returns:
            video_feature_dict : {"visual": (B*N, T, d), "sub": (B*N, T, d)}
            query_feature      : (B, L_q, d)
        """
        # flatten (B, N, T, D) → (B*N, T, D) for video features
        bsz, num_video = batch["visual"]["feat"].size()[:2]
        for mod in ["visual", "sub"]:
            for key in ["feat", "feat_mask", "feat_pos_id", "feat_token_id"]:
                s = batch[mod][key].size()[2:]
                batch[mod][key] = batch[mod][key].view(
                    (bsz * num_video,) + s)

        # project and embed
        proj_query  = self.query_proj(batch["query"]["feat"])
        proj_visual = self.visual_proj(batch["visual"]["feat"])
        proj_text   = self.text_proj(batch["sub"]["feat"])

        q_emb = self.dropoutQuery(self.LayerNormQuery(
            proj_query
            + self.token_type_embeddings(batch["query"]["feat_token_id"])))

        v_emb = self.dropoutVisual(self.LayerNormVisual(
            proj_visual
            + self.token_type_embeddings(batch["visual"]["feat_token_id"])
            + self.position_embeddings(batch["visual"]["feat_pos_id"])))

        t_emb = self.dropoutText(self.LayerNormText(
            proj_text
            + self.token_type_embeddings(batch["sub"]["feat_token_id"])
            + self.position_embeddings(batch["sub"]["feat_pos_id"])))

        # encode query  →  (B, d, L_q)  then transpose back
        query_feat, query_mask = self.queryEncoder(
            q_emb.transpose(-1, -2),
            batch["query"]["feat_mask"].unsqueeze(1))

        query_batch = query_feat.size(0)
        video_batch = v_emb.size(0)
        shared_video_num = video_batch // query_batch

        # expand query to match video batch  (B → B*N)
        expand_query      = torch.repeat_interleave(
            query_feat, shared_video_num, dim=0)
        expand_query_mask = torch.repeat_interleave(
            query_mask, shared_video_num, dim=0)

        # encode visual with cross-modal attention from query
        visual_feat, _ = self.visualEncoder(
            v_emb.transpose(-1, -2),
            batch["visual"]["feat_mask"].unsqueeze(1),
            expand_query,
            expand_query_mask)

        # encode subtitle with cross-modal attention from query
        text_feat, _ = self.textEncoder(
            t_emb.transpose(-1, -2),
            batch["sub"]["feat_mask"].unsqueeze(1),
            expand_query,
            expand_query_mask)

        video_feature_dict = {
            "visual": visual_feat.transpose(-1, -2),   # (B*N, T, d)
            "sub":    text_feat.transpose(-1, -2),     # (B*N, T, d)
        }
        return video_feature_dict, query_feat.transpose(-1, -2)  # (B, L_q, d)


# ---------------------------------------------------------------------------
# QueryWeightEncoder  –  MoE gating network
# ---------------------------------------------------------------------------

class QueryWeightEncoder(nn.Module):
    """Predicts per-modality soft fusion weights from the query.

    Pipeline:
        query → NetVLAD (aggregation) → Dropout → Linear → Sigmoid
    Output is a dict {modality: weight_tensor (B,)}.
    """

    def __init__(self, config, video_modality):
        super().__init__()
        self.video_modality = video_modality
        self.text_pooling = NetVLAD(
            feature_size=config.hidden_size,
            cluster_size=config.text_cluster)
        self.moe_dropout = nn.Dropout(config.moe_dropout_prob)
        self.moe_fc = nn.Linear(
            in_features=self.text_pooling.out_dim,
            out_features=len(video_modality),
            bias=False)

    def forward(self, query_feat):
        """
        Args:
            query_feat : (B, L_q, d)
        Returns:
            moe_weights_dict : {modality: (B,)}
        """
        pooled = self.text_pooling(query_feat)      # (B, K*d)
        pooled = self.moe_dropout(pooled)
        weights = torch.sigmoid(self.moe_fc(pooled))  # (B, M)
        moe_weights_dict = {}
        for mod, w in zip(self.video_modality,
                          torch.split(weights, 1, dim=1)):
            moe_weights_dict[mod] = w.squeeze(1)   # (B,)
        return moe_weights_dict
