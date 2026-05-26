"""
Transformer attention building blocks for DeMUL.
Includes:
  - LayerNorm        : channel-first (B, C, T) layer norm
  - MaskedMHCA       : full multi-head conv attention (self)
  - MaskedMHA        : full multi-head conv attention (self or cross)
  - LocalMaskedMHCA  : windowed local multi-head conv attention (self)
  - AffineDropPath   : stochastic depth with learnable affine scale
  - TransformerBlock : one Transformer layer with optional windowed
                       self-attention and cross-modal attention
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ---------------------------------------------------------------------------
# LayerNorm – channel-first format (B, C, T)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(
                torch.ones(num_channels, **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: (B, C, T) — normalise over the channel dim via a single fused kernel.
        # Transpose to (B, T, C), apply F.layer_norm on the last dim, transpose back.
        # The two .transpose() calls are O(1) views (no data copy).
        return F.layer_norm(
            x.transpose(1, 2), (self.num_channels,),
            self.weight, self.bias, self.eps
        ).transpose(1, 2)


# ---------------------------------------------------------------------------
# AffineDropPath  (stochastic depth with per-channel affine scale)
# ---------------------------------------------------------------------------

class AffineDropPath(nn.Module):
    def __init__(self, num_dim, drop_prob=0., scale_by_keep=True):
        super().__init__()
        self.scale = nn.Parameter(0.1 * torch.ones((1, num_dim, 1)))
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return self.scale * x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep_prob + torch.rand(shape, dtype=x.dtype,
                                       device=x.device)).floor_()
        if self.scale_by_keep:
            mask = mask / keep_prob
        return self.scale * x * mask


# ---------------------------------------------------------------------------
# MaskedMHCA – full self-attention (Conv1d projections)
# ---------------------------------------------------------------------------

class MaskedMHCA(nn.Module):
    """Multi-Head Conv Attention with binary mask (self-attention only)."""

    def __init__(self, n_embd, n_head, attn_pdrop=0.1, proj_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.attn_pdrop = attn_pdrop

        self.key   = nn.Conv1d(n_embd, n_embd, 1)
        self.query = nn.Conv1d(n_embd, n_embd, 1)
        self.value = nn.Conv1d(n_embd, n_embd, 1)
        self.proj  = nn.Conv1d(n_embd, n_embd, 1)
        self.proj_drop = nn.Dropout(proj_pdrop)

    def forward(self, x, mask):
        # x: (B, C, T),  mask: (B, 1, T) — may arrive as Long or bool
        mask = mask.bool()
        B, C, T = x.size()
        q = self.query(x).view(B, self.n_head, self.n_channels, T).transpose(2, 3)
        k = self.key(x).view(B, self.n_head, self.n_channels, T).transpose(2, 3)
        v = self.value(x).view(B, self.n_head, self.n_channels, T).transpose(2, 3)

        # Build additive attention bias from the padding mask: (B, 1, 1, T)
        # Invalid key positions get -inf so they contribute 0 after softmax.
        attn_bias = torch.zeros(B, 1, 1, T, dtype=q.dtype, device=q.device)
        attn_bias = attn_bias.masked_fill(~mask[:, :, None, :], float('-inf'))

        # F.scaled_dot_product_attention fuses scale/softmax/dropout/matmul
        # into a single optimised kernel (Flash Attention when available).
        dropout_p = self.attn_pdrop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=attn_bias,
                                             dropout_p=dropout_p)

        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask


# ---------------------------------------------------------------------------
# MaskedMHA – full attention supporting both self and cross attention
# ---------------------------------------------------------------------------

class MaskedMHA(nn.Module):
    """Multi-Head Conv Attention with mask; supports cross-attention."""

    def __init__(self, n_embd, n_head, attn_pdrop=0.0, proj_pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.attn_pdrop = attn_pdrop

        self.key   = nn.Conv1d(n_embd, n_embd, 1)
        self.query = nn.Conv1d(n_embd, n_embd, 1)
        self.value = nn.Conv1d(n_embd, n_embd, 1)
        self.proj  = nn.Conv1d(n_embd, n_embd, 1)
        self.proj_drop = nn.Dropout(proj_pdrop)

    def forward(self, x, mask, encoder_hidden_states=None,
                encoder_attention_mask=None):
        # x: (B, C, T),  mask: (B, 1, T) — may arrive as Long or bool
        mask = mask.bool()
        B, C, T = x.size()
        is_cross = encoder_hidden_states is not None

        if is_cross:
            kv_src  = encoder_hidden_states
            kv_mask = encoder_attention_mask.bool()
        else:
            kv_src  = x
            kv_mask = mask

        T_kv = kv_src.size(2)
        q = self.query(x).view(B, self.n_head, self.n_channels, T).transpose(2, 3)
        k = self.key(kv_src).view(B, self.n_head, self.n_channels, T_kv).transpose(2, 3)
        v = self.value(kv_src).view(B, self.n_head, self.n_channels, T_kv).transpose(2, 3)

        # Build additive key-padding bias: (B, 1, 1, T_kv)
        attn_bias = torch.zeros(B, 1, 1, T_kv, dtype=q.dtype, device=q.device)
        attn_bias = attn_bias.masked_fill(~kv_mask[:, :, None, :], float('-inf'))

        dropout_p = self.attn_pdrop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=attn_bias,
                                             dropout_p=dropout_p)

        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask


# ---------------------------------------------------------------------------
# LocalMaskedMHCA – windowed local self-attention (Conv1d projections)
# ---------------------------------------------------------------------------

class LocalMaskedMHCA(nn.Module):
    """Local (windowed) Multi-Head Conv Attention with mask.

    Uses sliding-chunk QK multiplication to achieve O(T * w * d) complexity
    instead of O(T^2 * d) for full attention.
    """

    def __init__(self, n_embd, n_head, window_size,
                 attn_pdrop=0.1, proj_pdrop=0.1, use_rel_pe=False):
        super().__init__()
        assert n_embd % n_head == 0
        assert window_size > 1 and n_head >= 1
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2
        self.use_rel_pe = use_rel_pe

        self.key   = nn.Conv1d(n_embd, n_embd, 1)
        self.query = nn.Conv1d(n_embd, n_embd, 1)
        self.value = nn.Conv1d(n_embd, n_embd, 1)
        self.proj  = nn.Conv1d(n_embd, n_embd, 1)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        if use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / n_embd) ** 0.5)

    # ---- sliding-chunk helpers ----

    @staticmethod
    def _chunk(x, window_overlap):
        x = x.view(x.size(0), x.size(1) // (window_overlap * 2),
                   window_overlap * 2, x.size(2))
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        x = F.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        bm2d = input_tensor.new_ones(
            affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        bm = bm2d[None, :, None, :]
        em = bm.flip(dims=(1, 3))
        bi = input_tensor[:, :affected_seq_len, :, :affected_seq_len + 1]
        bi.masked_fill_(bm.expand(bi.size()) == 1, float('-inf'))
        ei = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ei.masked_fill_(em.expand(ei.size()) == 1, float('-inf'))

    @staticmethod
    def _pad_and_diagonalize(x):
        nh, nc, wo, hd = x.size()
        x = F.pad(x, (0, wo + 1))
        x = x.view(nh, nc, -1)
        x = x[:, :, :-wo]
        x = x.view(nh, nc, wo, wo + hd)
        return x[:, :, :, :-1]

    def _sliding_chunks_qk_matmul(self, query, key, num_heads, window_overlap):
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        chunks_count = seq_len // window_overlap - 1
        cq = self._chunk(query, window_overlap)
        ck = self._chunk(key, window_overlap)
        diag = torch.einsum("bcxd,bcyd->bcxy", (cq, ck))
        diag = self._pad_and_transpose_last_two_dims(diag, (0, 0, 0, 1))
        scores = diag.new_empty(
            batch_size * num_heads, chunks_count + 1,
            window_overlap, window_overlap * 2 + 1)
        scores[:, :-1, :, window_overlap:] = diag[:, :, :window_overlap,
                                                   :window_overlap + 1]
        scores[:, -1, :, window_overlap:] = diag[:, -1, window_overlap:,
                                                  :window_overlap + 1]
        scores[:, 1:, :, :window_overlap] = diag[:, :,
                                                  -(window_overlap + 1):-1,
                                                  window_overlap + 1:]
        scores[:, 0, 1:window_overlap, 1:window_overlap] = diag[
            :, 0, :window_overlap - 1, 1 - window_overlap:]
        scores = scores.view(batch_size, num_heads, seq_len,
                             2 * window_overlap + 1).transpose(2, 1)
        self._mask_invalid_locations(scores, window_overlap)
        return scores

    def _sliding_chunks_av_matmul(self, attn_probs, value, num_heads,
                                  window_overlap):
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        chunks_count = seq_len // window_overlap - 1
        chunked_ap = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap,
            window_overlap, 2 * window_overlap + 1)
        padded_v = F.pad(value, (0, 0, window_overlap, window_overlap),
                         value=-1)
        cv_size = (batch_size * num_heads, chunks_count + 1,
                   3 * window_overlap, head_dim)
        cv_stride = padded_v.stride()
        cv_stride = (cv_stride[0], window_overlap * cv_stride[1],
                     cv_stride[1], cv_stride[2])
        chunked_v = padded_v.as_strided(size=cv_size, stride=cv_stride)
        chunked_ap = self._pad_and_diagonalize(chunked_ap)
        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_ap, chunked_v))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask):
        # x: (B, C, T),  mask: (B, 1, T) bool
        B, C, T = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        q *= self.scale
        att = self._sliding_chunks_qk_matmul(
            q, k, self.n_head, self.window_overlap)
        if self.use_rel_pe:
            att = att + self.rel_pe

        inv_kv_mask = torch.logical_not(
            mask[:, :, :, None].view(B, -1, 1))
        float_inv = inv_kv_mask.type_as(q).masked_fill(inv_kv_mask, -1e4)
        diag_mask = self._sliding_chunks_qk_matmul(
            float_inv.new_ones(size=float_inv.size()),
            float_inv, 1, self.window_overlap)
        att = att + diag_mask
        att = F.softmax(att, dim=-1)
        att = att.masked_fill(
            torch.logical_not(mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        out = self._sliding_chunks_av_matmul(
            att, v, self.n_head, self.window_overlap)
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask


# ---------------------------------------------------------------------------
# TransformerBlock – one complete Transformer layer
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with optional windowed self-attention
    and optional cross-modal (query-guided) attention.

    Input format: (B, C, T)  – channel-first.
    """

    def __init__(
        self,
        n_embd,
        n_head,
        n_out=None,
        n_hidden=None,
        act_layer=nn.GELU,
        attn_pdrop=0.1,
        proj_pdrop=0.1,
        path_pdrop=0.0,
        mha_win_size=-1,
        use_rel_pe=False,
        use_cross_modal=False,
    ):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd, n_head,
                window_size=mha_win_size,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe,
            )
        else:
            self.attn = MaskedMHCA(
                n_embd, n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )

        self.use_cross_modal = use_cross_modal
        if use_cross_modal:
            self.cross_attn = MaskedMHA(
                n_embd, n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )
            self.ln3 = LayerNorm(n_embd)
            self.ln4 = LayerNorm(n_embd)

        if n_hidden is None:
            n_hidden = 4 * n_embd
        if n_out is None:
            n_out = n_embd
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp  = AffineDropPath(n_out,  drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp  = nn.Identity()

    def forward(self, x, mask, cross_y=None, cross_y_mask=None,
                pos_embd=None):
        # self-attention (windowed or full)
        out, out_mask = self.attn(self.ln1(x), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = x * out_mask_float + self.drop_path_attn(out)

        # optional cross-modal attention (video attends to query)
        if self.use_cross_modal and cross_y is not None:
            cross_out, cross_out_mask = self.cross_attn(
                self.ln3(out), out_mask_float,
                self.ln4(cross_y), cross_y_mask)
            out_mask_float = out_mask.to(cross_out_mask.dtype)
            out = out * out_mask_float + self.drop_path_attn(cross_out)

        # FFN
        out = out + self.drop_path_mlp(
            self.mlp(self.ln2(out)) * out_mask_float)
        if pos_embd is not None:
            out = out + pos_embd * out_mask_float
        return out, out_mask
