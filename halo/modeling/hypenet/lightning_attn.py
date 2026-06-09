import torch
from torch import nn, Tensor
from typing import Optional, Tuple
from dataclasses import dataclass
from einops import rearrange, repeat
import math
from transformers.utils import logging

import torch.nn.functional as F

from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from .modeling_qwen3 import Qwen3RMSNorm
from .configuration_hypenet import HypeNetConfig
from .modeling_qwen3 import apply_rotary_pos_emb
from .cache import HybridCache
from fla.modules import ShortConvolution


logger = logging.get_logger(__name__)


@dataclass
class LightningCacheParams:
    """Per-layer vLLM state container for LightningAttention.

    Populated by HypeNetLinearAttn.forward() from vLLM's KV-cache
    infrastructure and passed to LightningAttention.forward_vllm().

    Attributes:
        state_indices_tensor: int64 tensor of shape (num_seqs,) mapping each
            sequence in the current batch to its slot in *recurrent_state*.
            A value of -1 (PAD_SLOT_ID) marks padding sequences.
        recurrent_state: float tensor of shape (max_batch, H, D, D) that
            holds the recurrent hidden state for all active sequences.
            This is backed by vLLM's Mamba KV-cache allocation.
        query_start_loc: int32 tensor of shape (num_seqs + 1,) with the
            cumulative token counts per sequence (i.e. the prefix-sum of
            sequence lengths), used to slice the flat token dimension.
    """
    state_indices_tensor: torch.Tensor  # (num_seqs,)
    recurrent_state: torch.Tensor       # (max_batch, H, D, D)
    query_start_loc: torch.Tensor       # (num_seqs + 1,)


def _build_slope_tensor(nheads: int):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.tensor(get_slopes(nheads))  # (nheads,)
    return slopes


class LightningAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_dropout: float = 0.0,
        use_output_gate: bool = False,
        use_short_conv: bool = False,
        conv_size: int = 4,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        use_rope: bool = False,
        use_output_norm: bool = False,
        qk_norm: bool = True,
        rope_head_dim: Optional[int] = None,
        scale: str = '1/sqrt(d)',
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim
        if scale == '1/sqrt(d)':
            self.scale = self.head_dim ** (-0.5)
        elif scale == '1/d':
            self.scale = self.head_dim ** (-1.0)
        else:
            self.scale = 1.0
        self.attention_dropout = attention_dropout
        self.is_causal = True
        self.use_output_gate = use_output_gate
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        self.use_output_norm = use_output_norm
        self.rope_head_dim = rope_head_dim if rope_head_dim is not None else head_dim
        assert self.rope_head_dim <= self.head_dim
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=self.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=self.attention_bias,
        )
        if self.use_output_norm:
            self.o_norm = Qwen3RMSNorm(
                hidden_size=self.num_attention_heads * self.head_dim,
                eps=self.rms_norm_eps,
            )

        if self.use_output_gate:
            self.z_proj = nn.Linear(
                self.hidden_size,
                self.num_attention_heads * self.head_dim,
                bias=self.attention_bias,
            )

        if self.qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=self.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=self.rms_norm_eps)

        if self.use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.num_attention_heads * self.hidden_size,
                kernel_size=conv_size,
                activation='silu',
                use_fast_conv1d=False,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.num_key_value_heads * self.hidden_size,
                kernel_size=conv_size,
                activation='silu',
                use_fast_conv1d=False,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.num_key_value_heads * self.hidden_size,
                kernel_size=conv_size,
                activation='silu',
                use_fast_conv1d=False,
            )

    def attn_fn(
        self,
        q: Tensor,  # (b, t, h, d)
        k: Tensor,  # (b, t, h, d)
        v: Tensor,  # (b, t, h, d)
        decay: Tensor,  # (h,)
        scale: float | None = None,  # will use dk^(-1) if None.
        initial_state: Tensor | None = None,  # (b, h, dk, dv)
        mode: str = 'chunk',
    ) -> tuple[Tensor, Tensor]:
        seqlen = q.shape[1]
        mode = "fused_recurrent" if seqlen < 64 else "chunk"
        if mode == "chunk":
            o, final_state = chunk_simple_gla(
                q=q,
                k=k,
                v=v,
                g_gamma=decay,  # (h,)
                initial_state=initial_state,
                output_final_state=True,
                scale=scale,
            )  # (b, t, h, d)
        elif mode == "fused_recurrent":
            o, final_state = fused_recurrent_simple_gla(
                q=q,
                k=k,
                v=v,
                g_gamma=decay,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return o, final_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[HybridCache] = None,
        use_cache: Optional[bool] = False,
        # cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[HybridCache]]:
        attention_mask = None
        bsz, seqlen, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # print('============ Lightning attention input ============')
        # print(hidden_states.shape)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            q, conv_state_q = self.q_conv1d(x=q,
                                            mask=conv_mask,
                                            cache=conv_state_q,
                                            output_final_state=use_cache)
            k, conv_state_k = self.k_conv1d(x=k,
                                            mask=conv_mask,
                                            cache=conv_state_k,
                                            output_final_state=use_cache)
            v, conv_state_v = self.v_conv1d(x=v,
                                            mask=conv_mask,
                                            cache=conv_state_v,
                                            output_final_state=use_cache)

        q = rearrange(q, "b t (h d) -> b t h d", d=self.head_dim)
        k = rearrange(k, "b t (h d) -> b t h d", d=self.head_dim)
        v = rearrange(v, "b t (h d) -> b t h d", d=self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            assert (
                position_embeddings is not None
            ), "position_embeddings is required when use_rope is True"
            cos, sin = position_embeddings

            # (B, T, H, D) -> (B, H, T, D)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        if self.num_key_value_heads < self.num_attention_heads:
            group_size = self.num_attention_heads // self.num_key_value_heads
            k = repeat(k, 'b t h d -> b t (h g) d', g=group_size)  # (B, T, nh, dh)
            v = repeat(v, 'b t h d -> b t (h g) d', g=group_size)  # (B, T, nh, dh)

        s = (
            _build_slope_tensor(self.num_attention_heads).to(
                k.device, dtype=torch.float32
            )
            * (-1.0)
        )  # (h)

        initial_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            layer_state = past_key_values[self.layer_idx]
            initial_state = layer_state['recurrent_state']

        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        s = s.to(torch.float32)

        o, final_state = self.attn_fn(
            q=q,
            k=k,
            v=v,
            decay=s,
            initial_state=initial_state,
            scale=self.scale,
        )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=final_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=seqlen,
            )

        o = rearrange(o, "b t h d -> b t (h d)").contiguous().to(hidden_states.dtype)  # (b, t, d)

        if self.use_output_norm:
            o = self.o_norm(o)  # (b, t, d)

        if self.use_output_gate:
            z = F.sigmoid(self.z_proj(hidden_states))  # (b, t, d)
            o = o * z  # (b, t, d)

        y = self.o_proj(o)
        return y, None, past_key_values

    def forward_vllm(
        self,
        hidden_states: torch.Tensor,           # (1, total_tokens, hidden_size)
        lightning_cache_params,                 # LightningCacheParams for this layer
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, None]:
        """vLLM-compatible forward: handles per-sequence recurrent state via LightningStateManager."""
        PAD_SLOT_ID = -1  # vLLM 0.19.0: PAD_SLOT_ID = -1

        _, total_tokens, _ = hidden_states.shape
        hidden_2d = hidden_states[0]  # (total_tokens, hidden_size)

        q = self.q_proj(hidden_2d)  # (total_tokens, nh*hd)
        k = self.k_proj(hidden_2d)
        v = self.v_proj(hidden_2d)

        q = rearrange(q, "t (h d) -> t h d", d=self.head_dim)
        k = rearrange(k, "t (h d) -> t h d", d=self.head_dim)
        v = rearrange(v, "t (h d) -> t h d", d=self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.num_key_value_heads < self.num_attention_heads:
            g = self.num_attention_heads // self.num_key_value_heads
            k = repeat(k, 't h d -> t (h g) d', g=g)
            v = repeat(v, 't h d -> t (h g) d', g=g)

        s = _build_slope_tensor(self.num_attention_heads).to(k.device, dtype=torch.float32) * (-1.0)

        state_indices = lightning_cache_params.state_indices_tensor   # (num_seqs,)
        recurrent_state = lightning_cache_params.recurrent_state       # (max_batch, H, D, D)
        query_start_loc = lightning_cache_params.query_start_loc       # (num_seqs + 1,)

        num_seqs = state_indices.shape[0]
        cos, sin = position_embeddings if (self.use_rope and position_embeddings is not None) else (None, None)

        # ── Fast batched decode path ──────────────────────────────────────────
        # When total_tokens == num_seqs every sequence has exactly 1 token
        # (pure decode step).  Process all sequences in a single Triton call
        # instead of num_seqs separate calls, reducing kernel-launch overhead
        # from O(num_seqs * num_layers) to O(num_layers).
        if total_tokens == num_seqs:
            # q/k/v: (num_seqs, H, D) → (num_seqs, 1, H, D)
            q_b = q.unsqueeze(1).to(torch.float32)
            k_b = k.unsqueeze(1).to(torch.float32)
            v_b = v.unsqueeze(1).to(torch.float32)

            if self.use_rope and cos is not None:
                # cos/sin: (1, num_seqs, rope_hd) → (num_seqs, 1, rope_hd)
                cos_b = cos[0].unsqueeze(1)[:, :, :self.rope_head_dim]
                sin_b = sin[0].unsqueeze(1)[:, :, :self.rope_head_dim]
                q_b, k_b = apply_rotary_pos_emb(q_b, k_b, cos_b, sin_b, unsqueeze_dim=2)

            # Gather initial states; replace PAD slots with slot-0 and zero out.
            valid_mask = (state_indices != PAD_SLOT_ID)        # (num_seqs,)
            safe_idx = state_indices.clone()
            safe_idx[~valid_mask] = 0
            # Save original state before zeroing (needed for CUDA-graph-compatible update)
            orig_state_b = recurrent_state[safe_idx]           # (num_seqs, H, D, D)
            init_b = orig_state_b.to(torch.float32)
            init_b[~valid_mask] = 0.0

            o_b, final_b = self.attn_fn(
                q=q_b, k=k_b, v=v_b,
                decay=s, initial_state=init_b, scale=self.scale,
            )  # o_b: (num_seqs, 1, H, D), final_b: (num_seqs, H, D, D)

            # CUDA-graph-compatible unconditional state update:
            # For valid seqs: write new final_b.
            # For PAD seqs: write back the original state (identity, no-op for slot 0).
            valid_mask_f = valid_mask.to(final_b.dtype).view(-1, 1, 1, 1)
            final_to_write = (final_b * valid_mask_f
                              + orig_state_b.to(final_b.dtype) * (1 - valid_mask_f))
            recurrent_state[safe_idx] = final_to_write.to(recurrent_state.dtype)

            output = rearrange(o_b.squeeze(1).to(hidden_states.dtype), "t h d -> t (h d)")
            output = output.unsqueeze(0)  # (1, num_seqs, H*D)

        else:
            # ── Per-sequence path (prefill / mixed batches) ───────────────────
            output_2d = torch.zeros(total_tokens, self.num_attention_heads, self.head_dim,
                                    dtype=torch.float32, device=hidden_2d.device)

            for seq_i in range(num_seqs):
                cache_idx = state_indices[seq_i].item()
                start = query_start_loc[seq_i].item()
                end = query_start_loc[seq_i + 1].item()

                q_i = q[start:end].unsqueeze(0)   # (1, T_i, H, D)
                k_i = k[start:end].unsqueeze(0)
                v_i = v[start:end].unsqueeze(0)

                if self.use_rope and cos is not None:
                    cos_i = cos[:, start:end, :self.rope_head_dim]  # (1, T_i, rope_hd)
                    sin_i = sin[:, start:end, :self.rope_head_dim]
                    q_i, k_i = apply_rotary_pos_emb(q_i, k_i, cos_i, sin_i, unsqueeze_dim=2)

                seq_len = end - start
                # For prefill (seq_len > 1): always start from zero state.
                # Loading torch.empty() cache for a brand-new sequence feeds garbage
                # into the recurrence, corrupting hidden states (causes </think>-only
                # outputs). For decode (seq_len == 1): load the saved state.
                init_state = None
                if cache_idx != PAD_SLOT_ID and seq_len == 1:
                    init_state = recurrent_state[cache_idx].unsqueeze(0).to(torch.float32)

                o_i, final_state_i = self.attn_fn(
                    q=q_i.to(torch.float32),
                    k=k_i.to(torch.float32),
                    v=v_i.to(torch.float32),
                    decay=s, initial_state=init_state, scale=self.scale,
                )  # o_i: (1, T_i, H, D), final_state_i: (1, H, D, D)

                output_2d[start:end] = o_i[0]

                if cache_idx != PAD_SLOT_ID:
                    recurrent_state[cache_idx].copy_(final_state_i[0].to(recurrent_state.dtype))

            output = rearrange(output_2d, "t h d -> t (h d)").to(hidden_states.dtype)
            output = output.unsqueeze(0)  # (1, total_tokens, H*D)

        if self.use_output_norm:
            output = self.o_norm(output)

        if self.use_output_gate:
            z = torch.sigmoid(self.z_proj(hidden_states))
            output = output * z

        y = self.o_proj(output)
        return y, None, None


def build_lightning_attn_with_attn(
    attn_layer: nn.Module,
    config: HypeNetConfig,
    layer_idx: int,
) -> nn.Module:

    layer = LightningAttention(
        layer_idx,
        hidden_size=config.hidden_size,
        num_attention_heads=config.lightning_nh,
        num_key_value_heads=config.lightning_nkv,
        head_dim=config.lightning_head_dim,
        attention_dropout=config.attention_dropout,
        use_output_gate=config.lightning_use_output_gate,
        use_output_norm=config.lightning_use_output_norm,
        attention_bias=config.attention_bias,
        rms_norm_eps=config.rms_norm_eps,
        use_rope=config.lightning_use_rope,
        qk_norm=config.lightning_use_qk_norm,
        rope_head_dim=config.head_dim,
        scale=config.lightning_scale,
        use_short_conv=config.lightning_use_short_conv,
        conv_size=config.lightning_conv_size,
    )

    if config.rand_init:
        return layer

    q_proj = attn_layer.q_proj
    k_proj = attn_layer.k_proj
    v_proj = attn_layer.v_proj
    o_proj = attn_layer.o_proj

    # (nh * head_dim, hidden_size)
    wq = q_proj.weight.data.clone()  # type: ignore
    wk = k_proj.weight.data.clone()  # type: ignore
    wv = v_proj.weight.data.clone()  # type: ignore
    wo = o_proj.weight.data.clone()  # type: ignore

    if config.expand_kv_proj:
        wk = wk.reshape(-1, config.head_dim, config.hidden_size)
        wv = wv.reshape(-1, config.head_dim, config.hidden_size)
        assert wk.shape[1] == wv.shape[1], wk.shape[1] == config.num_key_value_heads

        # Repeat KV projections to convert it to MHA
        target_kv_size = config.lightning_nkv * config.lightning_head_dim
        orig_kv_size = config.num_key_value_heads * config.head_dim
        expand_size = target_kv_size // orig_kv_size
        wk = wk.repeat_interleave(expand_size, dim=0)
        wv = wv.repeat_interleave(expand_size, dim=0)

        wk = wk.reshape(-1, config.hidden_size)
        wv = wv.reshape(-1, config.hidden_size)

    layer.q_proj.weight.data.copy_(wq)
    layer.k_proj.weight.data.copy_(wk)
    layer.v_proj.weight.data.copy_(wv)
    layer.o_proj.weight.data.copy_(wo)

    if hasattr(attn_layer, 'k_norm') and hasattr(layer, 'k_norm'):
        k_norm_weights = attn_layer.k_norm.weight.data.clone()
        layer.k_norm.weight.data.copy_(k_norm_weights)

    if hasattr(attn_layer, 'q_norm') and hasattr(layer, 'q_norm'):
        q_norm_weights = attn_layer.q_norm.weight.data.clone()
        layer.q_norm.weight.data.copy_(q_norm_weights)

    return layer
