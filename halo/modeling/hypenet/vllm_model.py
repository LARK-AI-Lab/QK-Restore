"""vLLM-compatible model for HypeNet (hybrid LightningAttn + FullAttn).

This module provides:
  - HypeNetLinearAttn   : MambaBase layer wrapping lightning attention for vLLM
  - HypeNetAttnVLLM     : Standard decoder attention (no RoPE, attn_use_rope=False)
  - HypeNetMLPVLLM      : Standard MLP block
  - HypeNetDecoderLayerVLLM: Unified decoder layer
  - HypeNetModelVLLM    : Stacked decoder body
  - HypeNetForCausalLMVLLM : Top-level causal LM (registered as "HypeNetForCausalLM")
  - register_hypenet_vllm(): registers the model class with vLLM's ModelRegistry

Usage (call once before creating any LLM / GRPOTrainer with use_vllm=True):
    from halo.modeling.hypenet.vllm_model import register_hypenet_vllm
    register_hypenet_vllm()
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla

from .lightning_attn import LightningCacheParams, _build_slope_tensor

# ---------------------------------------------------------------------------
# Helper: clear recurrent state for new (prefill) sequences so they start
# from a zero state, avoiding cross-sequence contamination.
# ---------------------------------------------------------------------------
_PAD_SLOT_ID = -1


def _clear_linear_cache_for_new_seqs(
    kv_cache: torch.Tensor,
    state_indices: torch.Tensor,
    attn_metadata: LinearAttentionMetadata,
) -> None:
    num_prefills = getattr(attn_metadata, "num_prefills", 0)
    if num_prefills <= 0:
        return
    num_decode_tokens = getattr(attn_metadata, "num_decode_tokens", 0)
    for prefill_idx in range(num_prefills):
        q_start = attn_metadata.query_start_loc[num_decode_tokens + prefill_idx]
        q_end = attn_metadata.query_start_loc[num_decode_tokens + prefill_idx + 1]
        context_len = (
            attn_metadata.seq_lens[num_decode_tokens + prefill_idx] - (q_end - q_start)
        )
        if context_len == 0:
            slot = state_indices[num_decode_tokens + prefill_idx]
            kv_cache[slot] = 0


# ---------------------------------------------------------------------------
# Lightning attention layer (MambaBase) for vLLM
# ---------------------------------------------------------------------------

class HypeNetLinearAttn(nn.Module, MambaBase):
    """vLLM-compatible LightningAttention layer for HypeNet.

    Inherits MambaBase so vLLM allocates a (max_batch, H, D, D) recurrent
    state buffer and routes the 'linear_attention' backend to this layer.
    """

    @property
    def mamba_type(self) -> str:
        return "linear_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None and self.cache_config is not None
        return MambaStateDtypeCalculator.linear_attention_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return MambaStateShapeCalculator.linear_attention_state_shape(
            num_heads=self.num_heads,
            tp_size=1,  # TP always 1 for rollout
            head_dim=self.head_dim,
        )

    def __init__(
        self,
        config,
        layer_idx: int,
        model_config=None,
        cache_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.lightning_nh
        self.num_kv_heads = config.lightning_nkv
        self.head_dim = config.lightning_head_dim
        self.hidden_size = config.hidden_size
        self.use_qk_norm = config.lightning_use_qk_norm
        self.use_output_norm = config.lightning_use_output_norm
        self.use_output_gate = config.lightning_use_output_gate
        self.use_rope = config.lightning_use_rope
        self.rope_head_dim = getattr(config, "lightning_rope_head_dim", None) or self.head_dim
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        # Scale factor (mirrors LightningAttention)
        scale_cfg = getattr(config, "lightning_scale", "1/sqrt(d)")
        if scale_cfg == "1/sqrt(d)":
            self.scale = self.head_dim ** (-0.5)
        elif scale_cfg == "1/d":
            self.scale = self.head_dim ** (-1.0)
        else:
            self.scale = 1.0

        # Projection weights (names match HuggingFace checkpoint)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if self.use_output_norm:
            self.o_norm = RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
        if self.use_output_gate:
            self.z_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        # Precompute decay slopes: shape (H,), negated for exponential decay
        slopes = _build_slope_tensor(self.num_heads)   # (H,)
        self.register_buffer("slopes", (-slopes).to(torch.float32))  # (H,)

        # RoPE (lightning_use_rope=True for the 2-B model)
        if self.use_rope:
            rope_theta = getattr(config, "rope_theta", 1_000_000.0)
            rope_scaling = getattr(config, "lightning_rope_scaling", None) or getattr(
                config, "rope_scaling", None
            )
            self.rotary_emb = get_rope(
                head_size=self.rope_head_dim,
                rotary_dim=self.rope_head_dim,
                max_position=getattr(config, "max_position_embeddings", 32768),
                base=rope_theta,
                rope_scaling=rope_scaling,
            )

        # Register in vLLM's compilation static forward context
        try:
            compilation_config = get_current_vllm_config().compilation_config
            if prefix not in compilation_config.static_forward_context:
                compilation_config.static_forward_context[prefix] = self
        except Exception:
            pass  # Not critical if compilation context is unavailable at init time

    # ------------------------------------------------------------------ #
    def _attn_fn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ):
        """Run chunk_simple_gla or fused_recurrent based on sequence length."""
        seqlen = q.shape[1]
        if seqlen < 64:
            o, final = fused_recurrent_simple_gla(
                q=q, k=k, v=v, g_gamma=decay,
                initial_state=initial_state, output_final_state=True,
                scale=self.scale,
            )
        else:
            o, final = chunk_simple_gla(
                q=q, k=k, v=v, g_gamma=decay,
                initial_state=initial_state, output_final_state=True,
                scale=self.scale,
            )
        return o, final

    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: torch.Tensor,  # (total_tokens, hidden_size)
        positions: torch.Tensor,       # (total_tokens,)
    ) -> torch.Tensor:
        total_tokens = hidden_states.shape[0]

        # ── Retrieve metadata & KV cache from vLLM forward context ──────
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(self.prefix)

        # During profiling/warmup vLLM may call forward without real metadata.
        # Fall back to a simple non-stateful forward in that case.
        if attn_metadata is None:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
            v = v.view(total_tokens, self.num_kv_heads, self.head_dim)
            if self.num_kv_heads < self.num_heads:
                g = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(g, dim=1)
                v = v.repeat_interleave(g, dim=1)
            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            s = self.slopes.to(hidden_states.device, dtype=torch.float32)
            q = q.unsqueeze(0).to(torch.float32)
            k = k.unsqueeze(0).to(torch.float32)
            v = v.unsqueeze(0).to(torch.float32)
            o, _ = self._attn_fn(q, k, v, s, None)
            output = o.squeeze(0).to(hidden_states.dtype).view(total_tokens, -1)
            if self.use_output_norm:
                output = self.o_norm(output)
            if self.use_output_gate:
                output = output * torch.sigmoid(self.z_proj(hidden_states))
            return self.o_proj(output)

        kv_cache = self.kv_cache[forward_context.virtual_engine][0]
        # kv_cache: (max_batch, H, D, D)

        # ── Projections ──────────────────────────────────────────────────
        q = self.q_proj(hidden_states)  # (T, nh*hd)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (T, H, D)
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

        # Expand KV heads to match Q if GQA
        if self.num_kv_heads < self.num_heads:
            g = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(g, dim=1)  # (T, nh, hd)
            v = v.repeat_interleave(g, dim=1)

        # Per-head QK norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE applied to the first rope_head_dim dims
        if self.use_rope:
            # vLLM's rotary_emb expects (T, H*rope_hd) tensors
            q_r = q[..., :self.rope_head_dim].contiguous().view(total_tokens, -1)
            k_r = k[..., :self.rope_head_dim].contiguous().view(total_tokens, -1)
            q_r, k_r = self.rotary_emb(positions, q_r, k_r)
            q[..., :self.rope_head_dim] = q_r.view(total_tokens, self.num_heads, self.rope_head_dim)
            k[..., :self.rope_head_dim] = k_r.view(total_tokens, self.num_heads, self.rope_head_dim)

        # ── Recurrent state management ───────────────────────────────────
        state_indices = attn_metadata.state_indices_tensor   # (num_seqs,)
        query_start_loc = attn_metadata.query_start_loc      # (num_seqs+1,)
        num_seqs = state_indices.shape[0]

        _clear_linear_cache_for_new_seqs(kv_cache, state_indices, attn_metadata)

        # Cast to float32 for kernel stability
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        s = self.slopes.to(k.device, dtype=torch.float32)  # (H,)

        # ── Fast batched decode (all sequences have exactly 1 token) ─────
        if total_tokens == num_seqs:
            q_b = q.unsqueeze(1)   # (B, 1, H, D)
            k_b = k.unsqueeze(1)
            v_b = v.unsqueeze(1)

            valid_mask = (state_indices != _PAD_SLOT_ID)
            safe_idx = state_indices.clone()
            safe_idx[~valid_mask] = 0

            orig_state = kv_cache[safe_idx]                  # (B, H, D, D)
            init_b = orig_state.to(torch.float32)
            init_b[~valid_mask] = 0.0

            o_b, final_b = self._attn_fn(q_b, k_b, v_b, s, init_b)

            # CUDA-graph-compatible unconditional write-back
            valid_f = valid_mask.to(final_b.dtype).view(-1, 1, 1, 1)
            final_write = (final_b * valid_f
                           + orig_state.to(final_b.dtype) * (1.0 - valid_f))
            kv_cache[safe_idx] = final_write.to(kv_cache.dtype)

            output_2d = o_b.squeeze(1).to(hidden_states.dtype)  # (B, H, D)

        else:
            # ── Per-sequence path (prefill / mixed batches) ───────────────
            output_2d = torch.zeros(
                total_tokens, self.num_heads, self.head_dim,
                dtype=torch.float32, device=hidden_states.device,
            )

            for seq_i in range(num_seqs):
                cache_idx = state_indices[seq_i].item()
                start = query_start_loc[seq_i].item()
                end = query_start_loc[seq_i + 1].item()
                seq_len = end - start

                q_i = q[start:end].unsqueeze(0)  # (1, T_i, H, D)
                k_i = k[start:end].unsqueeze(0)
                v_i = v[start:end].unsqueeze(0)

                # For prefill from scratch: zero initial state.
                # For decode (seq_len==1): load saved state.
                init_state = None
                if cache_idx != _PAD_SLOT_ID and seq_len == 1:
                    init_state = kv_cache[cache_idx].unsqueeze(0).to(torch.float32)

                o_i, final_i = self._attn_fn(q_i, k_i, v_i, s, init_state)
                output_2d[start:end] = o_i[0]

                if cache_idx != _PAD_SLOT_ID:
                    kv_cache[cache_idx].copy_(final_i[0].to(kv_cache.dtype))

            output_2d = output_2d.to(hidden_states.dtype)

        # ── Post-processing ───────────────────────────────────────────────
        output = output_2d.view(total_tokens, -1)  # (T, H*D)

        if self.use_output_norm:
            output = self.o_norm(output)

        if self.use_output_gate:
            z = torch.sigmoid(self.z_proj(hidden_states))
            output = output * z

        return self.o_proj(output)  # (T, hidden_size)


# ---------------------------------------------------------------------------
# Standard decoder attention (no RoPE; attn_use_rope=False for hypenet-2b)
# ---------------------------------------------------------------------------

class HypeNetAttnVLLM(nn.Module):
    """Full-attention decoder layer without RoPE (attn_use_rope=False)."""

    def __init__(
        self,
        config,
        layer_idx: int,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.use_output_gate = getattr(config, "attn_use_output_gate", False)

        # Projection weights (separate, matching HF checkpoint names)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Per-head QK norm (always present in hypenet attention layers)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if self.use_output_gate:
            self.o_gate = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=False
            )

        scale = self.head_dim ** -0.5
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=scale,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,  # (T, hidden_size)
    ) -> torch.Tensor:
        T = hidden_states.shape[0]

        q = self.q_proj(hidden_states)  # (T, nh*hd)
        k = self.k_proj(hidden_states)  # (T, nkv*hd)
        v = self.v_proj(hidden_states)  # (T, nkv*hd)

        # Per-head QK norm (q/k_norm has hidden_size=head_dim; reshape to apply)
        q = self.q_norm(q.view(T, self.num_heads, self.head_dim)).view(T, -1)
        k = self.k_norm(k.view(T, self.num_kv_heads, self.head_dim)).view(T, -1)

        # NO RoPE (attn_use_rope=False) — positions not used here
        output = self.attn(q, k, v)  # (T, nh*hd)

        if self.use_output_gate:
            output = output * torch.sigmoid(self.o_gate(hidden_states))

        return self.o_proj(output)  # (T, hidden_size)


# ---------------------------------------------------------------------------
# MLP block
# ---------------------------------------------------------------------------

class HypeNetMLPVLLM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class HypeNetDecoderLayerVLLM(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        layer_type: str,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_type = layer_type

        attn_prefix = f"{prefix}.self_attn"
        if layer_type in ("lightning-attn", "lightning_attn"):
            self.self_attn = HypeNetLinearAttn(
                config=config,
                layer_idx=layer_idx,
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                prefix=attn_prefix,
            )
        elif layer_type == "attn":
            self.self_attn = HypeNetAttnVLLM(
                config=config,
                layer_idx=layer_idx,
                vllm_config=vllm_config,
                prefix=attn_prefix,
            )
        else:
            raise ValueError(
                f"HypeNet vLLM model: unsupported layer type '{layer_type}' at layer {layer_idx}. "
                "Only 'attn' and 'lightning-attn' are supported."
            )

        self.mlp = HypeNetMLPVLLM(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Time mixing
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type in ("lightning-attn", "lightning_attn"):
            # lightning attention: forward(hidden_states, positions)
            hidden_states = self.self_attn(hidden_states, positions)
        else:
            # standard attention: forward(positions, hidden_states)
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        # Channel mixing
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# ---------------------------------------------------------------------------
# Model body
# ---------------------------------------------------------------------------

class HypeNetModelVLLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.layers = nn.ModuleList([
            HypeNetDecoderLayerVLLM(
                config=config,
                layer_idx=i,
                layer_type=config.mixer_types[i],
                vllm_config=vllm_config,
                prefix=f"{prefix}.layers.{i}" if prefix else f"layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)

        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# Top-level Causal LM — registered as "HypeNetForCausalLM"
# ---------------------------------------------------------------------------

class HypeNetForCausalLMVLLM(nn.Module, HasInnerState, IsHybrid):
    """vLLM-compatible HypeNetForCausalLM.

    Registered with ModelRegistry under 'HypeNetForCausalLM' so that vLLM
    can load any checkpoint saved with model_type='hypenet'.
    """

    has_inner_state: bool = True
    is_hybrid: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = HypeNetModelVLLM(vllm_config=vllm_config, prefix=model_prefix)

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # Placeholder kv_cache list for flash-attention layers (vLLM bookkeeping)
        flash_layer_count = sum(
            1 for t in config.mixer_types if t == "attn"
        )
        self.kv_cache = [torch.tensor([]) for _ in range(flash_layer_count)]

    # ── Required by IsHybrid ─────────────────────────────────────────────
    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the recurrent state shape for lightning-attention layers."""
        hf_config = vllm_config.model_config.hf_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        return MambaStateShapeCalculator.linear_attention_state_shape(
            num_heads=hf_config.lightning_nh,
            tp_size=tp_size,
            head_dim=hf_config.lightning_head_dim,
        )

    # ── vLLM model protocol ──────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    # ── Weight loading ────────────────────────────────────────────────────
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load HuggingFace checkpoint weights.

        The model attribute names match the HuggingFace checkpoint exactly
        (e.g. model.layers.0.self_attn.q_proj.weight), so a direct lookup
        works without name remapping.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Handle tied lm_head <-> embed_tokens
        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)

        for name, loaded_weight in weights:
            if name == "lm_head.weight" and tie_word_embeddings:
                # Share embed_tokens weight
                continue

            if name not in params_dict:
                # Skip weights that don't exist in our model (e.g. positional
                # encoding buffers from the HF model)
                continue

            param = params_dict[name]
            default_weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # If lm_head is tied, copy embed_tokens after loading
        if tie_word_embeddings and "model.embed_tokens.weight" in loaded_params:
            self.lm_head.weight.data.copy_(self.model.embed_tokens.weight.data)

        return loaded_params


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_hypenet_vllm() -> None:
    """Register HypeNetForCausalLMVLLM with vLLM's ModelRegistry.

    Call this once before creating any LLM instance that loads a HypeNet
    checkpoint (both TRL colocate mode and Verl vLLM rollout).

    Example::

        from halo.modeling.hypenet.vllm_model import register_hypenet_vllm
        register_hypenet_vllm()
        # Now vLLM can load HypeNet models
    """
    from vllm.model_executor.models import ModelRegistry
    ModelRegistry.register_model("HypeNetForCausalLM", HypeNetForCausalLMVLLM)
