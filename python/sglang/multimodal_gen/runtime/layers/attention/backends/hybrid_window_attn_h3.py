# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 window-softmax backend on the MiniMax-H3 packed layout.

An exact softmax over a chunk-aligned frame window: frame t belongs to chunk
t // chunk and attends to chunks [c - radius, c + radius]; frames 0 and F-1
are dense anchors; text and audio rows are dense both ways; padding rows sit
outside every mask; a per-(token, head) sigmoid gate scales the output. The
linear branch (``minimax_h3_vdn.py``) covers the window's complement. The
metadata is request-static and installed once per request through
``set_forward_context``. The window runs as a union of dense varlen
FlashAttention calls: the dense-query rows against all keys, then per-chunk
gathered [globals | window | anchors] K/V; same math as a masked kernel up to
bf16 reduction order.
"""

from __future__ import annotations

import functools
import importlib
import logging
import re
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.multimodal_gen.configs.models.dits.minimax_h3_vdn import (
    VDNHybridAttentionArchConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    flash_attn as _flash_attn_backend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import VDNH3Layout
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

logger = logging.getLogger(__name__)

_DIT_BLOCK_PREFIX = re.compile(r"^blocks\.(\d+)\.")
# "aiter" until the Triton varlen call is rejected, then "sdpa" for the process.
_ROCM_SOFTMAX = "aiter"


class HybridWindowAttentionH3Backend(AttentionBackend):
    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.HYBRID_WINDOW_ATTN_H3

    @staticmethod
    def get_impl_cls() -> type[HybridWindowAttentionH3Impl]:
        return HybridWindowAttentionH3Impl

    @staticmethod
    def get_metadata_cls() -> type[HybridWindowAttentionH3Metadata]:
        return HybridWindowAttentionH3Metadata

    @staticmethod
    def get_builder_cls() -> type[HybridWindowAttentionH3MetadataBuilder]:
        return HybridWindowAttentionH3MetadataBuilder


def window_mask_frames(
    hybrid: VDNHybridAttentionArchConfig, num_frames: int
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """(clamped per-frame window bounds, dense-ROW frames, dense-COLUMN frames)."""
    bounds = [
        (max(lo, 0), min(hi, num_frames - 1))
        for lo, hi in hybrid.window_bounds(num_frames)
    ]
    anchors = {0, num_frames - 1} if hybrid.anchor_frames != "none" else set()
    dense_rows = anchors if hybrid.anchor_frames in ("rows", "both") else set()
    dense_cols = anchors if hybrid.anchor_frames in ("columns", "both") else set()
    return bounds, dense_rows, dense_cols


def window_mask_reference(
    hybrid: VDNHybridAttentionArchConfig, layout: VDNH3Layout, device: torch.device
) -> torch.Tensor:
    """Dense boolean [used, used] mask of the softmax branch, for tests."""
    used = layout.used
    keep = torch.ones(used, used, dtype=torch.bool, device=device)
    vs, ve = layout.video_start, layout.video_end
    bounds, dense_rows, dense_cols = window_mask_frames(hybrid, layout.num_frames)
    tpf = layout.tokens_per_frame
    rows = torch.arange(vs, ve, device=device)
    frame_of = (rows - vs) // tpf
    qf = frame_of[:, None]
    kf = frame_of[None, :]
    lo = torch.tensor([b[0] for b in bounds], device=device)[qf]
    hi = torch.tensor([b[1] for b in bounds], device=device)[qf]
    inside = (kf >= lo) & (kf <= hi)
    for f in dense_rows:
        inside |= qf == f
    for f in dense_cols:
        inside |= kf == f
    keep[vs:ve, vs:ve] = inside
    return keep


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if out and out[-1][1] >= a:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _cat_ranges(ranges: list[tuple[int, int]], *, device: torch.device) -> torch.Tensor:
    if not ranges:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(
        [torch.arange(a, b, device=device, dtype=torch.long) for a, b in ranges]
    )


def _chunk_groups(
    raw_bounds: list[tuple[int, int]], dense_rows: set[int]
) -> list[list[int]]:
    # consecutive window frames with identical bounds share one varlen segment
    groups: list[list[int]] = []
    for f in range(len(raw_bounds)):
        if f in dense_rows:
            continue
        if (
            groups
            and raw_bounds[groups[-1][-1]] == raw_bounds[f]
            and groups[-1][-1] == f - 1
        ):
            groups[-1].append(f)
        else:
            groups.append([f])
    return groups


class _ChunkGroup(msgspec.Struct, frozen=True):
    frames: list[int]
    query_rows: torch.Tensor
    kv_rows: torch.Tensor


class _WindowPass(msgspec.Struct, frozen=True):
    query_rows: torch.Tensor
    query_slice: tuple[int, int] | None  # set when the query rows are contiguous
    kv_rows: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int


def _window_pass(
    layout: VDNH3Layout, groups: list[_ChunkGroup], device: torch.device
) -> _WindowPass:
    query_lens = [int(group.query_rows.numel()) for group in groups]
    kv_lens = [int(group.kv_rows.numel()) for group in groups]
    frames = [frame for group in groups for frame in group.frames]
    contiguous = frames == list(range(frames[0], frames[0] + len(frames)))
    zero = torch.zeros(1, dtype=torch.long)
    return _WindowPass(
        query_rows=torch.cat([group.query_rows for group in groups]),
        query_slice=(
            (layout.frame_rows(frames[0])[0], layout.frame_rows(frames[-1])[1])
            if contiguous
            else None
        ),
        kv_rows=torch.cat([group.kv_rows for group in groups]),
        cu_q=torch.cat([zero, torch.tensor(query_lens).cumsum(0)]).to(
            device, torch.int32
        ),
        cu_k=torch.cat([zero, torch.tensor(kv_lens).cumsum(0)]).to(device, torch.int32),
        max_q=max(query_lens),
        max_k=max(kv_lens),
    )


def _window_passes(
    layout: VDNH3Layout,
    groups: list[_ChunkGroup],
    max_gather_rows: int,
    device: torch.device,
) -> list[_WindowPass]:
    # one varlen call per pass; consecutive chunk groups fill up to max_gather_rows
    passes: list[_WindowPass] = []
    current: list[_ChunkGroup] = []
    current_rows = 0
    for group in groups:
        rows = int(group.kv_rows.numel())
        if current and current_rows + rows > max_gather_rows:
            passes.append(_window_pass(layout, current, device))
            current, current_rows = [], 0
        current.append(group)
        current_rows += rows
    if current:
        passes.append(_window_pass(layout, current, device))
    return passes


class _DecomposedPlan:
    """Query-row groups with identical kept key sets, as dense varlen calls:
    the dense-q rows against all ``used`` keys, then each chunk of frames
    against its gathered [globals | window | anchors] keys."""

    __slots__ = ("dense_q", "dense_cu_q", "dense_cu_k", "passes")

    def __init__(
        self,
        layout: VDNH3Layout,
        hybrid: VDNHybridAttentionArchConfig,
        device: torch.device,
        max_gather_rows: int = 200_000,
    ) -> None:
        used, num_frames = layout.used, layout.num_frames
        bounds, dense_rows, dense_cols = window_mask_frames(hybrid, num_frames)
        rows = functools.partial(_cat_ranges, device=device)
        dense_ranges = _merge_ranges(
            layout.global_ranges + [layout.frame_rows(f) for f in sorted(dense_rows)]
        )
        self.dense_q = rows(dense_ranges)
        # built once: a tensor from a Python list costs a pageable H2D copy + sync
        self.dense_cu_q = torch.tensor(
            [0, int(self.dense_q.numel())], dtype=torch.int32, device=device
        )
        self.dense_cu_k = torch.tensor([0, used], dtype=torch.int32, device=device)
        groups = []
        for frames in _chunk_groups(hybrid.window_bounds(num_frames), dense_rows):
            lo, hi = bounds[frames[0]]
            kv_frames = sorted(set(range(lo, hi + 1)) | dense_cols)
            groups.append(
                _ChunkGroup(
                    frames=frames,
                    query_rows=rows(
                        _merge_ranges([layout.frame_rows(f) for f in frames])
                    ),
                    kv_rows=rows(
                        _merge_ranges(
                            layout.global_ranges
                            + [layout.frame_rows(f) for f in kv_frames]
                        )
                    ),
                )
            )
        self.passes = _window_passes(layout, groups, max_gather_rows, device)
        window_rows = sum(int(p.query_rows.numel()) for p in self.passes)
        covered = int(self.dense_q.numel()) + window_rows
        if covered != used:
            raise ValueError(
                f"window decomposition covers {covered} of {used} packed rows"
            )


@dataclass
class HybridWindowAttentionH3Metadata(AttentionMetadata):
    layout: VDNH3Layout
    # radius >= F: the window IS dense attention and the linear branch is off
    full_cover: bool
    decomposed: _DecomposedPlan | None = None
    # (cos_sin [seq_len, rope_dim] bf16, positions [seq_len]) under Ulysses, else None
    rope_cache_full: tuple[torch.Tensor, torch.Tensor] | None = None


class HybridWindowAttentionH3MetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore[override]
        self,
        *,
        layout: VDNH3Layout,
        hybrid: VDNHybridAttentionArchConfig,
        device: torch.device,
        rope_cache_full: tuple[torch.Tensor, torch.Tensor] | None = None,
        current_timestep: int = 0,
        max_gather_rows: int = 200_000,
        **kwargs: dict[str, Any],
    ) -> HybridWindowAttentionH3Metadata:
        full_cover = hybrid.full_cover(layout.num_frames)
        decomposed = None
        if not full_cover:
            decomposed = _DecomposedPlan(
                layout, hybrid, device, max_gather_rows=max_gather_rows
            )
        return HybridWindowAttentionH3Metadata(
            current_timestep=current_timestep,
            layout=layout,
            full_cover=full_cover,
            decomposed=decomposed,
            rope_cache_full=rope_cache_full,
        )


@functools.lru_cache(maxsize=1)
def _aiter_triton_varlen_func():
    # Grouped-varlen ASM (aiter.flash_attn_varlen_func) hangs on MiniMax-H3
    # packed lengths on gfx942. gfx950 is unproven for that kernel, so the
    # window and the full-cover dense leg both stay on the Triton path.
    return importlib.import_module(
        "aiter.ops.triton.attention.mha"
    ).flash_attn_varlen_func


def _sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    scale: float,
    causal: bool = False,
) -> torch.Tensor:
    """One SDPA call per varlen segment. q and k lengths may differ."""
    q_bounds = [int(x) for x in cu_q.tolist()]
    k_bounds = [int(x) for x in cu_k.tolist()]
    out = torch.empty_like(q)
    gqa = q.shape[1] != k.shape[1]
    for (qs, qe), (ks, ke) in zip(
        zip(q_bounds[:-1], q_bounds[1:]),
        zip(k_bounds[:-1], k_bounds[1:]),
    ):
        if qs == qe:
            continue
        attn = F.scaled_dot_product_attention(
            q[qs:qe].transpose(0, 1).unsqueeze(0),
            k[ks:ke].transpose(0, 1).unsqueeze(0),
            v[ks:ke].transpose(0, 1).unsqueeze(0),
            scale=scale,
            is_causal=causal and (qe - qs) == (ke - ks),
            enable_gqa=gqa,
        )
        out[qs:qe] = attn.squeeze(0).transpose(0, 1)
    return out


def _rocm_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    scale: float,
    causal: bool = False,
) -> torch.Tensor:
    global _ROCM_SOFTMAX
    if _ROCM_SOFTMAX == "sdpa":
        return _sdpa_varlen(
            q, k, v, cu_q=cu_q, cu_k=cu_k, scale=scale, causal=causal
        )
    try:
        attn_out = _aiter_triton_varlen_func()(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            cu_seqlens_q=cu_q.to(device=q.device, dtype=torch.int32).contiguous(),
            cu_seqlens_k=cu_k.to(device=k.device, dtype=torch.int32).contiguous(),
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=scale,
            causal=causal,
        )
    except (ImportError, TypeError) as exc:
        logger.warning(
            "AITER Triton varlen is unavailable for VDN-H3 window softmax (%s). "
            "Using SDPA for the dense legs.",
            exc,
        )
        _ROCM_SOFTMAX = "sdpa"
        return _sdpa_varlen(
            q, k, v, cu_q=cu_q, cu_k=cu_k, scale=scale, causal=causal
        )
    return attn_out[0] if isinstance(attn_out, tuple) else attn_out


def _fa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if torch.version.hip is not None:
        attn_out = _rocm_varlen(
            q,
            k,
            v,
            cu_q=cu_q,
            cu_k=cu_k,
            max_q=max_q,
            max_k=max_k,
            scale=scale,
        )
    else:
        attn_out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=scale,
            causal=False,
            ver=_flash_attn_backend.fa_ver,
            out=out,
        )
        attn_out = attn_out[0] if isinstance(attn_out, tuple) else attn_out
    if out is not None and attn_out.data_ptr() != out.data_ptr():
        out.copy_(attn_out)
        return out
    return attn_out


class HybridWindowAttentionH3Impl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.prefix = prefix
        match = _DIT_BLOCK_PREFIX.match(prefix)
        self.layer_idx = int(match.group(1)) if match else None
        # non-DiT callers (the token refiner) resolve this backend too: dense FA
        self._dense_fallback = _flash_attn_backend.FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Dense attention for non-DiT callers (the Qwen3-VL text encoder).

        That caller passes [batch, seq, heads, dim]. On ROCm this cannot go
        through FlashAttention 3.
        """
        if torch.version.hip is None:
            return self._dense_fallback.forward(query, key, value, attn_metadata)
        if query.ndim != 4:
            raise ValueError(
                "hybrid_window_attn_h3 dense forward on ROCm expects "
                f"[batch, seq, heads, dim], got {tuple(query.shape)}"
            )
        batch, seq_q, heads, dim = query.shape
        seq_k = int(key.shape[1])
        cu_q = torch.arange(
            0,
            (batch + 1) * seq_q,
            seq_q,
            device=query.device,
            dtype=torch.int32,
        )
        cu_k = torch.arange(
            0,
            (batch + 1) * seq_k,
            seq_k,
            device=query.device,
            dtype=torch.int32,
        )
        flat_q = query.reshape(batch * seq_q, heads, dim)
        flat_k = key.reshape(batch * seq_k, key.shape[2], dim)
        flat_v = value.reshape(batch * seq_k, value.shape[2], dim)
        out = _rocm_varlen(
            flat_q,
            flat_k,
            flat_v,
            cu_q=cu_q,
            cu_k=cu_k,
            max_q=seq_q,
            max_k=seq_k,
            scale=self.softmax_scale,
            causal=self.causal,
        )
        return out.view(batch, seq_q, heads, dim)

    def dense_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        # Full-cover clips and the token refiner take this path. FlashAttention
        # 3/4 is CUDA-only, and the same packed length hangs AITER's ASM varlen.
        if torch.version.hip is not None:
            return _fa_varlen(
                query,
                key,
                value,
                cu_q=cu_seqlens,
                cu_k=cu_seqlens,
                max_q=max_seqlen,
                max_k=max_seqlen,
                scale=self.softmax_scale,
            )
        return self._dense_fallback.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_host=cu_seqlens_host,
        )

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
        attn_metadata: HybridWindowAttentionH3Metadata | None = None,
        softmax_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """query/key/value: [T, H, D] packed rows (post-norm, post-RoPE) ->
        [T, H, D]; ``softmax_gate`` [T, H] scales the output per (row, head).
        Rows at and past ``used`` (padding) are zero."""
        if self.layer_idx is not None and attn_metadata is None:
            raise RuntimeError(
                "hybrid_window_attn_h3 needs per-request attention metadata "
                "from the MiniMax-H3 denoising stage; none was set in the "
                "forward context."
            )
        if self.layer_idx is None:
            return self.dense_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )

        meta = attn_metadata
        layout = meta.layout
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        used = int(bounds[1])
        if used != layout.used or query.shape[0] != layout.seq_len:
            raise ValueError(
                f"hybrid_window_attn_h3 metadata was built for used={layout.used} "
                f"of seq_len={layout.seq_len} rows, got used={used} of "
                f"{query.shape[0]}. The request metadata and the packed layout "
                "have diverged."
            )

        if meta.full_cover:
            out = self.dense_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        else:
            out = self._decomposed(query, key, value, meta.decomposed, used)

        if softmax_gate is not None:
            out.mul_(softmax_gate.to(out.dtype).unsqueeze(-1))
        if used < out.shape[0]:
            out[used:].zero_()
        return out

    def _decomposed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: _DecomposedPlan,
        used: int,
    ) -> torch.Tensor:
        out = torch.empty_like(query)
        key_used = key[:used]
        value_used = value[:used]
        if not key_used.is_contiguous():
            key_used = key_used.contiguous()
        if not value_used.is_contiguous():
            value_used = value_used.contiguous()
        if plan.dense_q.numel():
            out[plan.dense_q] = _fa_varlen(
                torch.index_select(query, 0, plan.dense_q),
                key_used,
                value_used,
                cu_q=plan.dense_cu_q,
                cu_k=plan.dense_cu_k,
                max_q=int(plan.dense_q.numel()),
                max_k=used,
                scale=self.softmax_scale,
            )
        for window in plan.passes:
            # index_select on contiguous copies takes the vectorized gather kernel
            keys = torch.index_select(key_used, 0, window.kv_rows)
            values = torch.index_select(value_used, 0, window.kv_rows)
            if window.query_slice is not None:
                start, stop = window.query_slice
                _fa_varlen(
                    query[start:stop],
                    keys,
                    values,
                    cu_q=window.cu_q,
                    cu_k=window.cu_k,
                    max_q=window.max_q,
                    max_k=window.max_k,
                    scale=self.softmax_scale,
                    out=out[start:stop],
                )
            else:
                out[window.query_rows] = _fa_varlen(
                    torch.index_select(query, 0, window.query_rows),
                    keys,
                    values,
                    cu_q=window.cu_q,
                    cu_k=window.cu_k,
                    max_q=window.max_q,
                    max_k=window.max_k,
                    scale=self.softmax_scale,
                )
            del keys, values
        return out


__all__ = [
    "HybridWindowAttentionH3Backend",
    "HybridWindowAttentionH3Impl",
    "HybridWindowAttentionH3Metadata",
    "HybridWindowAttentionH3MetadataBuilder",
    "window_mask_frames",
    "window_mask_reference",
]
