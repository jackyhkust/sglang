# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 released VAE decode contract."""

import subprocess
import sys
import textwrap
from unittest import mock

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae import (
    CausalAttention,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
    _apply_qk_norm,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _init_kwargs(config: MiniMaxH3VideoVAEConfig):
    with mock.patch.object(
        AutoencoderKLLegacy, "__init__", autospec=True, return_value=None
    ) as init:
        model = MiniMaxH3VideoVAE(config)
    return model, init.call_args.kwargs


@pytest.mark.parametrize(
    "mode",
    [
        None,
        "auto",
        "tiled",
    ],
)
def test_decode_mode_uses_released_tiled_recipe(mode):
    config = (
        MiniMaxH3VideoVAEConfig()
        if mode is None
        else MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    )
    model, kwargs = _init_kwargs(config)

    assert model.parallel_decode_mode == "tiled"
    assert kwargs["decoder_tiling"] is True
    assert kwargs["parallel_tiling"] is True
    assert kwargs["decoder_parallel"] is False


@pytest.mark.parametrize("mode", ["spatial", "spatial_shard", "patch"])
def test_unvalidated_decode_modes_are_rejected(mode):
    config = MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    with pytest.raises(ValueError, match="use tiled"):
        config.resolved_parallel_decode_mode()


def test_vit_attention_uses_local_usp_backend_dispatch():
    module = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention"
    with (
        mock.patch(f"{module}.current_platform.is_cuda", return_value=True),
        mock.patch(f"{module}.USPAttention", autospec=True) as usp_attention,
    ):
        Attention(heads=2, dim_head=64)

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["skip_sequence_parallel"] is True
    assert kwargs["default_attention_backend"] == AttentionBackendEnum.TORCH_SDPA
    assert kwargs["supported_attention_backends"] == {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }


def test_vit_qk_norm_supports_affine_free_rmsnorm():
    norm = nn.RMSNorm(64, elementwise_affine=False)
    hidden_states = torch.randn(1, 2, 2, 64)

    output = _apply_qk_norm(norm, hidden_states)

    assert output.shape == hidden_states.shape


_VAE_ATTN = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention"


@pytest.mark.parametrize(
    ("rocm", "arch", "env", "aiter_found", "expected"),
    [
        (False, "gfx950", None, True, None),
        (True, "gfx942", None, True, None),
        (True, "gfx950", None, True, "aiter"),
        (True, "gfx950", None, False, "math"),
        (True, "gfx950", "math", True, "math"),
        (True, "gfx950", "SDPA", True, "sdpa"),
    ],
)
def test_rocm_vit_attention_backend_selection(
    monkeypatch, rocm, arch, env, aiter_found, expected
):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        attention,
    )

    if env is None:
        monkeypatch.delenv("SGLANG_MINIMAX_H3_VAE_ROCM_ATTN", raising=False)
    else:
        monkeypatch.setenv("SGLANG_MINIMAX_H3_VAE_ROCM_ATTN", env)
    props = mock.Mock(gcnArchName=f"{arch}:sramecc+:xnack-")
    with (
        mock.patch(f"{_VAE_ATTN}.current_platform.is_rocm", return_value=rocm),
        mock.patch("torch.cuda.get_device_properties", return_value=props),
        mock.patch(
            "importlib.util.find_spec",
            return_value=mock.Mock() if aiter_found else None,
        ),
    ):
        assert attention._rocm_vit_attention_backend() == expected


def test_rocm_vit_attention_backend_rejects_unknown(monkeypatch):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        attention,
    )

    monkeypatch.setenv("SGLANG_MINIMAX_H3_VAE_ROCM_ATTN", "flash")
    props = mock.Mock(gcnArchName="gfx950")
    with (
        mock.patch(f"{_VAE_ATTN}.current_platform.is_rocm", return_value=True),
        mock.patch("torch.cuda.get_device_properties", return_value=props),
        pytest.raises(ValueError, match="SGLANG_MINIMAX_H3_VAE_ROCM_ATTN"),
    ):
        attention._rocm_vit_attention_backend()


def test_vit_aiter_attention_keeps_bshd_layout_and_fp32_uses_math():
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        attention,
    )

    calls = []
    fake_aiter = mock.Mock()
    fake_aiter.flash_attn_func.side_effect = lambda q, k, v, **kw: (
        calls.append((q.shape, kw)) or q
    )
    query = torch.randn(2, 5, 3, 64, dtype=torch.float16)
    with (
        mock.patch.object(attention, "_ROCM_VIT_ATTN", "aiter"),
        mock.patch.dict(sys.modules, {"aiter": fake_aiter}),
    ):
        out = attention._sdpa_attention(query, query, query)
        assert out is query
        assert calls == [((2, 5, 3, 64), {"dropout_p": 0.0, "causal": False})]

        fp32 = torch.randn(1, 4, 2, 8)
        reference = torch.nn.functional.scaled_dot_product_attention(
            fp32.transpose(1, 2), fp32.transpose(1, 2), fp32.transpose(1, 2)
        ).transpose(1, 2)
        torch.testing.assert_close(
            attention._sdpa_attention(fp32, fp32, fp32), reference
        )
        assert len(calls) == 1


def test_audio_vae_attention_defaults_to_local_sdpa_and_allows_fa():
    class RecordingFA(nn.Module):
        backend = AttentionBackendEnum.FA
        dtype = torch.bfloat16

        def forward(self, query, key, value):
            self.input_dtype = query.dtype
            return query

    module = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae"
    recording_fa = RecordingFA()
    with (
        mock.patch(f"{module}.current_platform.is_cuda", return_value=True),
        mock.patch(
            f"{module}.USPAttention", autospec=True, return_value=recording_fa
        ) as usp_attention,
    ):
        attention = CausalAttention(in_dim=64, out_dim=32, num_heads=2)
        output = attention(torch.randn(1, 4, 64))

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["causal"] is True
    assert kwargs["skip_sequence_parallel"] is True
    assert kwargs["default_attention_backend"] == AttentionBackendEnum.TORCH_SDPA
    assert kwargs["supported_attention_backends"] == {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }
    assert recording_fa.input_dtype == torch.bfloat16
    assert output.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_audio_snake_first_call_matches_repeated_calls():
    # a fresh process prevents earlier tests from warming a profiling JIT graph
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import torch
                from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae import Snake1d

                torch.manual_seed(42)
                activation = Snake1d(64).cuda().eval()
                with torch.inference_mode():
                    activation.alpha.uniform_(0.1, 2.0)
                    x = torch.randn(2, 64, 4096, device="cuda")
                    original_x = x.clone()
                    original_alpha = activation.alpha.clone()
                    first = activation(x)
                    for _ in range(4):
                        torch.testing.assert_close(activation(x), first, rtol=0, atol=0)
                    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
                    torch.testing.assert_close(activation.alpha, original_alpha, rtol=0, atol=0)
                """
            ),
        ],
        check=True,
        timeout=120,
    )
