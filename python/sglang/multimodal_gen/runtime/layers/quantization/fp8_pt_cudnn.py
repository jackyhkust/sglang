# SPDX-License-Identifier: Apache-2.0
"""Online per-tensor FP8 linear on cuDNN (FlashInfer ``bmm_fp8``).

Weights quantize once at load with one E4M3 scale per tensor; activations take one
dynamic per-tensor scale per call. This is the scheme the ROCm online ``fp8`` path
uses. Selected with ``SGLANG_FP8_PER_TENSOR_CUDNN=1`` on CUDA.
"""

from typing import Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod
from sglang.multimodal_gen.runtime.layers.utils import direct_register_custom_op

_OUT_DTYPES = (torch.bfloat16, torch.float16)


def _fp8_pt_cudnn_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    import flashinfer

    out_dtype = x.dtype if x.dtype in _OUT_DTYPES else torch.bfloat16
    qx, x_scale = scaled_fp8_quant(x)
    out = flashinfer.bmm_fp8(
        qx.unsqueeze(0),
        weight.unsqueeze(0),
        x_scale,
        weight_scale,
        out_dtype,
        backend="cudnn",
    ).squeeze(0)
    if bias is not None:
        out = out + bias
    return out.to(x.dtype)


def _fp8_pt_cudnn_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.empty((x.shape[0], weight.shape[1]), dtype=x.dtype, device=x.device)


# cuDNN graph build and the two-pass activation quant stay opaque to torch.compile.
direct_register_custom_op(
    op_name="fp8_pt_cudnn_linear",
    op_func=_fp8_pt_cudnn_linear,
    mutates_args=[],
    fake_impl=_fp8_pt_cudnn_linear_fake,
)


class Fp8PerTensorCudnnLinearMethod(Fp8LinearMethod):
    def __init__(self, quant_config) -> None:
        super().__init__(quant_config)
        if self.block_quant or quant_config.is_checkpoint_fp8_serialized:
            raise ValueError("per-tensor cuDNN FP8 supports online quantization only")
        # base process_weights_after_loading takes the per-tensor branch
        self.cutlass_fp8_supported = False
        self.use_marlin = False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = torch.ops.sglang.fp8_pt_cudnn_linear(
            x.reshape(-1, x.shape[-1]).contiguous(),
            layer.weight,
            layer.weight_scale,
            bias,
        )
        return out.view(*x.shape[:-1], out.shape[-1])


__all__ = ["Fp8PerTensorCudnnLinearMethod"]
