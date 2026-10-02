# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 sampling params: the 8-NFE grid on the MiniMax-H3 request surface."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# 9 is the distilled 8-NFE grid. 50 is the shared comparison sample (same
# request as the 50-step MiniMax-H3 t2va runs).
_VDN_H3_STEP_COUNTS = (9, 50)


@dataclass
class VDNH3SamplingParams(MiniMaxH3SamplingParams):
    """VDN-H3: nine sigma grid points, i.e. the eight distilled DiT forwards
    (VDN counts NFEs; SGLang counts sigma grid points). The turbo adapter is
    only valid at 8 NFE with video shift 12 / audio shift 3 (the defaults).
    A 50-step request is accepted for the shared comparison sample."""

    num_inference_steps: int = 9

    def _validate(self) -> None:
        super()._validate()
        if self.num_inference_steps not in _VDN_H3_STEP_COUNTS:
            raise ValueError(
                "VDN-H3 accepts nine sigma grid points (eight distilled DiT "
                "forwards) or 50 for the shared comparison sample; got "
                f"num_inference_steps={self.num_inference_steps}."
            )
        if self.num_inference_steps != 9:
            logger.warning(
                "VDN-H3 is distilled for nine sigma grid points (eight DiT "
                "forwards). num_inference_steps=%s is the comparison sample, "
                "not the distilled schedule.",
                self.num_inference_steps,
            )
        if self.task is not None and self.task.strip().lower() not in (
            "t2va",
            "fl2va",
        ):
            raise ValueError(
                "VDN-H3 serves t2va and fl2va; ref2va was not trained (got "
                f"task={self.task!r}). Use MiniMaxAI/MiniMax-H3 --model-variant "
                "ref2va for that task."
            )


__all__ = ["VDNH3SamplingParams"]
