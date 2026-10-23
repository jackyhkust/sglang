# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 sampling params: the 8-NFE grid on the MiniMax-H3 request surface."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# num_inference_steps counts DiT forwards. 8 is the distilled schedule (nine
# sigma grid points). 50 is the shared comparison sample (same request as the
# 50-step MiniMax-H3 t2va runs).
_VDN_H3_STEP_COUNTS = (8, 50)


@dataclass
class VDNH3SamplingParams(MiniMaxH3SamplingParams):
    """VDN-H3: eight distilled DiT forwards (nine sigma grid points). The turbo
    adapter is only valid at 8 NFE with video shift 12 / audio shift 3 (the
    defaults). A 50-step request is accepted for the shared comparison sample."""

    num_inference_steps: int = 8

    def _validate(self) -> None:
        super()._validate()
        if self.num_inference_steps not in _VDN_H3_STEP_COUNTS:
            raise ValueError(
                "VDN-H3 accepts eight inference steps (the distilled schedule, "
                "nine sigma grid points) or 50 for the shared comparison sample; "
                f"got num_inference_steps={self.num_inference_steps}."
            )
        if self.num_inference_steps != 8:
            logger.warning(
                "VDN-H3 is distilled for eight inference steps. "
                "num_inference_steps=%s is the comparison sample, not the "
                "distilled schedule.",
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
