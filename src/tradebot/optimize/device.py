# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Compute backend selection: NVIDIA (CUDA) -> AMD (ROCm) -> multi-threaded CPU.

PyTorch's ROCm builds expose AMD GPUs through the same ``torch.cuda`` API,
so one code path serves both vendors; ``torch.version.hip`` tells them apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from tradebot.core import ops


@dataclass(frozen=True)
class Backend:
    kind: str  #: "cuda", "rocm", "cpu" (torch, multi-threaded) or "numpy" (thread pool)
    device: str | None  #: torch device string, None for numpy
    description: str

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "rocm")


def select_backend(prefer: str = "auto") -> Backend:
    """``prefer``: ``auto`` (GPU, else NumPy threads), ``gpu``, ``cpu`` (torch threads) or ``numpy``."""
    torch = ops.torch
    threads = os.cpu_count() or 1
    if prefer == "numpy" or torch is None:
        if prefer == "gpu":
            raise RuntimeError("GPU requested but PyTorch is not installed (pip install 'tradebot[gpu]')")
        return Backend("numpy", None, f"NumPy, {threads} threads")
    if prefer in ("auto", "gpu") and torch.cuda.is_available():
        kind = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        return Backend(kind, "cuda", f"{kind.upper()}: {torch.cuda.get_device_name(0)}")
    if prefer == "gpu":
        raise RuntimeError("GPU requested but no CUDA/ROCm device is visible to PyTorch")
    if prefer == "auto":
        # without a GPU the NumPy thread pool beats torch-CPU on these small batched kernels
        return Backend("numpy", None, f"NumPy, {threads} threads (no CUDA/ROCm GPU found)")
    torch.set_num_threads(threads)
    return Backend("cpu", "cpu", f"PyTorch CPU, {threads} threads")
