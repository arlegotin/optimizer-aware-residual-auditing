"""Past-only randomized range construction with explicit paid CPU decomposition.

The exact refresh gradient must be the pre-update gradient from ``source_step``.
This module never archives it. The trainer owns preserving that gradient until
construction and including this call's complete wall time in active time.
"""

from dataclasses import dataclass, field
import time

import torch
from torch import Tensor


@dataclass(frozen=True)
class BasisState:
    """Ordered retained modes and construction diagnostics; no refresh archive.

    ``numerical_rank`` is the number of non-null modes found in the sampled
    range; stored vector count may be smaller because of ``max_rank``. CPU
    decomposition is always explicit, including when the source is on MPS.
    """

    vectors: Tensor
    numerical_rank: int
    source_step: int
    orthogonality: float
    construction_width: int
    construction_wall_s: float
    transfer_bytes: int
    decomposition_device: str = field(default='cpu', init=False)
    decomposition_dtype: str = field(default='float64', init=False)
    schema_version: int = field(default=1, init=False)
    _tensor_version: int = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, '_tensor_version', self.vectors._version)

    def for_step(self, current_step: int, rank: int | None = None) -> Tensor:
        """Select a prefix only for a strictly later logical update."""
        if type(current_step) is not int or current_step <= self.source_step:
            raise ValueError("basis source_step must be strictly in the past")
        if self.vectors._version != self._tensor_version:
            raise RuntimeError("basis vectors were modified (version mismatch)")
        if rank is None:
            rank = self.vectors.shape[1]
        if type(rank) is not int or not 0 <= rank <= self.vectors.shape[1]:
            raise ValueError("rank exceeds available basis prefix")
        return self.vectors[:, :rank]


def build_basis(gradient: Tensor, max_rank: int, oversampling: int,
                generator: torch.Generator, source_step: int) -> BasisState:
    """Q = qr(G Omega), then ordered left singular vectors of Q.T G.

    Only a separate CPU construction generator is accepted. The transfer to CPU
    precedes FP64 casting (MPS cannot perform a combined FP64 transfer). Returned
    vectors use the source dtype/device. Numerical null modes are thresholded at
    the source precision, so QR completion cannot masquerade as useful modes.
    ``oversampling=8`` is the initial caller policy; no power iteration is used.
    """
    started = time.perf_counter()
    if gradient.ndim != 2 or min(gradient.shape) == 0 or gradient.dtype not in (torch.float32, torch.float64):
        raise ValueError("gradient must be a nonempty FP32/FP64 matrix")
    if (type(max_rank) is not int or max_rank < 0 or type(oversampling) is not int or oversampling < 0
            or type(source_step) is not int or source_step < 0):
        raise ValueError("rank, oversampling and source_step must be nonnegative integers")
    if generator.device.type != 'cpu':
        raise ValueError("basis construction requires a separate CPU generator")
    # .to('cpu') pays for and completes any outstanding source computation.
    g = gradient.detach().to('cpu').to(torch.float64)
    if not torch.isfinite(g).all():
        raise ValueError("gradient must be finite")
    width = min(gradient.shape[1], max_rank + oversampling) if max_rank else 0
    if width:
        omega = torch.randn(gradient.shape[1], width, dtype=torch.float64, generator=generator, device='cpu')
        q = torch.linalg.qr(g @ omega, mode='reduced').Q
        small_left, singular, _ = torch.linalg.svd(q.T @ g, full_matrices=False)
        threshold = torch.finfo(gradient.dtype).eps * max(gradient.shape) * singular[0]
        numerical_rank = int((singular > threshold).sum())
        vectors = (q @ small_left[:, :min(max_rank, numerical_rank)]).contiguous()
    else:
        numerical_rank = 0
        vectors = g.new_empty((gradient.shape[0], 0))
    # Diagnose the rounded representation that is copied, without a second
    # device-to-host transfer. Transfers preserve these stored values exactly.
    rounded = vectors.to(dtype=gradient.dtype)
    actual = rounded.to(torch.float64)
    orthogonality = torch.linalg.vector_norm(actual.T @ actual - torch.eye(actual.shape[1], dtype=torch.float64)).item()
    vectors = rounded.to(device=gradient.device)
    if gradient.device.type == 'mps':
        torch.mps.synchronize()
    transfer_bytes = 0 if gradient.device.type == 'cpu' else (
        gradient.numel() * gradient.element_size() + vectors.numel() * vectors.element_size())
    return BasisState(vectors, numerical_rank, source_step, orthogonality, width,
                      time.perf_counter() - started, transfer_bytes)
