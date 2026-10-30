"""Projected weight derivatives with exact input and bias adjoints.

The caller owns objective weighting and the effective-batch lifecycle. Create one
capture per layer and effective batch, reuse it across microbatches, then discard
it. No token-count normalization occurs here. Shared/tied parameters must be
excluded by the model integrator.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

if TYPE_CHECKING:
    from .basis import BasisState


@dataclass(frozen=True)
class LinearCapture:
    """Fixed projector/probes and detached streaming sufficient statistics.

    Tensor arguments are copied into capture-owned storage. Metadata assignment
    is frozen and in-place mutations of that storage fail version checks. Direct
    construction accepts an already qualified basis (useful for algebra tests);
    online callers use ``from_basis`` to enforce the past-only boundary.
    """

    basis: Tensor
    probes: Tensor
    basis_version: int
    h_sum: Tensor = field(init=False)
    r_sum: Tensor = field(init=False)
    backward_contributions: int = field(default=0, init=False)
    schema_version: int = field(default=1, init=False)
    _tensor_versions: tuple[int, int] = field(init=False, repr=False)
    _parameter_signature: tuple[int, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        u, z = self.basis, self.probes
        if (u.ndim != 2 or z.ndim != 2 or u.shape[0] == 0
                or u.shape[1] > u.shape[0] or z.shape[0] == 0 or z.shape[1] == 0):
            raise ValueError("basis/probes must have valid matrix shapes")
        if u.dtype not in (torch.float32, torch.float64) or u.dtype != z.dtype or u.device != z.device:
            raise ValueError("basis/probes require matching FP32/FP64 dtype and device")
        if type(self.basis_version) is not int or self.basis_version < 0:
            raise ValueError("basis_version must be a nonnegative integer")
        if not torch.isfinite(u).all() or not torch.isfinite(z).all():
            raise ValueError("basis/probes must be finite")
        if u.shape[1]:
            defect = torch.linalg.vector_norm(u.T @ u - torch.eye(u.shape[1], dtype=u.dtype, device=u.device))
            if defect > 10 * torch.finfo(u.dtype).eps * max(u.shape):
                raise ValueError("basis must be orthonormal")
        u, z = u.detach().clone(), z.detach().clone()
        object.__setattr__(self, 'basis', u)
        object.__setattr__(self, 'probes', z)
        object.__setattr__(self, 'h_sum', u.new_zeros((u.shape[1], z.shape[0])))
        object.__setattr__(self, 'r_sum', u.new_zeros((u.shape[0], z.shape[1])))
        object.__setattr__(self, '_tensor_versions', (u._version, z._version))

    @classmethod
    def from_basis(cls, state: 'BasisState', *, current_step: int, probes: Tensor,
                   basis_version: int, rank: int | None = None) -> 'LinearCapture':
        return cls(state.for_step(current_step, rank), probes, basis_version)

    def _check_versions(self):
        if (self.basis._version, self.probes._version) != self._tensor_versions:
            raise RuntimeError("fixed capture basis/probes were modified (version mismatch)")

    def _bind_parameters(self, weight: Tensor, bias: Tensor | None):
        signature = (id(weight), weight._version) + (() if bias is None else (id(bias), bias._version))
        if self._parameter_signature is None:
            object.__setattr__(self, '_parameter_signature', signature)
        elif signature != self._parameter_signature:
            raise RuntimeError("effective-batch parameter identity/version changed")


class _ProjectedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None, capture: LinearCapture):
        ctx.save_for_backward(x, weight, bias, capture.basis, capture.probes)
        ctx.capture = capture
        ctx.has_bias = bias is not None
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, d: Tensor):
        # once_differentiable alone silently detaches some create_graph requests.
        if torch.is_grad_enabled():
            raise RuntimeError("projected_linear does not support higher-order differentiation")
        capture = ctx.capture
        capture._check_versions()
        x, weight, _bias, u, z = ctx.saved_tensors
        x2 = x.reshape(-1, weight.shape[1])
        d2 = d.reshape(-1, weight.shape[0])
        h = (d2 @ u).T @ x2
        dw = u @ h
        residual = d2.T @ (x2 @ z) - u @ (h @ z)
        dx = (d2 @ weight).reshape(x.shape) if ctx.needs_input_grad[0] else None
        db = d2.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        capture.h_sum.add_(h.detach())
        capture.r_sum.add_(residual.detach())
        object.__setattr__(capture, 'backward_contributions', capture.backward_contributions + 1)
        return dx, dw if ctx.needs_input_grad[1] else None, db, None


def projected_linear(x: Tensor, weight: Tensor, bias: Tensor | None,
                     capture: LinearCapture) -> Tensor:
    """Native linear forward; exact dX/db; dW = U ((D U).T X).

    ``D`` already contains all loss weighting. Only H and R are retained by the
    capture after backward; autograd releases per-invocation saved X/W normally.
    A retained graph or repeated backward is the caller's explicit responsibility.
    """
    capture._check_versions()
    if (weight.ndim != 2 or x.ndim == 0 or min(weight.shape) == 0
            or x.shape[-1] != weight.shape[1]
            or capture.basis.shape[0] != weight.shape[0]
            or capture.probes.shape[0] != weight.shape[1]):
        raise ValueError("incompatible linear, basis or probe shapes")
    tensors = (x, weight, capture.basis, capture.probes) + (() if bias is None else (bias,))
    if any(t.dtype != weight.dtype or t.device != weight.device for t in tensors):
        raise ValueError("linear tensors must share dtype and device")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("bias shape must match output width")
    capture._bind_parameters(weight, bias)
    return _ProjectedLinear.apply(x, weight, bias, capture)
