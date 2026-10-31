# SPDX-License-Identifier: AGPL-3.0-only
"""Fixed-budget row sampling of dense weight gradients, with exact dX/db.

The caller weights losses before backward, excludes shared sampled parameters,
checks every model gradient before optimizer.step, and owns effective batches.
Exact dispatch is native F.linear (including its native differentiation policy).
Sampled backwards check their produced gradients and reject higher derivatives.
Dynamic shared-weight detection requires Torch's current-backward-task primitive;
no private optimizer/kernel implementation is used.
"""
from __future__ import annotations

import hashlib
import json
import math
import weakref
from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def named_seed(base_seed: int, domain: str, *ids: str) -> int:
    """Resolve the prospective compact ASCII JSON SHA256 named stream."""
    if type(base_seed) is not int or not isinstance(domain, str) or any(not isinstance(i, str) for i in ids):
        raise ValueError('named seeds require integer base_seed and string domain/ids')
    encoded = json.dumps([base_seed, domain, *ids], separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big') % (2**63-1)


@dataclass
class SamplingState:
    base_seed: int
    layer_id: str
    sampler: str = 'importance'
    rho: float = .25
    seed: int = field(init=False)
    generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self):
        if self.sampler not in ('importance', 'uniform'):
            raise ValueError('sampler must be importance or uniform')
        if self.rho not in (.125, .25, .5, 1):
            raise ValueError('rho must be one of 1/8, 1/4, 1/2, 1')
        self.seed = named_seed(self.base_seed, 'sampling', self.layer_id, self.sampler)
        self.generator = torch.Generator(device='cpu').manual_seed(self.seed)

    def state_dict(self) -> dict:
        return {'base_seed': self.base_seed, 'layer_id': self.layer_id, 'sampler': self.sampler,
                'rho': self.rho, 'seed': self.seed, 'generator_state': self.generator.get_state().clone()}

    def load_state_dict(self, state: dict) -> None:
        for key in ('base_seed', 'layer_id', 'sampler', 'rho', 'seed'):
            if state.get(key) != getattr(self, key):
                raise ValueError(f'sampling state metadata mismatch: {key}')
        rng = state.get('generator_state')
        if not isinstance(rng, Tensor) or rng.device.type != 'cpu' or rng.dtype != torch.uint8:
            raise ValueError('generator_state must be a CPU byte tensor')
        self.generator.set_state(rng.clone())


def _finite(tensor: Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f'{name} must be finite')


def _operand_scales(x2: Tensor, d2: Tensor) -> tuple[Tensor, Tensor]:
    # Preserve CPU FP64 extreme reference scales until AFTER normalization.
    sx, sd = x2.detach().abs().amax(), d2.detach().abs().amax()
    _finite(sx, 'X scale')
    _finite(sd, 'D scale')
    return sx, sd


def importance_probabilities(x2: Tensor, d2: Tensor) -> Tensor | None:
    """CPU FP64 mixture probabilities; None only for proved unscaled zero.

    Only the n device FP32 scores move to CPU; complete operands stay on device.
    Nonzero operands whose normalized FP32 scores vanish are an explicit error.
    """
    if x2.ndim != 2 or d2.ndim != 2 or x2.shape[0] != d2.shape[0] or not x2.shape[0]:
        raise ValueError('score operands require matching nonempty row matrices')
    sx, sd = _operand_scales(x2, d2)
    if bool(sx == 0) or bool(sd == 0):
        return None
    xs = (x2.detach()/sx).to(dtype=torch.float32)
    ds = (d2.detach()/sd).to(dtype=torch.float32)
    scores = torch.linalg.vector_norm(xs, dim=1)*torch.linalg.vector_norm(ds, dim=1)
    _finite(scores, 'sampling scores')
    # MPS cannot construct FP64, including within a combined transfer/cast.
    scores_cpu = scores.to(device='cpu').to(dtype=torch.float64)
    total = scores_cpu.sum()
    if not bool(torch.isfinite(total)) or not bool(total > 0):
        raise RuntimeError('nonzero operands produced invalid or zero sampling score sum')
    n = len(scores_cpu)
    p = .9*(scores_cpu/total)+.1/n
    if (not bool(torch.isfinite(p).all()) or not bool((p > 0).all())
            or abs(p.sum().item()-1) > 5e-15
            or p.min().item() < .1/n*(1-1e-12)):
        raise RuntimeError('invalid sampling probabilities')
    return p


def gather_weight_gradient(x2: Tensor, d2: Tensor, indices_cpu: Tensor,
                           probabilities_cpu: Tensor, k: int) -> Tensor:
    """Actual importance gather/split-inverse-scale/small-GEMM runtime path.

    Does not dispatch k>=n: enumeration deliberately exercises duplicates for
    any supplied k. Production exact dispatch happens in sampled_linear.
    """
    if type(k) is not int or k <= 0:
        raise ValueError('k must be a positive integer')
    n = x2.shape[0]
    if (indices_cpu.device.type != 'cpu' or indices_cpu.dtype != torch.int64
            or indices_cpu.shape != (k,) or probabilities_cpu.device.type != 'cpu'
            or probabilities_cpu.dtype != torch.float64 or probabilities_cpu.shape != (n,)):
        raise ValueError('gather requires CPU int64 indices and CPU FP64 probabilities')
    if bool((indices_cpu < 0).any()) or bool((indices_cpu >= n).any()):
        raise ValueError('sample index out of bounds')
    selected = probabilities_cpu.index_select(0, indices_cpu)
    if not bool(torch.isfinite(selected).all()) or not bool((selected > 0).all()):
        raise RuntimeError('sample probabilities must be positive and finite')
    weights_cpu = (k*selected).rsqrt()
    # Cast on CPU before transferring so no FP64 operand reaches MPS.
    weights = weights_cpu.to(dtype=x2.dtype).to(device=x2.device)
    # The FP64 probabilities above also determine the inverse scales, with no
    # second normalization or independently rounded probability calculation.
    rows = indices_cpu.to(device=x2.device)
    xs = x2.index_select(0, rows)*weights[:, None]
    ds = d2.index_select(0, rows)*weights[:, None]
    dw = ds.T@xs
    _finite(dw, 'sampled dW')
    return dw


# One live weak reference and last graph-task ID per sampled weight; no graph
# or operand is retained. Different SamplingState objects cannot evade this.
_last_backward: dict[int, tuple[weakref.ReferenceType, int]] = {}


def _claim_backward(weight: Tensor) -> None:
    primitive = getattr(torch._C, '_current_graph_task_id', None)
    if primitive is None:
        raise RuntimeError('Torch current backward graph task primitive unavailable')
    task = primitive()
    if task < 0:
        raise RuntimeError('Torch current backward graph task primitive unsupported')
    key = id(weight)
    previous = _last_backward.get(key)
    if previous is not None and previous[0]() is weight and previous[1] == task:
        raise RuntimeError('same weight reused/shared in one sampled backward graph')
    def remove(ref, key=key):
        current = _last_backward.get(key)
        if current is not None and current[0] is ref:
            _last_backward.pop(key, None)
    _last_backward[key] = (weakref.ref(weight, remove), task)


class _SampledLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, state, k):
        ctx.save_for_backward(x, weight)
        ctx.state, ctx.k, ctx.has_bias = state, k, bias is not None
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, d):
        if torch.is_grad_enabled():
            raise RuntimeError('sampled_linear does not support higher-order differentiation')
        x, weight = ctx.saved_tensors
        x2, d2 = x.reshape(-1, weight.shape[1]), d.reshape(-1, weight.shape[0])
        dx = (d2@weight).reshape(x.shape) if ctx.needs_input_grad[0] else None
        db = d2.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        dw = None
        if ctx.needs_input_grad[1]:
            _claim_backward(weight)
            state, k = ctx.state, ctx.k
            if state.sampler == 'importance':
                p = importance_probabilities(x2, d2)
                if p is None:
                    dw = torch.zeros_like(weight)
                else:
                    indices = torch.multinomial(p, k, replacement=True, generator=state.generator)
                    dw = gather_weight_gradient(x2, d2, indices, p, k)
            else:
                sx, sd = _operand_scales(x2, d2)
                if bool(sx == 0) or bool(sd == 0):
                    dw = torch.zeros_like(weight)
                else:
                    rows = torch.randint(x2.shape[0], (k,), generator=state.generator).to(x2.device)
                    xs, ds = x2.index_select(0, rows), d2.index_select(0, rows)
                    # Uniform has no per-row inverse probabilities: apply its
                    # single constant to the narrower gathered operand once.
                    scale = x2.shape[0]/k
                    dw = ds.T@(xs*scale) if xs.shape[1] <= ds.shape[1] else (ds*scale).T@xs
        for value, name in ((dx, 'dX'), (dw, 'dW'), (db, 'db')):
            if value is not None:
                _finite(value, name)
        return dx, dw, db, None, None


def sampled_linear(x: Tensor, weight: Tensor, bias: Tensor | None,
                   state: SamplingState, *, k: int | None = None) -> Tensor:
    """Native forward/exact input and bias adjoints; iid sampled dense dW.

    k defaults to max(1, ceil(rho*n)). k>=n uses native F.linear directly,
    consumes no sampler RNG, and permits native shared/tied weights. The caller
    checks native and unsampled gradients for finiteness before optimizer.step.
    """
    if any(torch.is_autocast_enabled(device) for device in ('cpu', 'cuda', 'mps')):
        raise RuntimeError('sampled_linear rejects autocast/AMP')
    if (weight.ndim != 2 or x.ndim < 1 or min(weight.shape) == 0
            or x.shape[-1] != weight.shape[1]):
        raise ValueError('incompatible linear shapes')
    n = x.numel()//weight.shape[1]
    if n == 0:
        raise ValueError('empty linear rows are unsupported')
    if weight.dtype != torch.float32 and not (weight.dtype == torch.float64 and weight.device.type == 'cpu'):
        raise ValueError('dtype must be FP32 CPU/MPS or CPU FP64 reference')
    if weight.device.type not in ('cpu', 'mps'):
        raise ValueError('device must be CPU or MPS')
    tensors = (x, weight) if bias is None else (x, weight, bias)
    if any(t.dtype != weight.dtype or t.device != weight.device for t in tensors):
        raise ValueError('linear tensors must share dtype/device')
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError('bias shape must match output width')
    if not isinstance(state, SamplingState):
        raise TypeError('state must be SamplingState')
    budget = max(1, math.ceil(state.rho*n)) if k is None else k
    if type(budget) is not int or budget < 1:
        raise ValueError('k must be a positive integer')
    if budget >= n:
        return F.linear(x, weight, bias)
    return _SampledLinear.apply(x, weight, bias, state, budget)


def validate_sampled_parameters(model: nn.Module, eligible_module_names: Iterable[str],
                                optimizer_parameters: Iterable[Tensor]) -> None:
    """Reject sampled aliases/views/reused modules and nonunique optimizer sets.

    Exact ties elsewhere in the model remain allowed; the optimizer must contain
    every unique model Parameter exactly once, with no additional parameters.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    names = list(eligible_module_names)
    if len(set(names)) != len(names):
        raise ValueError('sampled module names must be unique')
    all_entries = list(model.named_parameters(remove_duplicate=False))
    unique = {id(p): p for _, p in all_entries}
    optimized = list(optimizer_parameters)
    if len(optimized) != len(unique) or {id(p) for p in optimized} != set(unique):
        raise ValueError('optimizer must contain every unique model parameter exactly once')
    for name in names:
        module = modules.get(name)
        if module is None or not isinstance(module, nn.Linear):
            raise ValueError(f'sampled module {name!r} must be Linear')
        if sum(other is module for other in modules.values()) != 1:
            raise ValueError(f'sampled module {name!r} is reused/shared')
        weight = module.weight
        if sum(p is weight for _, p in all_entries) != 1:
            raise ValueError(f'sampled weight {name!r} has multiple owners/aliases')
        storage = weight.untyped_storage()
        for param in unique.values():
            if param is weight:
                continue
            if param.device == weight.device and param.untyped_storage().data_ptr() == storage.data_ptr():
                raise ValueError(f'sampled weight {name!r} has shared parameter storage/view')
