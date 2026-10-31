"""Detached AdamW preview and single-use commit for eager FP32/FP64 tensors.

The qualified rule is native non-AMSGrad, non-differentiable, non-fused,
non-foreach AdamW, with scalar hyperparameters and L2 global clipping.
All transfers, finite checks and preview/commit work belong in active time.
The raw alignment reduction explicitly transfers to CPU, then uses FP64.
"""

import math

import torch
from torch import Tensor

from .records import AdamWGroup, AdamWParamState, PendingUpdate


def _valid_number(value, lower=0., upper=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and lower <= value < upper


def _tensor(value, reference=None):
    if (not isinstance(value, Tensor) or value.layout != torch.strided
            or value.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(value).all()):
        raise ValueError('require finite dense FP32/FP64 tensors')
    if reference is not None and (value.shape != reference.shape
                                 or value.dtype != reference.dtype
                                 or value.device != reference.device):
        raise ValueError('gradient/moment shape, dtype and device must match parameter')


def _writable_parameter(value):
    # Sufficient nonoverlap proof for transposes and ordinary strided slices,
    # including holes. Each next dimension must start beyond the entire span
    # of the smaller-stride dimensions; uncertain exotic layouts fail closed.
    if not value.numel():
        return
    span = 1
    for stride, size in sorted((stride, size) for stride, size in
                               zip(value.stride(), value.shape) if size > 1):
        if stride < span:
            raise ValueError('require provably nonoverlapping writable parameter layout')
        span += (size - 1) * stride


def _validate(parameters, raw_gradients, state, groups, pv, sv, clip):
    if any(type(v) is not int or v < 0 for v in (pv, sv)):
        raise ValueError('versions must be nonnegative integers')
    if clip is not None and not _valid_number(clip):
        raise ValueError('max_grad_norm must be finite and nonnegative')
    if not raw_gradients.keys() <= parameters.keys() or not state.keys() <= parameters.keys():
        raise ValueError('unknown gradient/state parameter')
    names = []
    for g in groups:
        if (not isinstance(g, AdamWGroup) or not _valid_number(g.lr)
                or not _valid_number(g.eps) or not _valid_number(g.weight_decay)
                or len(g.betas) != 2 or not all(_valid_number(b, upper=1.) for b in g.betas)):
            raise ValueError('invalid AdamW group hyperparameters')
        names.extend(g.parameter_ids)
    if len(names) != len(set(names)) or set(names) != parameters.keys():
        raise ValueError('groups must assign every parameter exactly once')
    if len({id(p) for p in parameters.values()}) != len(parameters):
        raise ValueError('parameters require unique canonical identities')
    storages = set()
    for name, p in parameters.items():
        _tensor(p)
        _writable_parameter(p)
        if p.numel():
            storage = (p.device, p.untyped_storage().data_ptr())
            if storage in storages:
                raise ValueError('aliased parameter storage requires one canonical parameter')
            storages.add(storage)
        if (grad := raw_gradients.get(name)) is not None:
            _tensor(grad, p)
        if (s := state.get(name)) is not None:
            if not isinstance(s, AdamWParamState) or type(s.step) is not int or s.step < 0:
                raise ValueError('AdamW state step must be a nonnegative integer')
            _tensor(s.exp_avg, p)
            _tensor(s.exp_avg_sq, p)
            if (s.exp_avg_sq < 0).any():
                raise ValueError('second moment must be nonnegative')


def _tensor_signature(mapping):
    return tuple((name, id(t), t._version) for name, t in sorted(mapping.items()))


def _state_signature(state):
    return tuple((name, id(s), s.step, id(s.exp_avg), s.exp_avg._version,
                  id(s.exp_avg_sq), s.exp_avg_sq._version) for name, s in sorted(state.items()))


def _pending_signature(pending):
    return (_tensor_signature(pending.candidate_parameters),
            _tensor_signature(pending.displacements), _state_signature(pending.optimizer_state))


@torch.no_grad()
def preview_adamw(parameters: dict[str, Tensor], raw_gradients: dict[str, Tensor | None],
                  state: dict[str, AdamWParamState], groups: tuple[AdamWGroup, ...],
                  parameter_version: int, state_version: int,
                  max_grad_norm: float | None) -> PendingUpdate:
    """Preview a complete native-equivalent update without modifying live inputs.

    Missing keys and None gradients skip decay, state initialization and counters;
    zero tensors participate. Candidate parameters, actual stored displacements
    and pending moments own separate storage. Clipping uses native stabilization
    on detached copies. Raw alignment includes decay and uses unclipped inputs.
    Parameters require provably nonoverlapping writable layouts: transposes
    and ordinary strided slices are supported; ambiguous exotic layouts reject.
    Finite candidates/moments do not guarantee finite derived displacement or
    alignment: subtraction/reduction can overflow at extreme finite endpoints.
    Consumers must fail closed on nonfinite derived metadata (as decide_margin
    does); this API does not establish arbitrary finite-input arithmetic closure.
    """
    _validate(parameters, raw_gradients, state, groups, parameter_version, state_version, max_grad_norm)
    active = [name for g in groups for name in g.parameter_ids if raw_gradients.get(name) is not None]
    candidates = {name: parameters[name].detach().clone() for name in active}
    for name in active:
        candidates[name].grad = raw_gradients[name].detach().clone()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(list(candidates.values()), max_grad_norm,
                                      foreach=False, error_if_nonfinite=True)
    tentative = {}
    for g in groups:
        beta1, beta2 = g.betas
        for name in g.parameter_ids:
            if name not in candidates:
                continue
            p = candidates[name]
            h = p.grad
            old = state.get(name)
            m = torch.zeros_like(p) if old is None else old.exp_avg.detach().clone()
            v = torch.zeros_like(p) if old is None else old.exp_avg_sq.detach().clone()
            # Native noncapturable AdamW hosts its default scalar counter on CPU.
            # Reproduce its rounding too, including the FP32 2**24 boundary.
            counter_dtype = torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
            counter = torch.tensor(0 if old is None else old.step, dtype=counter_dtype, device='cpu')
            counter.add_(1)
            step = int(counter.item())
            # Match native single-tensor operation order, including FP rounding.
            if g.weight_decay != 0:
                p.mul_(1 - g.lr * g.weight_decay)
            m.lerp_(h, 1 - beta1)
            v.mul_(beta2).addcmul_(h, h, value=1 - beta2)
            denom = (v.sqrt() / (1 - beta2 ** step) ** .5).add_(g.eps)
            p.addcdiv_(m, denom, value=-g.lr / (1 - beta1 ** step))
            p.grad = None  # Only retain candidate, displacement and pending moments.
            _tensor(p)
            _tensor(m)
            _tensor(v)
            tentative[name] = AdamWParamState(step, m, v)
    displacements = {name: parameters[name].detach() - candidates[name] for name in active}
    alignment = sum((raw_gradients[name].detach().to('cpu').to(torch.float64)
                     * displacements[name].to('cpu').to(torch.float64)).sum().item() for name in active)
    pending = PendingUpdate(parameter_version, state_version, candidates, displacements,
                            tentative, raw_alignment=alignment)
    pending._ownership = (parameters, state, _tensor_signature(parameters),
                          _state_signature(state), _pending_signature(pending))
    return pending


@torch.no_grad()
def commit_pending(pending: PendingUpdate, parameters: dict[str, Tensor],
                   state: dict[str, AdamWParamState], parameter_version: int,
                   state_version: int) -> tuple[int, int]:
    """Validate the whole transaction before writing, then transfer moment ownership.

    Callers serialize commits and advance both returned versions. In-place tensor
    writes through normal Torch APIs are detected; .data/raw-storage writes and
    concurrent mutations bypass that contract. A checkpoint-restored pending
    record is transport data, not a live transaction: preview it again.
    """
    if pending.consumed:
        raise RuntimeError('pending update already consumed')
    if (pending.parameter_version, pending.state_version) != (parameter_version, state_version):
        raise RuntimeError('pending update version mismatch')
    ownership = pending._ownership
    if ownership is None or ownership[0] is not parameters or ownership[1] is not state:
        raise RuntimeError('pending update owner mismatch')
    if (ownership[2] != _tensor_signature(parameters) or ownership[3] != _state_signature(state)
            or ownership[4] != _pending_signature(pending)):
        raise RuntimeError('pending or source tensors modified since preview (version mismatch)')
    for name in pending.candidate_parameters:
        _writable_parameter(parameters[name])
    for name, candidate in pending.candidate_parameters.items():
        parameters[name].copy_(candidate)
    state.update(pending.optimizer_state)
    pending.consumed = True
    return parameter_version + 1, state_version + 1
