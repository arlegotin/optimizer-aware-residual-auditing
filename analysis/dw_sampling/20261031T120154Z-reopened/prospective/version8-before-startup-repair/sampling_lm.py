# SPDX-License-Identifier: AGPL-3.0-only
"""Fixed contextual byte language model and native optimizer/evaluation core.

Only candidate FFN matrices use the sampled weight adjoint. Initialization uses
one isolated CPU stream, and evaluation never enters the sampled operator.
"""
from __future__ import annotations

from array import array
from contextlib import suppress
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .sampled_linear import SamplingState, named_seed, sampled_linear, validate_sampled_parameters


@dataclass(frozen=True)
class ModelConfig:
    """Frozen production architecture; smaller dimensions support CPU fixtures."""

    blocks: int = 4
    width: int = 512
    heads: int = 8
    ffn_width: int = 2048
    context: int = 256
    vocab_size: int = 259

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in (
            self.blocks, self.width, self.heads, self.ffn_width, self.context, self.vocab_size
        )):
            raise ValueError('model dimensions must be positive integers')
        if self.width % self.heads:
            raise ValueError('width must be divisible by heads')
        if self.context > 256 or self.vocab_size != 259:
            raise ValueError('byte model requires context <=256 and vocabulary 259')


class _NoInitLinear(nn.Linear):
    """Native Linear forward without implicit global-generator initialization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias, device='cpu', dtype=torch.float32)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()
            if self.bias is not None:
                self.bias.zero_()


class _NoInitEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim, device='cpu', dtype=torch.float32)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()


class _CandidateLinear(_NoInitLinear):
    def __init__(self, in_features: int, out_features: int, sampling_state: SamplingState):
        super().__init__(in_features, out_features, bias=True)
        self.sampling_state = sampling_state

    def forward(self, x: Tensor) -> Tensor:
        if not torch.is_grad_enabled() or not self.weight.requires_grad:
            return F.linear(x, self.weight, self.bias)
        return sampled_linear(x, self.weight, self.bias, self.sampling_state)


class _ByteBlock(nn.Module):
    def __init__(self, config: ModelConfig, states: dict[str, SamplingState], index: int):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.ln1 = nn.LayerNorm(config.width, eps=1e-5, device='cpu', dtype=torch.float32)
        self.qkv = _NoInitLinear(config.width, 3 * config.width)
        self.attention_out = _NoInitLinear(config.width, config.width)
        self.ln2 = nn.LayerNorm(config.width, eps=1e-5, device='cpu', dtype=torch.float32)
        if states:
            self.ffn_in = _CandidateLinear(config.width, config.ffn_width, states[f'block{index}.ffn_in'])
            self.ffn_out = _CandidateLinear(config.ffn_width, config.width, states[f'block{index}.ffn_out'])
        else:
            self.ffn_in = _NoInitLinear(config.width, config.ffn_width)
            self.ffn_out = _NoInitLinear(config.ffn_width, config.width)

    def forward(self, x: Tensor, allow: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (part.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
                   for part in (q, k, v))
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=allow, dropout_p=0., is_causal=False)
        attention = attention.transpose(1, 2).contiguous().reshape(batch, length, width)
        x = x + self.attention_out(attention)
        return x + self.ffn_out(F.gelu(self.ffn_in(self.ln2(x)), approximate='none'))


class ByteLM(nn.Module):
    """Pre-LN causal byte model with exact tied embedding/output weights."""

    def __init__(self, config: ModelConfig = ModelConfig(), *, base_seed: int,
                 sampler: str | None = None, rho: float = .25):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError('config must be ModelConfig')
        if sampler not in (None, 'importance', 'uniform'):
            raise ValueError('sampler must be None, importance or uniform')
        self.config = config
        self.base_seed = base_seed
        self.sampler = sampler
        self.init_seed = named_seed(base_seed, 'init')
        self.init_generator = torch.Generator(device='cpu').manual_seed(self.init_seed)
        sampling_started = time.perf_counter()
        self.sampling_states = {} if sampler is None else {
            f'block{index}.{suffix}': SamplingState(base_seed, f'block{index}.{suffix}', sampler, rho)
            for index in range(config.blocks) for suffix in ('ffn_in', 'ffn_out')
        }
        self.sampling_setup_s = time.perf_counter()-sampling_started if sampler is not None else 0.
        self.resolved_seeds = {'init': self.init_seed, 'sampling': {
            name: state.seed for name, state in self.sampling_states.items()
        }}
        self.token_embedding = _NoInitEmbedding(config.vocab_size, config.width)
        self.position_embedding = _NoInitEmbedding(config.context, config.width)
        self.blocks = nn.ModuleList(_ByteBlock(config, self.sampling_states, i) for i in range(config.blocks))
        self.final_ln = nn.LayerNorm(config.width, eps=1e-5, device='cpu', dtype=torch.float32)
        self.head = _NoInitLinear(config.width, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        ordered = [self.token_embedding.weight, self.position_embedding.weight]
        for block in self.blocks:
            ordered.extend([block.qkv.weight, block.attention_out.weight, block.ffn_in.weight, block.ffn_out.weight])
        with torch.no_grad():
            for weight in ordered:
                weight.normal_(0., .02, generator=self.init_generator)
        initial = hashlib.sha256()
        for name, parameter in self.named_parameters():
            initial.update(name.encode('ascii'))
            initial.update(str(tuple(parameter.shape)).encode('ascii'))
            initial.update(parameter.detach().contiguous().numpy().tobytes())
        self.initial_parameters_sha256 = initial.hexdigest()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.head.weight = self.token_embedding.weight
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.head.weight = self.token_embedding.weight
        return result

    def sampling_state_dict(self) -> dict:
        return {name: state.state_dict() for name, state in self.sampling_states.items()}

    def load_sampling_state_dict(self, states: dict) -> None:
        if set(states) != set(self.sampling_states):
            raise ValueError('sampling state IDs mismatch')
        for name, state in self.sampling_states.items():
            state.load_state_dict(states[name])

    def forward(self, input_ids: Tensor) -> Tensor:
        # Shape/dtype metadata checks introduce no device scalar synchronization.
        # Verified collation guarantees token range and a nonempty valid prefix.
        if input_ids.ndim != 2 or input_ids.dtype != torch.int64:
            raise ValueError('input_ids must be a rank-two int64 tensor')
        batch, length = input_ids.shape
        if batch == 0 or not 1 <= length <= self.config.context:
            raise ValueError('input shape exceeds the nonempty model context')
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        causal = positions[None, :] <= positions[:, None]
        allow = (input_ids != 258)[:, None, None, :] & causal[None, None, :, :]
        for block in self.blocks:
            x = block(x, allow)
        return self.head(self.final_ln(x))


def optimizer_groups(model: ByteLM) -> list[dict]:
    """Place every unique matrix in decay and bias/LN parameters in no decay."""
    parameters = list(model.parameters())
    eligible = [f'blocks.{i}.{suffix}' for i in range(model.config.blocks)
                for suffix in ('ffn_in', 'ffn_out')] if model.sampling_states else []
    validate_sampled_parameters(model, eligible, parameters)
    return [
        {'params': [p for p in parameters if p.ndim >= 2], 'weight_decay': .01},
        {'params': [p for p in parameters if p.ndim < 2], 'weight_decay': 0.},
    ]


def make_optimizer(model: ByteLM, variant: str = 'single', lr: float = .001) -> torch.optim.AdamW:
    """Build a qualified native AdamW variant; no custom optimizer path."""
    variants = {
        'default': {},
        'single': {'foreach': False, 'fused': False},
        'foreach': {'foreach': True, 'fused': False},
        'fused': {'foreach': False, 'fused': True},
    }
    if variant not in variants:
        raise ValueError('unknown native AdamW variant')
    return torch.optim.AdamW(optimizer_groups(model), lr=lr, betas=(.9, .999), eps=1e-8, **variants[variant])


def evaluate(model: ByteLM, tokenized_articles: Sequence[Sequence[int]], windows: Sequence,
             *, batch_size: int = 16) -> dict:
    """Evaluate byte NLL in supplied source order, weighting every target once."""
    from .sampling_data import collate_blocks

    if type(batch_size) is not int or batch_size <= 0 or not windows:
        raise ValueError('evaluation requires windows and a positive integer batch_size')
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32:
        raise ValueError('evaluation requires native FP32 model parameters')
    started = time.monotonic()
    prior_training = model.training
    model.eval()
    summed_loss, valid_targets, batch_count = 0., 0, 0
    batch_sizes = []
    try:
        with torch.no_grad(), torch.autocast(device_type=parameter.device.type, enabled=False):
            for start in range(0, len(windows), batch_size):
                selected = windows[start:start + batch_size]
                batch = collate_blocks(tokenized_articles, selected, device=parameter.device)
                logits = model(batch['input_ids'])
                losses = F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                                         batch['target_ids'].reshape(-1), reduction='none',
                                         ignore_index=-100).reshape(batch['target_ids'].shape)
                cpu_losses = losses.to(device='cpu').to(dtype=torch.float64)
                scored_losses = cpu_losses[batch['target_mask_cpu']]
                if not bool(torch.isfinite(scored_losses).all()):
                    raise RuntimeError('nonfinite scored evaluation losses')
                summed_loss += scored_losses.sum().item()
                valid_targets += batch['valid_targets']
                batch_count += 1
                batch_sizes.append(len(selected))
    finally:
        model.train(prior_training)
    nll = summed_loss / valid_targets
    if not math.isfinite(nll):
        raise RuntimeError('nonfinite evaluation NLL')
    try:
        perplexity = math.exp(nll)
    except OverflowError as exc:
        raise RuntimeError('evaluation perplexity is not finite') from exc
    if not math.isfinite(perplexity):
        raise RuntimeError('evaluation perplexity is not finite')
    return {'nll': nll, 'perplexity': perplexity, 'valid_targets': valid_targets,
            'window_count': len(windows), 'batch_count': batch_count, 'batch_sizes': batch_sizes,
            'wall_s': time.monotonic() - started}


def _cpu_tree(value):
    """Detach/clone checkpoint tensors, keeping only native primitive containers."""
    if isinstance(value, Tensor):
        return value.detach().to(device='cpu').clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f'unsupported checkpoint value type: {type(value).__name__}')


def _validate_primitive_tree(value) -> None:
    if isinstance(value, Tensor):
        return
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise ValueError('checkpoint dictionary keys must be strings or integers')
        for item in value.values():
            _validate_primitive_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_primitive_tree(item)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError(f'unsupported checkpoint value type: {type(value).__name__}')
    elif type(value) is float and not math.isfinite(value):
        raise ValueError('checkpoint primitive numbers must be finite')


def _checked_generator_state(value, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.device.type != 'cpu' or value.dtype != torch.uint8:
        raise ValueError(f'{name} must be a CPU byte tensor')
    try:
        torch.Generator(device='cpu').set_state(value)
    except RuntimeError as exc:
        raise ValueError(f'invalid {name}') from exc
    return value.clone()


class BatchStream:
    """Ongoing named CPU shuffle stream with lazy epochs and actual final B."""

    def __init__(self, tokenized_articles: Sequence[Sequence[int]], blocks: Sequence,
                 *, batch_size: int, base_seed: int):
        if type(batch_size) is not int or batch_size <= 0 or not blocks:
            raise ValueError('batch stream requires blocks and positive integer batch_size')
        self.tokenized_articles = tuple(tuple(tokens) for tokens in tokenized_articles)
        self.blocks = tuple(blocks)
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.seed = named_seed(base_seed, 'data')
        self.generator = torch.Generator(device='cpu').manual_seed(self.seed)
        # Canonical length-prefixed uint16 byte-token arrays, then block bounds.
        # One article buffer at a time avoids a corpus-sized JSON allocation.
        digest = hashlib.sha256(b'oar.byte_batch_manifest.v1\0')
        digest.update(len(self.tokenized_articles).to_bytes(8, 'little'))
        for tokens in self.tokenized_articles:
            encoded = array('H', tokens)
            if sys.byteorder != 'little':
                encoded.byteswap()
            digest.update(len(tokens).to_bytes(8, 'little'))
            digest.update(encoded.tobytes())
        records = [[block.article_index, block.target_start, block.target_end, block.input_start]
                   for block in self.blocks]
        digest.update(json.dumps(records, separators=(',', ':')).encode('ascii'))
        self.manifest_sha256 = digest.hexdigest()
        self.order = torch.empty(0, dtype=torch.int64, device='cpu')
        self.cursor = 0
        self.epoch = 0

    def _metadata(self) -> dict:
        return {'schema_version': 1, 'kind': 'oar.batch_stream',
                'manifest_sha256': self.manifest_sha256, 'batch_size': self.batch_size,
                'base_seed': self.base_seed, 'seed': self.seed, 'block_count': len(self.blocks)}

    def next_batch(self, device='cpu') -> dict:
        from .sampling_data import collate_blocks

        if self.cursor == len(self.order):
            self.order = torch.randperm(len(self.blocks), generator=self.generator, device='cpu')
            self.cursor = 0
            self.epoch += 1
        stop = min(self.cursor + self.batch_size, len(self.order))
        selected = [self.blocks[i] for i in self.order[self.cursor:stop].tolist()]
        batch = collate_blocks(self.tokenized_articles, selected, device=device)
        self.cursor = stop
        return batch

    def state_dict(self) -> dict:
        return {**self._metadata(), 'order': self.order.clone(), 'cursor': self.cursor,
                'epoch': self.epoch, 'generator_state': self.generator.get_state().clone()}

    def _validated_state(self, state: dict) -> dict:
        if not isinstance(state, dict) or any(state.get(key) != value for key, value in self._metadata().items()):
            raise ValueError('batch stream metadata/manifest binding mismatch')
        order, cursor, epoch = state.get('order'), state.get('cursor'), state.get('epoch')
        if (not isinstance(order, Tensor) or order.device.type != 'cpu' or order.dtype != torch.int64
                or order.ndim != 1 or len(order) not in (0, len(self.blocks))):
            raise ValueError('batch stream order must be a CPU int64 permutation')
        if len(order) and not torch.equal(order.sort().values, torch.arange(len(self.blocks), dtype=torch.int64, device='cpu')):
            raise ValueError('batch stream order must be a permutation')
        if type(cursor) is not int or not 0 <= cursor <= len(order):
            raise ValueError('batch stream cursor is out of range')
        if len(order) and cursor == 0:
            raise ValueError('drawn epoch cursor must follow a completed nonempty batch')
        if cursor != len(order) and cursor % self.batch_size:
            raise ValueError('batch stream cursor must be a possible batch boundary')
        if type(epoch) is not int or epoch < 0 or (epoch == 0) != (len(order) == 0):
            raise ValueError('batch stream epoch/order mismatch')
        rng = _checked_generator_state(state.get('generator_state'), 'data generator_state')
        return {'order': order.clone(), 'cursor': cursor, 'epoch': epoch, 'generator_state': rng}

    def load_state_dict(self, state: dict) -> None:
        checked = self._validated_state(state)
        self.order = checked['order']
        self.cursor, self.epoch = checked['cursor'], checked['epoch']
        self.generator.set_state(checked['generator_state'])


class ChargedClock:
    """Current worker active time plus prior charged attempts, excluding downtime."""

    def __init__(self, launch_perf_s: float, *, prior_elapsed_s: float = 0., clock=time.perf_counter):
        if (type(launch_perf_s) not in (int, float) or not math.isfinite(launch_perf_s)
                or launch_perf_s < 0 or type(prior_elapsed_s) not in (int, float)
                or not math.isfinite(prior_elapsed_s) or prior_elapsed_s < 0):
            raise ValueError('clock launch and prior elapsed must be finite and nonnegative')
        self.launch_perf_s = float(launch_perf_s)
        self.prior_elapsed_s = float(prior_elapsed_s)
        self.clock = clock
        self._last_observed = self.launch_perf_s
        self.elapsed()

    def elapsed(self) -> float:
        now = self.clock()
        if (type(now) not in (int, float) or not math.isfinite(now)
                or now < self.launch_perf_s or now < self._last_observed):
            raise ValueError('clock must share the launch monotonic host domain without regression')
        self._last_observed = now
        elapsed = self.prior_elapsed_s + (now - self.launch_perf_s)
        if not math.isfinite(elapsed):
            raise ValueError('clock cumulative elapsed must be finite')
        return elapsed

    def state_dict(self) -> dict:
        return {'launch_perf_s': self.launch_perf_s, 'prior_elapsed_s': self.prior_elapsed_s,
                'elapsed_s': self.elapsed()}


def _require_all_trainable(model: ByteLM) -> None:
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('training engine requires all unique ByteLM parameters trainable')


class TrainingState:
    """Fully trainable ByteLM updates; caller owns synchronized commits.

    Every unique parameter receives a gradient tensor on every update, including
    zero gradients. Parameter freezing is supported by the operator separately,
    not by this fixed training/recovery engine.
    """

    def __init__(self, model: ByteLM, optimizer: torch.optim.AdamW, stream: BatchStream):
        if type(optimizer) is not torch.optim.AdamW:
            raise ValueError('training requires native torch.optim.AdamW')
        _require_all_trainable(model)
        if optimizer.state:
            raise ValueError('fresh training engine requires empty optimizer state; use restore')
        if any(p.dtype != torch.float32 or p.device.type not in ('cpu', 'mps') for p in model.parameters()):
            raise ValueError('training requires FP32 CPU/MPS parameters')
        optimized = [p for group in optimizer.param_groups for p in group['params']]
        eligible = [f'blocks.{i}.{suffix}' for i in range(model.config.blocks)
                    for suffix in ('ffn_in', 'ffn_out')] if model.sampling_states else []
        validate_sampled_parameters(model, eligible, optimized)
        self.model, self.optimizer, self.stream = model, optimizer, stream
        self._parameters = tuple(model.parameters())
        self.logical_updates = self.valid_tokens = self.physical_tokens = 0
        self._failed = False
        static_keys = ('betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach',
                       'capturable', 'differentiable', 'fused', 'decoupled_weight_decay')
        parameter_names = {id(p): name for name, p in model.named_parameters()}
        self._bindings = {
            'model_config': asdict(model.config), 'base_seed': model.base_seed, 'sampler': model.sampler,
            'all_parameters_trainable': True,
            'sampling_rhos': {name: state.rho for name, state in model.sampling_states.items()},
            'parameter_layout': [[name, list(p.shape), str(p.dtype)] for name, p in model.named_parameters()],
            'device_type': next(model.parameters()).device.type,
            'optimizer_static': [{key: group.get(key) for key in static_keys} for group in optimizer.param_groups],
            'optimizer_parameter_names': [[parameter_names[id(p)] for p in group['params']]
                                          for group in optimizer.param_groups],
            'stream': stream._metadata(),
        }

    def step(self, lr: float) -> dict:
        if self._failed:
            raise RuntimeError('failed training state requires whole-state restore before another step')
        if any(not parameter.requires_grad for parameter in self._parameters):
            raise ValueError('training engine requires all parameters trainable')
        if any(torch.is_autocast_enabled(device) for device in ('cpu', 'cuda', 'mps')):
            raise RuntimeError('FP32 training primitive rejects autocast/AMP')
        if type(lr) not in (int, float) or not math.isfinite(lr) or lr < 0:
            raise ValueError('learning rate must be finite and nonnegative')
        self._failed = True
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = self.stream.next_batch(device=next(self.model.parameters()).device)
        valid, physical = batch['valid_targets'], batch['physical_rows']
        if type(valid) is not int or valid <= 0:
            raise ValueError('training batch requires positive valid targets')
        logits = self.model(batch['input_ids'])
        loss = F.cross_entropy(logits.reshape(-1, self.model.config.vocab_size),
                               batch['target_ids'].reshape(-1), reduction='sum', ignore_index=-100) / valid
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self._parameters, 1., error_if_nonfinite=True)
        # Native clipping has already checked/synchronized the backward norm.
        # Validate the loss before AdamW without a separate forward/backward
        # host barrier, which would unnecessarily burden the native baseline.
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('training loss must be finite')
        self.optimizer.step()
        # The controller synchronizes once per block before transferring/logging
        # these detached scalars; extracting them here would fence every update.
        result = {'loss': loss.detach(), 'norm': norm.detach(), 'valid_targets': valid,
                  'physical_rows': physical, 'logical_update': self.logical_updates + 1}
        self.logical_updates += 1
        self.valid_tokens += valid
        self.physical_tokens += physical
        self._failed = False
        return result

    def snapshot(self) -> dict:
        if self._failed:
            raise RuntimeError('cannot snapshot failed or uncommitted training step')
        _require_all_trainable(self.model)
        device = next(self.model.parameters()).device
        return {
            'schema_version': 1, 'kind': 'oar.training_state', 'bindings': _cpu_tree(self._bindings),
            'model': _cpu_tree(self.model.state_dict()), 'optimizer': _cpu_tree(self.optimizer.state_dict()),
            'stream': self.stream.state_dict(), 'sampler': self.model.sampling_state_dict(),
            'init_generator_state': self.model.init_generator.get_state().clone(),
            'cpu_rng_state': torch.get_rng_state().clone(),
            'mps_rng_state': torch.mps.get_rng_state(device).cpu().clone() if device.type == 'mps' else None,
            'logical_updates': self.logical_updates, 'valid_tokens': self.valid_tokens,
            'physical_tokens': self.physical_tokens, 'model_training': self.model.training,
        }

    def restore(self, snapshot: dict) -> None:
        # Validate compatibility and all CPU stream metadata before any mutation.
        _require_all_trainable(self.model)
        _validate_primitive_tree(snapshot)
        if (not isinstance(snapshot, dict) or type(snapshot.get('schema_version')) is not int
                or snapshot['schema_version'] != 1 or snapshot.get('kind') != 'oar.training_state'):
            raise ValueError('training snapshot schema mismatch')
        if snapshot.get('bindings') != self._bindings:
            raise ValueError('training snapshot metadata binding mismatch')
        self.stream._validated_state(snapshot.get('stream'))
        current = self.model.state_dict()
        saved = snapshot.get('model')
        if not isinstance(saved, dict) or set(saved) != set(current):
            raise ValueError('training snapshot model keys mismatch')
        for key, value in saved.items():
            if (not isinstance(value, Tensor) or value.device.type != 'cpu'
                    or value.shape != current[key].shape or value.dtype != current[key].dtype):
                raise ValueError(f'training snapshot model tensor mismatch: {key}')
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'training snapshot model tensor must be finite: {key}')
        if not torch.equal(saved['head.weight'], saved['token_embedding.weight']):
            raise ValueError('training snapshot tied model weights disagree')
        sampler = snapshot.get('sampler')
        if not isinstance(sampler, dict) or set(sampler) != set(self.model.sampling_states):
            raise ValueError('training snapshot sampler IDs mismatch')
        for name, state in self.model.sampling_states.items():
            probe = SamplingState(state.base_seed, name, state.sampler, state.rho)
            probe.load_state_dict(sampler[name])
        init_rng = _checked_generator_state(snapshot.get('init_generator_state'), 'init generator_state')
        cpu_rng = _checked_generator_state(snapshot.get('cpu_rng_state'), 'CPU rng_state')
        device = next(self.model.parameters()).device
        mps_rng = snapshot.get('mps_rng_state')
        if device.type == 'mps':
            if not isinstance(mps_rng, Tensor) or mps_rng.device.type != 'cpu' or mps_rng.dtype != torch.uint8:
                raise ValueError('MPS rng_state must be a CPU byte tensor')
        elif mps_rng is not None:
            raise ValueError('CPU snapshot must not contain MPS RNG state')
        counters = [snapshot.get(key) for key in ('logical_updates', 'valid_tokens', 'physical_tokens')]
        if (any(type(value) is not int or value < 0 for value in counters)
                or not counters[0] <= counters[1] <= counters[2] or type(snapshot.get('model_training')) is not bool):
            raise ValueError('training snapshot counters/mode mismatch')
        optimizer_state = snapshot.get('optimizer')
        if (not isinstance(optimizer_state, dict) or set(optimizer_state) != {'state', 'param_groups'}
                or not isinstance(optimizer_state['state'], dict)
                or not isinstance(optimizer_state['param_groups'], list)
                or len(optimizer_state['param_groups']) != len(self.optimizer.param_groups)):
            raise ValueError('training snapshot optimizer structure mismatch')
        native_groups = self.optimizer.state_dict()['param_groups']
        expected_parameters = {}
        for loaded, actual, static, native in zip(optimizer_state['param_groups'], self.optimizer.param_groups,
                                                 self._bindings['optimizer_static'], native_groups):
            if not isinstance(loaded, dict) or len(loaded.get('params', [])) != len(actual['params']):
                raise ValueError('training snapshot optimizer parameter groups mismatch')
            if loaded['params'] != native['params']:
                raise ValueError('training snapshot optimizer parameter IDs/order mismatch')
            if any(loaded.get(key) != value for key, value in static.items()):
                raise ValueError('training snapshot optimizer static metadata mismatch')
            lr = loaded.get('lr')
            if type(lr) not in (int, float) or not math.isfinite(lr) or lr < 0:
                raise ValueError('training snapshot optimizer learning rate mismatch')
            for parameter_id, parameter in zip(loaded['params'], actual['params']):
                if type(parameter_id) is not int:
                    raise ValueError('training snapshot optimizer parameter IDs must be integers')
                expected_parameters[parameter_id] = (parameter, loaded['amsgrad'], loaded['fused'])
        moments = optimizer_state['state']
        if counters[0] == 0:
            if moments:
                raise ValueError('zero-update snapshot optimizer state must be empty')
        else:
            if set(moments) != set(expected_parameters):
                raise ValueError('training snapshot optimizer must contain every parameter state exactly once')
            for parameter_id, (parameter, amsgrad, fused) in expected_parameters.items():
                state = moments[parameter_id]
                required = {'step', 'exp_avg', 'exp_avg_sq'}
                if amsgrad:
                    required.add('max_exp_avg_sq')
                if not isinstance(state, dict) or set(state) != required:
                    raise ValueError('training snapshot optimizer moment fields mismatch')
                step = state['step']
                step_dtypes = (torch.float32,) if fused else (torch.float32, torch.float64)
                if (not isinstance(step, Tensor) or step.device.type != 'cpu'
                        or step.ndim != 0 or step.dtype not in step_dtypes):
                    raise ValueError('training snapshot optimizer step must be a native CPU scalar tensor')
                step_number = step.item()
                if (not math.isfinite(step_number) or step_number < 0
                        or not step_number.is_integer() or step_number != counters[0]):
                    raise ValueError('training snapshot optimizer step must equal logical_updates')
                for name in required - {'step'}:
                    moment = state[name]
                    if (not isinstance(moment, Tensor) or moment.device.type != 'cpu'
                            or moment.shape != parameter.shape or moment.dtype != parameter.dtype
                            or not bool(torch.isfinite(moment).all())):
                        raise ValueError(f'training snapshot optimizer moment tensor mismatch: {name}')
                    if name != 'exp_avg' and not bool((moment >= 0).all()):
                        raise ValueError(f'training snapshot optimizer second moment must be nonnegative: {name}')
        self._failed = True
        self.model.load_state_dict(saved)
        # Native load maps moments to each Parameter's device and preserves step policy.
        self.optimizer.load_state_dict(_cpu_tree(optimizer_state))
        self.stream.load_state_dict(snapshot['stream'])
        self.model.load_sampling_state_dict(sampler)
        self.model.init_generator.set_state(init_rng)
        torch.set_rng_state(cpu_rng)
        if device.type == 'mps':
            torch.mps.set_rng_state(mps_rng, device)
        self.logical_updates, self.valid_tokens, self.physical_tokens = counters
        self.model.train(snapshot['model_training'])
        self.optimizer.zero_grad(set_to_none=True)
        self._failed = False


def _validate_checkpoint_payload(payload: dict) -> None:
    _validate_primitive_tree(payload)
    required = {'schema_version', 'bindings', 'training_state', 'clock', 't0_s', 'scheduler_rule'}
    if not isinstance(payload, dict) or not required <= set(payload):
        raise ValueError('checkpoint schema is missing required keys')
    if type(payload['schema_version']) is not int or payload['schema_version'] != 1:
        raise ValueError('checkpoint schema_version mismatch')
    if any(not isinstance(payload[key], dict) for key in ('bindings', 'training_state', 'clock', 'scheduler_rule')):
        raise ValueError('checkpoint bindings/state/clock/scheduler must be dictionaries')
    state = payload['training_state']
    if type(state.get('schema_version')) is not int or state['schema_version'] != 1 or state.get('kind') != 'oar.training_state':
        raise ValueError('checkpoint training state schema mismatch')
    t0 = payload['t0_s']
    if t0 is not None and (type(t0) not in (int, float) or not math.isfinite(t0) or t0 < 0):
        raise ValueError('checkpoint t0_s must be finite nonnegative or None')
    for key in ('launch_perf_s', 'prior_elapsed_s', 'elapsed_s'):
        value = payload['clock'].get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('checkpoint clock metadata must be finite and nonnegative')


def _file_sha256(file) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: file.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path, payload: dict) -> dict:
    """Fsync and atomically publish a new immutable checkpoint destination.

    The exclusive lock coordinates writers using this helper. The run controller
    supplies unique names and excludes unrelated writers from its directory.
    """
    started = time.monotonic()
    _validate_checkpoint_payload(payload)
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f'checkpoint already exists: {path}')
    lock = path.with_name(f'.{path.name}.lock')
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    owns_lock = owns_temporary = False
    try:
        with lock.open('xb'):
            owns_lock = True
        if os.path.lexists(path):
            raise FileExistsError(f'checkpoint already exists: {path}')
        with temporary.open('xb') as file:
            owns_temporary = True
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        with temporary.open('rb') as file:
            sha256 = _file_sha256(file)
        size_bytes = temporary.stat().st_size
        if os.path.lexists(path):
            raise FileExistsError(f'checkpoint already exists: {path}')
        os.rename(temporary, path)
        owns_temporary = False
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {'sha256': sha256, 'size_bytes': size_bytes, 'wall_s': time.monotonic() - started}
    finally:
        if owns_temporary:
            with suppress(FileNotFoundError):
                temporary.unlink()
        if owns_lock:
            with suppress(FileNotFoundError):
                lock.unlink()


def load_checkpoint(path, expected_sha256: str, *, expected_bindings: dict) -> dict:
    """Verify exact bytes/schema/bindings before exposing state to any caller."""
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(char not in '0123456789abcdef' for char in expected_sha256)):
        raise ValueError('expected SHA256 must be a lowercase 64-character digest')
    with Path(path).open('rb') as file:
        if _file_sha256(file) != expected_sha256:
            raise ValueError('checkpoint SHA256 integrity mismatch')
        file.seek(0)
        payload = torch.load(file, weights_only=True, map_location='cpu')
    _validate_checkpoint_payload(payload)
    if payload['bindings'] != expected_bindings:
        raise ValueError('checkpoint bindings mismatch')
    return payload


def time_budget_lr(elapsed, *, t0, cutoff, peak):
    """Preserved run t0, five-percent warmup, then cosine to one tenth peak."""
    if not all(math.isfinite(v) for v in (elapsed, t0, cutoff, peak)) or not 0 <= t0 < cutoff or peak <= 0:
        raise ValueError('invalid charged-time scheduler inputs')
    u = min(1., max(0., (elapsed-t0)/(cutoff-t0)))
    return peak*u/.05 if u <= .05 else peak*(.1+.9*(1+math.cos(math.pi*(u-.05)/.95))/2)


class ModelFixture:
    """Complete actual-data model step; also used for paired native qualification."""

    def __init__(self, data, *, base_seed, variant='single', sampler=None, rho=.25,
                 batch_size=16, device='mps', model_config=None, synchronize_each_update=False):
        self.model = ByteLM(model_config or ModelConfig(), base_seed=base_seed, sampler=sampler, rho=rho).to(device)
        self.opt = make_optimizer(self.model, variant)
        self.stream = BatchStream(data['training_tokens'], data['training_blocks'], batch_size=batch_size, base_seed=base_seed)
        self.training = TrainingState(self.model, self.opt, self.stream)
        self.p = dict(self.model.named_parameters())
        self.states = self.model.sampling_states
        self._timing_initial_sampling = self.model.sampling_state_dict()
        self.init_generator, self.data_generator = self.model.init_generator, self.stream.generator
        self.init_seed, self.data_seed = self.model.init_seed, self.stream.seed
        self.sampling_setup_s = self.model.sampling_setup_s
        self.lr = .001
        self.synchronize_each_update = synchronize_each_update

    @property
    def updates(self):
        return self.training.logical_updates

    def step(self):
        result = self.training.step(self.lr)
        if self.synchronize_each_update and next(self.model.parameters()).device.type == 'mps':
            torch.mps.synchronize()
        return result

    def snapshot(self):
        if self.states:
            raise ValueError('paired timing common history must be native')
        return self.training.snapshot()

    def restore(self, snapshot):
        # This fixture deliberately compares different gradient estimators from
        # one native history. Check every shared binding before installing only
        # this arm's prospectively named initial sampler stream. Production
        # checkpoint restore above never rewrites compatibility metadata.
        expected = {**self.training._bindings, 'sampler': None, 'sampling_rhos': {}}
        if snapshot.get('bindings') != expected:
            raise ValueError('paired timing native history binding mismatch')
        self.training.restore({**snapshot, 'bindings': self.training._bindings,
                               'sampler': self._timing_initial_sampling})


def load_worker_data(path, expected_sha256, *, evaluation_path=None, evaluation_sha256=None, phase='development'):
    """Read only exact prepared bytes; validation loss access requires its phase."""
    from .sampling_data import TargetBlock
    def read(filename, expected):
        with Path(filename).open('rb') as stream:
            if _file_sha256(stream) != expected:
                raise ValueError('prepared data SHA256 mismatch')
            stream.seek(0)
            value = torch.load(stream, weights_only=True, map_location='cpu')
        if value.get('schema_version') != 1:
            raise ValueError('prepared data schema mismatch')
        return value
    training = read(path, expected_sha256)
    if phase == 'development':
        evaluation, prefix = training, 'development'
    elif phase == 'confirmation':
        if evaluation_path is None or evaluation_sha256 is None:
            raise ValueError('confirmation requires explicitly frozen validation data')
        evaluation, prefix = read(evaluation_path, evaluation_sha256), 'validation'
    else:
        raise ValueError('unknown protected evaluation phase')
    return {'training_tokens': training['fitting_tokens'],
            'training_blocks': [TargetBlock(**v) for v in training['training_blocks']],
            'evaluation_tokens': evaluation[prefix+'_tokens'],
            'evaluation_full': [TargetBlock(**v) for v in evaluation[prefix+'_full']],
            'evaluation_monitor': [TargetBlock(**v) for v in evaluation[prefix+'_monitor']]}


def train_budget(state, data, request, charged_clock):
    """One fixed-budget trajectory; all logging/evaluation/checkpoint work is paid.

    The launcher owns fresh processes, protected split authority, source bindings
    and the cumulative active-attempt ledger. This loop never retries a draw.
    """
    horizon, interval, budget = (request[k] for k in ('H', 'Delta', 'T_budget'))
    if (not all(math.isfinite(v) and v > 0 for v in (horizon, interval, budget)) or budget <= horizon
            or request['block_updates'] not in (1, 8)):
        raise ValueError('invalid frozen training budget')
    run_dir, checkpoint_dir = Path(request['run_dir']), Path(request['checkpoint_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    event_path, receipt_path = run_dir/'events.jsonl', run_dir/'checkpoint_receipts.jsonl'
    with event_path.open('x'):
        pass
    with receipt_path.open('x'):
        pass
    device = next(state.model.parameters()).device
    sync = torch.mps.synchronize if device.type == 'mps' else lambda: None
    steps, monitors, missed, checkpoints = [], [], [], []
    history = request.get('checkpoint_history', [])
    if request.get('checkpoint_history_sha256') is not None:
        encoded = json.dumps(history, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
        if hashlib.sha256(encoded).hexdigest() != request['checkpoint_history_sha256']:
            raise ValueError('incoming checkpoint history hash mismatch')
    for row in history:
        path = Path(row['path'])
        if path.is_symlink() or path.resolve().parent != checkpoint_dir.resolve() or row['kind'] not in ('initial', 'periodic', 'final'):
            raise ValueError('incoming checkpoint history is not owned by this logical trial')
        if not row.get('pruned'):
            with path.open('rb') as stream:
                if _file_sha256(stream) != row['sha256']:
                    raise ValueError('incoming live checkpoint hash mismatch')
        checkpoints.append(dict(row))
    if len([r for r in checkpoints if not r.get('pruned') and r['kind'] != 'final']) > 2:
        raise ValueError('incoming checkpoint history exceeds two live resumes')
    secondary_history = [dict(r) for r in request.get('secondary_history', [])]
    for row in secondary_history:
        path = Path(row['path'])
        if path.is_symlink() or path.resolve().parent != checkpoint_dir.resolve() or row['kind'] != 'secondary':
            raise ValueError('secondary history is not owned by this logical trial')
        if not row.get('pruned'):
            with path.open('rb') as stream:
                if _file_sha256(stream) != row['sha256']:
                    raise ValueError('secondary checkpoint history hash mismatch')
    secondary = None
    secondary_update = request.get('secondary_update')
    if secondary_update is not None and (type(secondary_update) is not int or secondary_update <= 0):
        raise ValueError('secondary update must be a fixed positive integer')
    schedule = {k: request[k] for k in ('H', 'Delta', 'T_budget', 'block_updates', 'peak_lr', 'overshoot_limit_s')}
    initial_full = initial_monitor = None
    t0, next_index = None, 1
    restart = None
    def append(path, row):
        with path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
    def event(kind, **row):
        append(event_path, {'kind': kind, **row})
    def measure_evaluation(windows, kind, **extra):
        result = evaluate(state.model, data['evaluation_tokens'], windows, batch_size=16)
        sync()
        row = {**result, 'completed_s': charged_clock.elapsed(), **extra}
        if not math.isfinite(row['nll']):
            raise RuntimeError('evaluation produced nonfinite quality evidence')
        event(kind, **row)
        return row
    def checkpoint(kind):
        sync()
        path = checkpoint_dir/f'{kind}-u{state.logical_updates:08d}-{uuid.uuid4().hex[:12]}.pt'
        payload = {'schema_version': 1, 'bindings': request['bindings'], 'training_state': state.snapshot(),
                   'clock': charged_clock.state_dict(), 't0_s': t0, 'scheduler_rule': schedule,
                   'initial_full': initial_full, 'initial_monitor': initial_monitor,
                   'monitors': monitors, 'missed_deadlines': missed, 'next_deadline_index': next_index,
                   'optimization_frozen': kind in ('final', 'secondary'),
                   'clock_implementation': request.get('clock_implementation'),
                   'secondary_checkpoint': secondary}
        receipt = {'path': str(path), **save_checkpoint(path, payload), 'kind': kind,
                   'logical_updates': state.logical_updates, 'completed_s': charged_clock.elapsed()}
        append(receipt_path, receipt)
        # The receipt is durable before any declared ephemeral-resume pruning.
        with receipt_path.open('a') as stream:
            stream.flush()
            os.fsync(stream.fileno())
        if kind == 'secondary':
            secondary_history.append(receipt)
            event('secondary_checkpoint', checkpoint=receipt)
            return receipt
        checkpoints.append(receipt)
        event('checkpoint', checkpoint=receipt)
        live = [r for r in checkpoints if r['kind'] != 'final' and not r.get('pruned')]
        for old in (live if kind == 'final' else live[:-2]):
            Path(old['path']).unlink()
            old['pruned'] = True
            event('checkpoint_pruned', path=old['path'], sha256=old['sha256'],
                  policy='two live resume files while training; final alone after its durable receipt; every receipt retained')
        return receipt
    def skip_past(now, reason):
        nonlocal next_index
        while next_index*interval <= now and next_index*interval < horizon:
            row = {'deadline_s': next_index*interval, 'observed_s': now, 'reason': reason}
            missed.append(row)
            event('missed_deadline', **row)
            next_index += 1
    resumed = request.get('resume')
    if resumed:
        payload = load_checkpoint(resumed['path'], resumed['sha256'], expected_bindings=request['bindings'])
        if payload.get('optimization_frozen'):
            raise ValueError('final/frozen checkpoint cannot resume optimization')
        if payload.get('clock_implementation') != request.get('clock_implementation'):
            raise ValueError('checkpoint host clock implementation mismatch')
        if payload['scheduler_rule'] != schedule or payload['t0_s'] is None:
            raise ValueError('resume cannot change the original scheduler or restart warmup')
        if charged_clock.prior_elapsed_s < payload['clock']['elapsed_s']:
            raise ValueError('restart ledger omits previously charged work')
        if (type(payload.get('next_deadline_index')) is not int or payload['next_deadline_index']<1
                or not isinstance(payload.get('monitors'),list) or not isinstance(payload.get('missed_deadlines'),list)):
            raise ValueError('resume controller/deadline metadata invalid')
        previous_completion = -1.
        for observation in [payload.get('initial_full'),payload.get('initial_monitor'),*payload['monitors']]:
            if (not isinstance(observation,dict) or type(observation.get('nll')) not in (int,float)
                    or not math.isfinite(observation['nll']) or type(observation.get('completed_s')) not in (int,float)
                    or not previous_completion <= observation['completed_s'] <= payload['clock']['elapsed_s']):
                raise ValueError('resume controller evaluation/clock history invalid')
            previous_completion = observation['completed_s']
        if not payload['initial_monitor']['completed_s'] <= payload['t0_s'] < horizon:
            raise ValueError('resume controller original t0 invalid')
        saved_secondary=payload.get('secondary_checkpoint')
        if saved_secondary is not None and not any(r['path']==saved_secondary['path'] and r['sha256']==saved_secondary['sha256'] and not r.get('pruned') for r in secondary_history):
            raise ValueError('resumed secondary pointer has no own live receipt')
        if secondary_update is not None and payload['training_state']['logical_updates'] >= secondary_update and saved_secondary is None:
            raise ValueError('resume would lose a previously required equal-step capture')
        state.restore(payload['training_state'])
        t0, initial_full, initial_monitor = payload['t0_s'], payload['initial_full'], payload['initial_monitor']
        monitors, missed = list(payload['monitors']), list(payload['missed_deadlines'])
        next_index = payload['next_deadline_index']
        secondary = payload.get('secondary_checkpoint')
        if secondary is not None and not any(r['path']==secondary['path'] and r['sha256']==secondary['sha256'] and not r.get('pruned') for r in secondary_history):
            raise ValueError('resumed secondary pointer has no own live receipt')
        # A secondary captured after the selected recovery boundary belongs to
        # discarded work. Retire only those proved own files; capture again if
        # the resumed canonical trajectory reaches the declared update.
        for old in secondary_history:
            if not old.get('pruned') and (secondary is None or old['path'] != secondary['path']):
                Path(old['path']).unlink()
                old['pruned'] = True
                event('secondary_checkpoint_pruned', path=old['path'], sha256=old['sha256'],
                      reason='discarded work after explicitly selected recovery boundary')
        if secondary_update is not None and state.logical_updates >= secondary_update and secondary is None:
            raise ValueError('resume would lose a previously required equal-step capture')
        restart = {'checkpoint': resumed, 'prior_charged_s': charged_clock.prior_elapsed_s,
                   'loading_completed_s': charged_clock.elapsed(), 'original_t0_s': t0}
        event('restart', **restart)
    else:
        initial_full = measure_evaluation(data['evaluation_full'], 'initial_full')
        initial_monitor = measure_evaluation(data['evaluation_monitor'], 'initial_monitor')
        sync()
        t0 = charged_clock.elapsed()
        if t0 >= horizon:
            raise RuntimeError('initialization and initial evaluations exhausted the optimization cutoff')
        # This initial committed boundary belongs to the first block; its IO is
        # charged, and that block holds the schedule's initial zero LR.
        checkpoint('initial')
    if not 0 <= t0 < horizon:
        raise ValueError('invalid preserved training t0')
    fresh_first = not resumed
    skip_past(charged_clock.elapsed(), 'initialization_or_restart')
    cutoff_completion = None
    while True:
        now = charged_clock.elapsed()
        if now >= horizon:
            sync()
            cutoff_completion = charged_clock.elapsed()
            break
        block_start = t0 if fresh_first else now
        fresh_first = False
        lr = time_budget_lr(block_start, t0=t0, cutoff=horizon, peak=request['peak_lr'])
        event('training_block_start', started_s=block_start, lr=lr, t0_s=t0,
              logical_updates_before=state.logical_updates, planned_updates=request['block_updates'])
        raw = []
        try:
            for _ in range(request['block_updates']):
                raw.append(state.step(lr))
                if secondary_update is not None and state.logical_updates == secondary_update and secondary is None:
                    secondary = checkpoint('secondary')
            sync()
            # One small host transfer after the synchronized block, never after
            # each optimizer step. Device scalar references are then released.
            scalars = torch.stack([torch.stack((r['loss'], r['norm'])) for r in raw]).to('cpu')
        except BaseException as exc:
            state._failed = True
            failure = {'exception': type(exc).__name__, 'reason': str(exc), 'lr': lr,
                       'charged_s': charged_clock.elapsed(), 't0_s': t0,
                       'logical_updates': state.logical_updates,
                       'host_completed_updates_in_block': len(raw),
                       'successful_updates_in_block': len(raw) if device.type == 'cpu' else None,
                       'device_commit_confirmed': False, 'valid_tokens_host': state.valid_tokens,
                       'physical_tokens_host': state.physical_tokens,
                       'last_committed_checkpoint': checkpoints[-1] if checkpoints else resumed}
            try:
                with (run_dir/'failure.json').open('x') as stream:
                    json.dump(failure, stream, allow_nan=False, indent=2)
                event('interruption' if isinstance(exc, KeyboardInterrupt) else 'training_failure', **failure)
            except OSError:
                # The launcher still retains the original exception/active time
                # if the same filesystem failure prevents this extra record.
                pass
            raise
        block_end = charged_clock.elapsed()
        block_steps = [{**{k: v for k, v in row.items() if k not in ('loss', 'norm')},
                        'loss': float(scalars[i, 0]), 'norm': float(scalars[i, 1]), 'lr': lr,
                        'block_start_s': block_start, 'block_end_s': block_end}
                       for i, row in enumerate(raw)]
        steps.extend(block_steps)
        event('training_block', started_s=block_start, completed_s=block_end, steps=block_steps,
              valid_tokens=state.valid_tokens, physical_tokens=state.physical_tokens)
        del raw, scalars
        boundary = charged_clock.elapsed()
        if boundary >= horizon:
            cutoff_completion = boundary
            break
        if next_index*interval < horizon and boundary >= next_index*interval:
            deadline_index = next_index
            next_index += 1
            row = measure_evaluation(data['evaluation_monitor'], 'periodic_monitor',
                                     deadline_s=deadline_index*interval, deadline_index=deadline_index,
                                     logical_updates=state.logical_updates)
            monitors.append(row)
            if charged_clock.elapsed() >= horizon:
                cutoff_completion = charged_clock.elapsed()
                break
            if deadline_index % 5 == 0:
                checkpoint('periodic')
            skip_past(charged_clock.elapsed(), 'monitor_or_checkpoint')
    # Model parameters freeze here. Endpoint work cannot add training updates.
    sync()
    endpoint = measure_evaluation(data['evaluation_full'], 'endpoint_full')
    final = checkpoint('final')
    completion = charged_clock.elapsed()
    overshoot = cutoff_completion-horizon
    reasons = []
    if completion > budget:
        reasons.append('endpoint evaluation/checkpoint exceeded common reserve')
    if overshoot > request['overshoot_limit_s']:
        reasons.append('optimization cutoff overshoot exceeded frozen bound')
    if len(monitors) < 8:
        reasons.append('fewer than eight completed periodic monitors')
    status = 'pass' if not reasons else 'INCONCLUSIVE_MEASUREMENT'
    result = {'status': status, 'reason': '; '.join(reasons) if reasons else 'complete fixed-budget run',
              'id': request['id'], 'phase': request['phase'], 'arm': request.get('arm'),
              'initial_full': initial_full, 'initial_monitor': initial_monitor, 'monitors': monitors,
              'missed_deadlines': missed, 'endpoint_full': endpoint, 't0_s': t0, 'steps': steps,
              'cutoff_completion_s': cutoff_completion, 'cutoff_overshoot_s': overshoot,
              'completion_s': completion, 'unused_reserve_s': budget-completion,
              'logical_updates': state.logical_updates, 'valid_tokens': state.valid_tokens,
              'physical_tokens': state.physical_tokens, 'final_checkpoint': final,
              'checkpoint_manifest': checkpoints, 'restart': restart,
              'secondary_checkpoint': secondary,
              'secondary_status': 'captured' if secondary else ('unavailable_not_reached' if secondary_update else 'not_requested'),
              'resolved_seeds': {**state.model.resolved_seeds, 'data': state.stream.seed},
              'initial_parameters_sha256': state.model.initial_parameters_sha256,
              'sampling_final_state_hashes': {name: hashlib.sha256(bytes(s.generator.get_state().tolist())).hexdigest()
                                              for name, s in state.model.sampling_states.items()}}
    with (run_dir/'result.json').open('x') as stream:
        json.dump(result, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def run_training_worker(request):
    """Fresh-worker entry; launch time precedes Torch import in the controller."""
    implementation = time.get_clock_info('perf_counter').implementation
    if request['clock_implementation'] != implementation:
        raise ValueError('parent and worker must use the same host monotonic clock')
    charged = ChargedClock(request['launch_perf_s'], prior_elapsed_s=request.get('prior_elapsed_s', 0.))
    arm = request['arm']
    if (arm.get('precision', 'fp32') != 'fp32' or arm['sampler'] not in (None, 'importance', 'uniform')
            or arm['batch'] not in (4, 8, 16, 32)
            or (arm['sampler'] is not None and (arm['batch'] != 16 or arm['rho'] not in (.125, .25, .5)))):
        raise ValueError('worker arm differs from the frozen FP32 byte-model protocol')
    data = load_worker_data(request['data_path'], request['data_sha256'],
                            evaluation_path=request.get('evaluation_path'),
                            evaluation_sha256=request.get('evaluation_sha256'), phase=request['phase'])
    fixture = ModelFixture(data, base_seed=request['seed'], variant=request['variant'],
                           sampler=arm['sampler'], rho=arm.get('rho') or .25, batch_size=arm['batch'])
    return train_budget(fixture.training, data, request, charged)


class NativePrecisionUnsupported(RuntimeError):
    """Explicit installed native autocast/scaler capability failure."""


class NativePrecisionState(TrainingState):
    """Isolated dense native autocast control, retaining FP32 model/Adam states."""

    def __init__(self, model, optimizer, stream, *, precision, scaler=None, scaler_qualification=None):
        if model.sampler is not None or model.sampling_states:
            raise ValueError('precision control requires a native unsampled model')
        if precision not in ('bf16', 'fp16'):
            raise ValueError('unknown native precision')
        device = next(model.parameters()).device.type
        if precision == 'fp16':
            if (device != 'mps' or type(scaler) is not torch.amp.GradScaler or not scaler.is_enabled()
                    or getattr(scaler, '_device', None) != 'mps'
                    or scaler_qualification != {'backend': 'mps', 'native_grad_scaler': True}):
                raise ValueError('FP16 requires qualified installed native MPS GradScaler')
        elif scaler is not None:
            raise ValueError('BF16 control does not use a scaler')
        super().__init__(model, optimizer, stream)
        self.precision, self.scaler = precision, scaler
        self._bindings['native_precision'] = precision
        self.skipped_updates = 0

    def step(self, lr):
        if self._failed:
            raise RuntimeError('failed precision state requires whole-state restore')
        _require_all_trainable(self.model)
        if type(lr) not in (int,float) or not math.isfinite(lr) or lr < 0:
            raise ValueError('learning rate must be finite and nonnegative')
        self._failed = True
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        device = next(self.model.parameters()).device
        batch = self.stream.next_batch(device=device)
        valid, physical = batch['valid_targets'], batch['physical_rows']
        if type(valid) is not int or valid <= 0:
            raise ValueError('precision batch requires positive valid targets')
        dtype = torch.bfloat16 if self.precision == 'bf16' else torch.float16
        with torch.autocast(device_type=device.type, dtype=dtype):
            if not torch.is_autocast_enabled(device.type):
                raise NativePrecisionUnsupported('native autocast is disabled for selected precision/backend')
            logits = self.model(batch['input_ids'])
            loss = F.cross_entropy(logits.reshape(-1,self.model.config.vocab_size),batch['target_ids'].reshape(-1),
                                   reduction='sum',ignore_index=-100)/valid
        if self.scaler is None:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        norm = nn.utils.clip_grad_norm_(self._parameters,1.,error_if_nonfinite=True)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('precision loss must be finite')
        if self.scaler is None:
            self.optimizer.step()
        else:
            before = float(self.scaler.get_scale())
            self.scaler.step(self.optimizer); self.scaler.update()
            if float(self.scaler.get_scale()) < before:
                self.skipped_updates += 1
                raise RuntimeError('native scaler skipped update; trajectory is numerically invalid')
            # A native fused scaler can suppress a step. Verify the native
            # counters before committing this update to the training ledger.
            if any(float(s['step']) != self.logical_updates+1 for s in self.optimizer.state.values()):
                self.skipped_updates += 1
                raise RuntimeError('native scaler optimizer counters show skipped update')
        self.logical_updates += 1; self.valid_tokens += valid; self.physical_tokens += physical
        self._failed = False
        return {'loss':loss.detach(),'norm':norm.detach(),'valid_targets':valid,'physical_rows':physical,
                'logical_update':self.logical_updates}

    def snapshot(self):
        value = super().snapshot()
        value['precision_state'] = {'precision':self.precision,'skipped_updates':self.skipped_updates,
                                    'scaler':_cpu_tree(self.scaler.state_dict()) if self.scaler else None}
        return value

    def restore(self, snapshot):
        precision = snapshot.get('precision_state')
        if (not isinstance(precision,dict) or set(precision) != {'precision','skipped_updates','scaler'}
                or precision['precision'] != self.precision or precision['skipped_updates'] != 0):
            raise ValueError('precision snapshot metadata/skipped update mismatch')
        if self.scaler is None:
            if precision['scaler'] is not None:
                raise ValueError('BF16 precision snapshot cannot contain scaler')
        else:
            scaler = precision['scaler']
            if (not isinstance(scaler,dict) or set(scaler) != set(self.scaler.state_dict())
                    or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in scaler.values())
                    or scaler.get('scale',0) <= 0 or scaler.get('growth_factor',0)<=1
                    or not 0<scaler.get('backoff_factor',0)<1
                    or type(scaler.get('growth_interval')) is not int or scaler['growth_interval']<=0
                    or type(scaler.get('_growth_tracker')) is not int
                    or not 0<=scaler['_growth_tracker']<scaler['growth_interval']):
                raise ValueError('invalid native precision scaler state')
            probe = torch.amp.GradScaler('mps'); probe.load_state_dict(scaler)
        super().restore(snapshot)
        if self.scaler is not None:
            self.scaler.load_state_dict(precision['scaler'])
        self.skipped_updates = 0


def precision_backend_unsupported(exc):
    """Only explicit backend capability failures qualify for FP16 fallback."""
    text = str(exc).lower()
    return isinstance(exc,NativePrecisionUnsupported) or isinstance(exc,(RuntimeError,TypeError,NotImplementedError)) and any(
        fragment in text for fragment in ('bfloat16 is not supported','bfloat16 is not yet supported',
            'does not support bfloat16','autocast is not supported','not implemented for \'bfloat16\'',
            "gradscaler does not support mps","not implemented for mps",
            "not currently implemented for the mps device","not implemented for 'mps'"))


def validate_native_precision_replay(expected, actual, *, device_type):
    """MPS native tolerance only for model and Adam moment floats; rest exact.

    This is the existing prospective FP32 native MPS tolerance, applied before
    any AMP backend outcome. Stored CPU snapshots from CPU execution stay exact.
    """
    if device_type not in ('cpu','mps'):
        raise ValueError('unknown native precision replay device')
    maxima={'model':0.,'moment':0.}
    def compare(a,b,path=()):
        label='/'.join(map(str,path))
        if isinstance(a,Tensor):
            if not isinstance(b,Tensor) or a.shape!=b.shape or a.dtype!=b.dtype or a.device!=b.device:
                raise RuntimeError('native precision replay tensor metadata mismatch: '+label)
            model = len(path)==2 and path[0]=='model'
            moment = (len(path)==4 and path[:2]==('optimizer','state')
                      and path[-1] in ('exp_avg','exp_avg_sq','max_exp_avg_sq'))
            permitted = device_type=='mps' and a.is_floating_point() and (model or moment)
            if (model or moment) and a.is_floating_point():
                if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
                    raise RuntimeError('native precision replay nonfinite model/moment: '+label)
                delta=float((a-b).abs().max()) if a.numel() else 0.
                key='model' if model else 'moment';maxima[key]=max(maxima[key],delta)
            matched=torch.allclose(a,b,rtol=1e-4,atol=1e-6) if permitted else torch.equal(a,b)
            if not matched:
                raise RuntimeError('native precision checkpoint replay mismatch: '+label)
            return
        if isinstance(a,dict):
            if not isinstance(b,dict) or a.keys()!=b.keys():
                raise RuntimeError('native precision replay mapping mismatch: '+label)
            for key in a:compare(a[key],b[key],(*path,key))
        elif isinstance(a,(list,tuple)):
            if type(a) is not type(b) or len(a)!=len(b):
                raise RuntimeError('native precision replay sequence mismatch: '+label)
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,(*path,i))
        elif type(a) is not type(b) or a!=b:
            raise RuntimeError('native precision replay exact metadata mismatch: '+label)
    compare(expected,actual)
    return {'policy':{'device_type':device_type,'rtol':1e-4 if device_type=='mps' else 0.,
                      'atol':1e-6 if device_type=='mps' else 0.,
                      'tolerated_fields':'floating model tensors and Adam exp_avg/exp_avg_sq/max_exp_avg_sq on MPS only',
                      'all_other_fields':'exact; includes RNG/data/sampler, optimizer step, scaler, counters and group hyperparameters'},
            'max_model_absolute_deviation':maxima['model'],'max_adam_moment_absolute_deviation':maxima['moment']}


def qualify_native_precision(request):
    """Guarded backend probe; caller validates authority before importing Torch."""
    if not torch.backends.mps.is_available():
        return {'status':'unsupported','reason':'MPS unavailable; no native precision claim','attempts':[]}
    started = time.perf_counter(); attempts = []
    prefix = Path(request['run_dir'])/'precision_attempts.jsonl' if request.get('run_dir') else None
    if prefix is not None:
        prefix.parent.mkdir(parents=True,exist_ok=True)
        with prefix.open('x'):
            pass
    def retain(row):
        attempts.append(row)
        if prefix is not None:
            with prefix.open('a') as stream:
                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                stream.flush();os.fsync(stream.fileno())
    data = load_worker_data(request['data_path'],request['data_sha256'],phase='development')
    for precision in ('bf16','fp16'):
        try:
            fixture = ModelFixture(data,base_seed=request['seed'],variant=request['variant'],batch_size=request['arm']['batch'])
            if precision=='fp16' and not hasattr(torch.amp,'GradScaler'):
                raise NativePrecisionUnsupported('installed native MPS GradScaler unavailable')
            scaler = torch.amp.GradScaler('mps') if precision == 'fp16' else None
            if scaler is not None and (not scaler.is_enabled() or getattr(scaler,'_device',None)!='mps'):
                raise NativePrecisionUnsupported('installed native MPS GradScaler is disabled/unavailable')
            state = NativePrecisionState(fixture.model,fixture.opt,fixture.stream,precision=precision,scaler=scaler,
                scaler_qualification={'backend':'mps','native_grad_scaler':True} if scaler else None)
            for _ in range(2): state.step(request['arm']['lr'])
            torch.mps.synchronize(); saved = state.snapshot()
            state.step(request['arm']['lr']); torch.mps.synchronize(); expected = state.snapshot()
            state.restore(saved); state.step(request['arm']['lr']); torch.mps.synchronize(); replay = state.snapshot()
            replay_check = validate_native_precision_replay(expected,replay,device_type=next(state.model.parameters()).device.type)
            evaluation = evaluate(state.model,data['evaluation_tokens'],data['evaluation_monitor'],batch_size=16)
            retain({'precision':precision,'status':'pass','fp32_evaluation':evaluation,'replay_check':replay_check})
            return {'status':'pass','precision':precision,'native_grad_scaler':scaler is not None,
                    'backend':'mps','attempts':attempts,'replay_check':replay_check,'wall_s':time.perf_counter()-started}
        except Exception as exc:
            unsupported = precision_backend_unsupported(exc)
            retain({'precision':precision,'status':'unsupported' if unsupported else 'numerical_failure',
                    'reason':str(exc),'exception':type(exc).__name__,'wall_s':time.perf_counter()-started})
            if not unsupported:
                raise
    return {'status':'unsupported','reason':'installed native precision modes unsupported','attempts':attempts,
            'wall_s':time.perf_counter()-started}


def run_precision_worker(request):
    implementation = time.get_clock_info('perf_counter').implementation
    if request['clock_implementation'] != implementation:
        raise ValueError('parent/precision worker clock mismatch')
    arm = request['arm']
    if arm['sampler'] is not None or arm['precision'] not in ('bf16','fp16'):
        raise ValueError('precision worker accepts dense native controls only')
    evidence = request['precision_qualification']
    if evidence.get('status') != 'pass' or evidence.get('precision') != arm['precision'] or evidence.get('backend') != 'mps':
        raise ValueError('precision worker lacks qualified actual backend evidence')
    charged = ChargedClock(request['launch_perf_s'],prior_elapsed_s=request['prior_elapsed_s'])
    data = load_worker_data(request['data_path'],request['data_sha256'],phase=request['phase'])
    fixture = ModelFixture(data,base_seed=request['seed'],variant=request['variant'],batch_size=arm['batch'])
    scaler = torch.amp.GradScaler('mps') if arm['precision'] == 'fp16' else None
    state = NativePrecisionState(fixture.model,fixture.opt,fixture.stream,precision=arm['precision'],scaler=scaler,
        scaler_qualification={'backend':'mps','native_grad_scaler':True} if scaler and evidence.get('native_grad_scaler') else None)
    return train_budget(state,data,request,charged)
