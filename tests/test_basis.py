"""Construction has its own RNG, drops null modes and enforces past-only use."""

import pytest
import torch

from oar.basis import build_basis
from oar.linear import LinearCapture


@pytest.mark.parametrize("shape,rank", [((9, 6), 3), ((4, 7), 4), ((7, 4), 4), ((5, 6), 0)])
def test_basis_ordering_and_projector_match_exact_low_rank_reference(shape, rank):
    g = torch.Generator().manual_seed(701)
    left = torch.linalg.qr(torch.randn(shape[0], shape[0], generator=g, dtype=torch.float64)).Q
    right = torch.linalg.qr(torch.randn(shape[1], shape[1], generator=g, dtype=torch.float64)).Q
    k = min(shape)
    singular = torch.arange(k, 0, -1, dtype=torch.float64)
    gradient = (left[:, :k] * singular) @ right[:, :k].T
    before = torch.get_rng_state().clone()
    state = build_basis(gradient.T.contiguous().T, rank, 8, g, source_step=11)
    assert torch.equal(before, torch.get_rng_state())
    assert state.source_step == 11
    assert state.vectors.shape == (shape[0], rank)
    u = state.for_step(12)
    torch.testing.assert_close(u @ u.T, left[:, :rank] @ left[:, :rank].T, rtol=1e-10, atol=1e-12)
    energies = (u.T @ gradient).square().sum(1)
    assert torch.all(energies[:-1] >= energies[1:])
    assert state.orthogonality < 1e-12
    assert not u.requires_grad
    assert state.construction_wall_s >= 0
    assert state.decomposition_device == 'cpu'
    assert state.construction_width == (min(shape[1], rank + 8) if rank else 0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("nonzero_rank", [0, 1, 2])
def test_rank_deficiency_does_not_promote_qr_completion(dtype, nonzero_rank):
    g = torch.Generator().manual_seed(705)
    a = torch.randn(8, nonzero_rank, generator=g, dtype=dtype)
    b = torch.randn(nonzero_rank, 6, generator=g, dtype=dtype)
    gradient = (a @ b).requires_grad_()
    state = build_basis(gradient, 5, 8, g, source_step=0)
    assert state.numerical_rank == nonzero_rank
    assert state.vectors.shape == (8, nonzero_rank)
    torch.testing.assert_close(state.vectors @ (state.vectors.T @ gradient.detach()), gradient.detach(), rtol=2e-5, atol=2e-6)


def test_past_only_capture_rejects_current_and_future_basis():
    state = build_basis(torch.eye(4, dtype=torch.float64), 3, 8, torch.Generator().manual_seed(709), 5)
    z = torch.ones(4, 2, dtype=torch.float64)
    for current in (4, 5):
        with pytest.raises(ValueError, match="past|source_step"):
            LinearCapture.from_basis(state, current_step=current, probes=z, basis_version=2)
    capture = LinearCapture.from_basis(state, current_step=6, probes=z, basis_version=2, rank=1)
    assert capture.basis.shape == (4, 1)
    with pytest.raises(ValueError, match="rank"):
        state.for_step(6, rank=4)
    state.vectors.add_(1)
    with pytest.raises(RuntimeError, match="modified|version"):
        state.for_step(6)


def test_skinny_randomized_construction_uses_only_its_own_generator():
    gradient = torch.randn(19, 13, generator=torch.Generator().manual_seed(712), dtype=torch.float64)
    before = torch.get_rng_state().clone()
    states = [build_basis(gradient, 2, 3, torch.Generator().manual_seed(seed), 0) for seed in (713, 713, 714)]
    assert torch.equal(before, torch.get_rng_state())
    assert all(state.construction_width == 5 for state in states)
    projectors = [s.vectors @ s.vectors.T for s in states]
    torch.testing.assert_close(projectors[0], projectors[1], rtol=0, atol=0)
    # A full exact SVD shortcut would incorrectly ignore construction randomness.
    assert torch.linalg.vector_norm(projectors[0] - projectors[2]) > 0.01
    for state in states:
        energy = (state.vectors.T @ gradient).square().sum(1)
        assert energy[0] >= energy[1]
        assert state.orthogonality < 1e-12


@pytest.mark.parametrize("gradient,rank,p", [
    (torch.ones(3), 1, 8), (torch.ones(3, 4), -1, 8), (torch.ones(3, 4), 1, -1),
    (torch.ones(3, 4, dtype=torch.int64), 1, 8),
    (torch.full((3, 4), float('nan')), 1, 8),
])
def test_invalid_basis_construction_fails(gradient, rank, p):
    with pytest.raises(ValueError):
        build_basis(gradient, rank, p, torch.Generator().manual_seed(710), 0)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual visible MPS; rerun outside sandbox")
def test_mps_basis_explicit_cpu_decomposition_preserves_model_rng():
    before = torch.mps.get_rng_state().clone()
    gradient = torch.arange(30, dtype=torch.float32).reshape(5, 6).to('mps')
    state = build_basis(gradient, 4, 8, torch.Generator().manual_seed(711), 1)
    assert state.vectors.device.type == 'mps'
    assert state.numerical_rank == 2
    assert state.transfer_bytes == gradient.numel() * gradient.element_size() + state.vectors.numel() * state.vectors.element_size()
    assert torch.equal(before, torch.mps.get_rng_state())
    u = state.for_step(2).cpu().double()
    torch.testing.assert_close(u @ (u.T @ gradient.cpu().double()), gradient.cpu().double(), rtol=2e-5, atol=2e-5)
