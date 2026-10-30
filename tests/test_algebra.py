"""Dense autograd references detect projection, orientation and scaling defects."""

import pytest
import torch
from torch.nn import functional as F

from oar.linear import LinearCapture, projected_linear


@pytest.mark.parametrize("rank", [0, 2, 5])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 7)])
def test_projected_derivatives_match_dense_reference(rank, with_bias, shape):
    g = torch.Generator().manual_seed(401)
    # Transpose makes even the vector fixture strided.
    x = torch.randn(7, *shape[:-1], 2, generator=g, dtype=torch.float64)[..., 0].movedim(0, -1).requires_grad_()
    w = torch.randn(7, 5, generator=g, dtype=torch.float64).T.requires_grad_()
    b = torch.randn(5, generator=g, dtype=torch.float64).requires_grad_() if with_bias else None
    u = torch.linalg.qr(torch.randn(5, 5, generator=g, dtype=torch.float64)).Q[:, :rank]
    z = torch.randn(7, 4, generator=g, dtype=torch.float64)
    capture = LinearCapture(u, z, basis_version=3)
    y = projected_linear(x, w, b, capture)
    reference = F.linear(x, w, b)
    torch.testing.assert_close(y, reference, rtol=0, atol=0)
    d = torch.randn(y.shape, generator=g, dtype=y.dtype)
    parameters = (x, w, b) if with_bias else (x, w)
    exact = torch.autograd.grad(reference, parameters, d)
    actual = torch.autograd.grad(y, parameters, d)
    torch.testing.assert_close(actual[0], exact[0], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual[1], u @ (u.T @ exact[1]), rtol=1e-10, atol=1e-12)
    if with_bias:
        torch.testing.assert_close(actual[2], exact[2], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(capture.h_sum, u.T @ exact[1], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(capture.r_sum, (exact[1] - actual[1]) @ z, rtol=1e-10, atol=1e-12)
    assert capture.backward_contributions == 1
    assert not capture.h_sum.requires_grad and not capture.r_sum.requires_grad


def test_zero_adjoints_and_empty_rows():
    for n in (0, 3):
        x = torch.ones(n, 4, dtype=torch.float64, requires_grad=True)
        w = torch.ones(3, 4, dtype=torch.float64, requires_grad=True)
        capture = LinearCapture(torch.eye(3, dtype=torch.float64)[:, :1], torch.ones(4, 2, dtype=torch.float64), 0)
        y = projected_linear(x, w, None, capture)
        dx, dw = torch.autograd.grad(y, (x, w), torch.zeros_like(y))
        assert dx.count_nonzero() == dw.count_nonzero() == capture.r_sum.count_nonzero() == 0


def test_full_basis_passes_gradcheck():
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(2, dtype=torch.float64, requires_grad=True)
    def operator(x, w, b):
        capture = LinearCapture(torch.eye(2, dtype=x.dtype), torch.ones(3, 1, dtype=x.dtype), 0)
        return projected_linear(x, w, b, capture)
    assert torch.autograd.gradcheck(operator, (x, w, b))


def test_higher_order_request_rejected_before_capture_mutation():
    x = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    w = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    capture = LinearCapture(torch.eye(2, dtype=x.dtype), torch.ones(3, 1, dtype=x.dtype), 0)
    with pytest.raises(RuntimeError, match="higher.order"):
        torch.autograd.grad(projected_linear(x, w, None, capture).sum(), x, create_graph=True)
    assert capture.backward_contributions == 0


@pytest.mark.parametrize("target", ["basis", "probes"])
def test_mutated_capture_fails_before_streaming(target):
    x = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    w = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    capture = LinearCapture(torch.eye(2, dtype=x.dtype), torch.ones(3, 1, dtype=x.dtype), 0)
    y = projected_linear(x, w, None, capture)
    getattr(capture, target).add_(1)
    with pytest.raises(RuntimeError, match="modified|version"):
        y.sum().backward()
    assert capture.backward_contributions == 0


@pytest.mark.parametrize('parameter', ['weight', 'bias'])
def test_effective_batch_parameters_cannot_change_between_microbatches(parameter):
    x = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    w = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    b = torch.ones(2, dtype=torch.float64, requires_grad=True)
    capture = LinearCapture(torch.eye(2, dtype=x.dtype), torch.ones(3, 1, dtype=x.dtype), 0)
    projected_linear(x, w, b, capture).sum().backward()
    with torch.no_grad():
        (w if parameter == 'weight' else b).add_(1)
    with pytest.raises(RuntimeError, match='parameter|version'):
        projected_linear(x, w, b, capture)
    assert capture.backward_contributions == 1


@pytest.mark.parametrize("basis,probes", [
    (torch.ones(2, 3), torch.ones(3, 1)),
    (torch.eye(2), torch.ones(3, 0)),
    (torch.full((2, 1), float('nan')), torch.ones(3, 1)),
    (torch.eye(2), torch.full((3, 1), float('inf'))),
    (torch.ones(2, 1), torch.ones(3, 1)),
])
def test_invalid_capture_rejected(basis, probes):
    with pytest.raises(ValueError):
        LinearCapture(basis, probes, 0)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual visible MPS; rerun outside sandbox")
def test_mps_fp32_operator_against_cpu_fp64():
    g = torch.Generator().manual_seed(409)
    x64 = torch.randn(4, 3, 7, generator=g, dtype=torch.float64).transpose(0, 1)
    w64 = torch.randn(5, 7, generator=g, dtype=torch.float64)
    b64 = torch.randn(5, generator=g, dtype=torch.float64)
    u64 = torch.linalg.qr(torch.randn(5, 5, generator=g, dtype=torch.float64)).Q[:, :2]
    z64 = torch.randn(7, 6, generator=g, dtype=torch.float64)
    d64 = torch.randn(3, 4, 5, generator=g, dtype=torch.float64)
    x, w, b = [t.float().to('mps').requires_grad_() for t in (x64, w64, b64)]
    capture = LinearCapture(u64.float().to('mps'), z64.float().to('mps'), 0)
    y = projected_linear(x, w, b, capture)
    actual = torch.autograd.grad(y, (x, w, b), d64.float().to('mps'))
    xr, wr, br = [t.requires_grad_() for t in (x64, w64, b64)]
    exact = torch.autograd.grad(F.linear(xr, wr, br), (xr, wr, br), d64)
    expected = (exact[0], u64 @ (u64.T @ exact[1]), exact[2])
    errors = {}
    for name, value, ref in zip(('dx', 'dw', 'db'), actual, expected):
        value = value.cpu().double()
        errors[name] = (value - ref).abs().max().item()
        torch.testing.assert_close(value, ref, rtol=2e-5, atol=3e-6)
    residual = (exact[1] - expected[1]) @ z64
    errors['residual'] = (capture.r_sum.cpu().double() - residual).abs().max().item()
    torch.testing.assert_close(capture.r_sum.cpu().double(), residual, rtol=2e-5, atol=1e-5)
    print(f'MPS FP32 max absolute errors: {errors}')
