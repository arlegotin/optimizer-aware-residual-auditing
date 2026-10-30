"""Unequal masked means must produce the effective-batch objective exactly once."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from oar.linear import LinearCapture, projected_linear


@pytest.fixture
def accumulated_case():
    g = torch.Generator().manual_seed(502)
    counts = (1, 3, 2, 6, 4, 7, 5)
    sizes = (2, 5, 3, 8, 6, 9, 7)
    xs = [torch.randn(n, 6, generator=g, dtype=torch.float64).requires_grad_() for n in sizes]
    targets = [torch.randn(n, 4, generator=g, dtype=torch.float64) for n in sizes]
    w = torch.randn(4, 6, generator=g, dtype=torch.float64).requires_grad_()
    b = torch.randn(4, generator=g, dtype=torch.float64).requires_grad_()
    u = torch.linalg.qr(torch.randn(4, 4, generator=g, dtype=torch.float64)).Q[:, :2]
    z = torch.randn(6, 5, generator=g, dtype=torch.float64)
    capture = LinearCapture(u, z, 6)
    # The reference is one materialized dense effective objective, using native autograd.
    dense_loss = sum((F.linear(x, w, b)[:c] - t[:c]).square().sum()
                     for x, t, c in zip(xs, targets, counts)) / sum(counts)
    exact = torch.autograd.grad(dense_loss, (*xs, w, b))
    sketches = []
    for x, target, count in zip(xs, targets, counts):
        previous = capture.r_sum.clone()
        loss_mean = (projected_linear(x, w, b, capture)[:count] - target[:count]).square().sum() / count
        (count / sum(counts) * loss_mean).backward()
        sketches.append(capture.r_sum - previous)
    return SimpleNamespace(capture=capture, exact_residual=exact[-2] - u @ (u.T @ exact[-2]),
                           z=z, input_grad=torch.cat([x.grad for x in xs]),
                           dense_input_grad=torch.cat(exact[:-2]), w=w, b=b, exact=exact, u=u,
                           sketches=sketches)


def test_streamed_residual_is_aggregate(accumulated_case):
    case = accumulated_case
    torch.testing.assert_close(case.capture.r_sum, case.exact_residual @ case.z, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(case.input_grad, case.dense_input_grad, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(case.w.grad, case.u @ (case.u.T @ case.exact[-2]), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(case.b.grad, case.exact[-1], rtol=1e-10, atol=1e-12)
    assert case.capture.backward_contributions == 7
    # A sum of squared norms drops cross-microbatch terms and is a different statistic.
    assert not torch.isclose(case.capture.r_sum.square().sum(), sum(r.square().sum() for r in case.sketches))
