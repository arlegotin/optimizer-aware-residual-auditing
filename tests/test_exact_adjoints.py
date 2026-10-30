"""Upstream credit propagation remains dense through several residual blocks."""

import pytest
import torch
from torch.nn import functional as F

from oar.linear import LinearCapture, projected_linear


def residual_gradients(operator):
    g = torch.Generator().manual_seed(605)
    x = torch.randn(2, 3, 5, generator=g, dtype=torch.float64).requires_grad_()
    params = []
    captures = []
    h = x
    for _ in range(4):
        w1 = (torch.randn(7, 5, generator=g, dtype=x.dtype) / 4).requires_grad_()
        w2 = (torch.randn(5, 7, generator=g, dtype=x.dtype) / 4).requires_grad_()
        b1 = torch.randn(7, generator=g, dtype=x.dtype).requires_grad_()
        b2 = torch.randn(5, generator=g, dtype=x.dtype).requires_grad_()
        c1 = LinearCapture(torch.eye(7, dtype=x.dtype)[:, :2], torch.randn(5, 3, generator=g, dtype=x.dtype), 0)
        c2 = LinearCapture(torch.eye(5, dtype=x.dtype)[:, :1], torch.randn(7, 3, generator=g, dtype=x.dtype), 0)
        hidden = operator(h, w1, b1, c1)
        h = h + operator(F.gelu(hidden), w2, b2, c2)
        params.extend((w1, w2, b1, b2))
        captures.extend((c1, c2))
    loss = (h.sin() + h.square() / 3).mean()
    return torch.autograd.grad(loss, (x, *params)), captures


def assert_exact_upstream(actual, dense):
    torch.testing.assert_close(actual[0], dense[0], rtol=1e-10, atol=1e-12)
    for index in range(4):
        for offset in (3, 4):
            torch.testing.assert_close(actual[4 * index + offset], dense[4 * index + offset], rtol=1e-10, atol=1e-12)


def test_deep_residual_blocks_preserve_input_and_bias_adjoints():
    actual, captures = residual_gradients(projected_linear)
    dense, _ = residual_gradients(lambda x, w, b, capture: F.linear(x, w, b))
    assert_exact_upstream(actual, dense)
    for index, capture in enumerate(captures):
        weight_index = 1 + 4 * (index // 2) + index % 2
        u = capture.basis
        torch.testing.assert_close(actual[weight_index], u @ (u.T @ dense[weight_index]), rtol=1e-10, atol=1e-12)


def test_reference_detects_injected_wrong_input_derivative():
    def corrupted(x, w, b, capture):
        # Identical forward, deliberately halve only the incoming derivative.
        x_wrong = x.detach() + 0.5 * (x - x.detach())
        return projected_linear(x_wrong, w, b, capture)
    wrong, _ = residual_gradients(corrupted)
    dense, _ = residual_gradients(lambda x, w, b, capture: F.linear(x, w, b))
    with pytest.raises(AssertionError):
        assert_exact_upstream(wrong, dense)
