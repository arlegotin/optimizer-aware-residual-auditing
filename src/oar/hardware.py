"""Redacted native discovery and tiny capability checks, never throughput claims."""

from datetime import datetime, timezone
from importlib import metadata
import json
import os
import platform
import re
import shutil
import subprocess
import time

import psutil


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _collect_hardware():
    raw = {"architecture": _command(["uname", "-m"]),
           "python_architecture": platform.machine(),
           "os_version": _command(["sw_vers", "-productVersion"]),
           "physical_memory_bytes": None, "disk_free_bytes": shutil.disk_usage(".").free}
    memory = _command(["sysctl", "-n", "hw.memsize"])
    if memory and memory.isdecimal():
        raw["physical_memory_bytes"] = int(memory)
    profile = _command(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"])
    try:
        raw["system_profiler"] = json.loads(profile) if profile else {}
    except ValueError:
        raw["system_profiler"] = {}
    power = _command(["pmset", "-g", "custom"])
    source = _command(["pmset", "-g", "batt"])
    raw["power"] = {"source": "ac" if source and "AC Power" in source else
                    "battery" if source and "Battery Power" in source else "unknown",
                    "low_power_mode": None, "mode_code": None}
    # pmset custom may show different AC/battery policies: choose current source only.
    if power and raw["power"]["source"] != "unknown":
        heading = "AC Power:" if raw["power"]["source"] == "ac" else "Battery Power:"
        section = power.split(heading, 1)[-1].split("Power:", 1)[0]
        match = re.search(r"lowpowermode\s+(\d+)", section)
        if match:
            raw["power"]["low_power_mode"] = match.group(1) != "0"
        mode = re.search(r"\bpowermode\s+(\d+)", section)
        if mode:
            raw["power"]["mode_code"] = int(mode.group(1))
    # vm_stat is observed for availability, but its unfiltered text is not published.
    raw["vm_stat_available"] = _command(["vm_stat"]) is not None
    return raw


def redact_hardware(raw: dict) -> dict:
    """Allowlist needed facts; never retain arbitrary profiler fields or raw output."""
    result = {key: raw.get(key) for key in (
        "architecture", "python_architecture", "os_version", "physical_memory_bytes", "disk_free_bytes")}
    profile = raw.get("system_profiler", {})
    hardware = profile.get("SPHardwareDataType", [{}]) or [{}]
    displays = profile.get("SPDisplaysDataType", [])
    result.update(chip=hardware[0].get("chip_type"), model=hardware[0].get("machine_model"),
                  gpu_models=[item["sppci_model"] for item in displays if "sppci_model" in item])
    power = raw.get("power", {})
    result["power"] = {key: power.get(key) for key in ("source", "low_power_mode", "mode_code")}
    result["vm_stat_available"] = raw.get("vm_stat_available", False)
    return result


def _backend_state():
    import torch
    return {"built": torch.backends.mps.is_built(), "available": torch.backends.mps.is_available()}


def check_operations(device: str) -> list[dict[str, object]]:
    """Run tiny FP32 checks; CPU FP64 decomposition is explicit and separately charged.

    Unsupported operations publish only exception classes. Shape-specific capability,
    custom OAR autograd and replay/checkpoint equivalence belong to later gates.
    """
    if device not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        return [{"name": "fallback_policy", "critical": True, "status": "blocked",
                 "reason": "start a fresh process with PYTORCH_ENABLE_MPS_FALLBACK=0"}]
    import torch
    import torch.nn.functional as F
    rows = []

    def run(name, action, critical=True, dtype="float32"):
        start = time.perf_counter()
        row = {"name": name, "device": device, "dtype": dtype, "critical": critical}
        try:
            details = action() or {}
            if device == "mps":
                torch.mps.synchronize()
            row.update(details, status="pass")
        except (RuntimeError, NotImplementedError, AssertionError) as exc:
            row.update(status="unsupported", error_type=type(exc).__name__)
        row["wall_s"] = time.perf_counter() - start
        rows.append(row)

    def gemm():
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).T.to(device)
        actual = torch.matmul(x, x.T)
        expected = torch.tensor([[80, 92, 104, 116], [92, 107, 122, 137],
                                 [104, 122, 140, 158], [116, 137, 158, 179]], dtype=torch.float32)
        torch.testing.assert_close(actual.cpu(), expected)
        return {"input_shape": [4, 3], "noncontiguous": not x.is_contiguous()}

    def autograd():
        x = torch.tensor([[1., 2.]], device=device, requires_grad=True)
        w = torch.tensor([[3., 4.]], device=device, requires_grad=True)
        b = torch.tensor([5.], device=device, requires_grad=True)
        F.linear(x, w, b).sum().backward()
        torch.testing.assert_close(x.grad.cpu(), torch.tensor([[3., 4.]]))
        torch.testing.assert_close(w.grad.cpu(), torch.tensor([[1., 2.]]))
        torch.testing.assert_close(b.grad.cpu(), torch.tensor([1.]))

    def adamw():
        p = torch.tensor([1., -2.], device=device, requires_grad=True)
        optimizer = torch.optim.AdamW([p], lr=.1, weight_decay=.01, foreach=False, fused=False)
        p.grad = torch.tensor([1., -1.], device=device)
        optimizer.step()
        torch.testing.assert_close(p.detach().cpu(), torch.tensor([.899, -1.898]))
        assert optimizer.state[p]["step"].item() == 1
        return {"foreach": False, "fused": False}

    def probes():
        before = torch.get_rng_state().clone()
        audit = torch.Generator(device="cpu").manual_seed(22)
        construction = torch.Generator(device="cpu").manual_seed(11)
        start = time.perf_counter()
        z = torch.randn(4, 3, generator=audit, device="cpu")
        other = torch.randn(4, 3, generator=construction, device="cpu")
        generation_s = time.perf_counter() - start
        assert not torch.equal(z, other)
        replay = torch.randn(4, 3, generator=torch.Generator().manual_seed(22))
        assert torch.equal(z, replay) and torch.equal(before, torch.get_rng_state())
        start = time.perf_counter()
        moved = z.to(device)
        if device == "mps":
            torch.mps.synchronize()
        transfer_s = time.perf_counter() - start
        assert torch.equal(moved.cpu(), z)
        return {"generation_device": "cpu", "independent_streams": True,
                "generation_s": generation_s, "transfer_s": transfer_s}

    def model_rng():
        get = torch.get_rng_state if device == "cpu" else torch.mps.get_rng_state
        put = torch.set_rng_state if device == "cpu" else torch.mps.set_rng_state
        initial = get().clone()
        try:
            a = torch.rand(8, device=device)
            put(initial)
            b = torch.rand(8, device=device)
            assert torch.equal(a.cpu(), b.cpu())
        finally:
            put(initial)
        return {"replayed_equal": True}

    def reference():
        x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        torch.testing.assert_close(x.T @ x, torch.tensor([[10., 14.], [14., 20.]], dtype=torch.float64))
        return {"execution_device": "cpu"}

    def decomposition(kind, cpu=False):
        target = "cpu" if cpu else device
        dtype = torch.float64 if cpu else torch.float32
        x = torch.tensor([[1., 2.], [3., 5.], [2., 1.]], device=device)
        transfer_start = time.perf_counter()
        x = x.to(device=target).to(dtype=dtype)
        transfer_s = time.perf_counter() - transfer_start
        if kind == "qr":
            q, r = torch.linalg.qr(x)
            reconstructed = q @ r
        else:
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            reconstructed = (u * s) @ vh
        torch.testing.assert_close(reconstructed, x)
        # Include the explicit return transfer for refresh bases.
        return_start = time.perf_counter()
        reconstructed.to(dtype=torch.float32).to(device=device)
        if device == "mps":
            torch.mps.synchronize()
        return_transfer_s = time.perf_counter() - return_start
        return {"execution_device": target, "transfer_s": transfer_s + return_transfer_s,
                "input_transfer_s": transfer_s, "return_transfer_s": return_transfer_s,
                "policy": "explicit_cpu_calibration" if cpu else "native"}

    def network_ops():
        x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4).requires_grad_()
        y = F.gelu(F.layer_norm(x, (4,)))
        F.cross_entropy(y, torch.tensor([0, 1, 2], device=device), reduction="none")[:2].mean().backward()
        assert torch.isfinite(x.grad).all().item()

    def indexing():
        x = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)
        indices = torch.tensor([0, 2], device=device)
        gathered = x.index_select(0, indices)
        out = torch.zeros_like(x).index_add_(0, indices, gathered)
        torch.testing.assert_close(out.cpu(), torch.tensor([[0., 1.], [0., 0.], [4., 5.], [0., 0.]]))
        samples = torch.multinomial(torch.tensor([0., 1.], device=device), 8, replacement=True)
        assert (samples == 1).all().item()

    def attention():
        q = torch.zeros(1, 1, 2, 2, device=device, requires_grad=True)
        k = torch.zeros_like(q, requires_grad=True)
        v = torch.tensor([[[[1., 2.], [3., 4.]]]], device=device, requires_grad=True)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        torch.testing.assert_close(y.cpu(), torch.tensor([[[[2., 3.], [2., 3.]]]]))
        y.sum().backward()
        torch.testing.assert_close(v.grad.cpu(), torch.ones(1, 1, 2, 2))
        assert torch.isfinite(q.grad).all().item() and torch.isfinite(k.grad).all().item()

    # CPU global model state stays unchanged even when an individual operation fails.
    with torch.random.fork_rng(devices=[0] if device == "mps" else [], device_type="mps" if device == "mps" else "cpu"):
        run("gemm", gemm)
        run("autograd", autograd)
        run("adamw", adamw)
        run("cpu_gaussian_transfer", probes)
        run("model_rng_restore", model_rng)
        run("reference_fp64", reference, dtype="float64")
        run("qr", lambda: decomposition("qr"), critical=False)
        run("svd", lambda: decomposition("svd"), critical=False)
        if device == "mps":
            run("qr_cpu_fp64", lambda: decomposition("qr", cpu=True), dtype="float64")
            run("svd_cpu_fp64", lambda: decomposition("svd", cpu=True), dtype="float64")
        run("norm_gelu_masked_loss_backward", network_ops)
        run("attention_backward", attention)
        run("gather_scatter_sampling", indexing, critical=False)
    return rows


def discover_environment(device: str = "auto") -> dict[str, object]:
    """Report visible capabilities; an unavailable sandbox GPU proves no hardware limit."""
    if device not in ("auto", "cpu", "mps"):
        raise ValueError("device must be auto, cpu or mps")
    start = time.perf_counter()
    hardware = redact_hardware(_collect_hardware())
    backend = _backend_state()
    fallback_value = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    fallback_disabled = fallback_value in (None, "0")
    cpu = check_operations("cpu")
    mps = check_operations("mps") if backend["available"] and device != "cpu" else []
    def passed(rows):
        critical = [row for row in rows if row["critical"]]
        return bool(critical) and all(row["status"] == "pass" for row in critical)
    native = hardware["architecture"] == hardware["python_architecture"] == "arm64"
    gpu_eligible = native and bool(backend["available"]) and fallback_disabled and passed(cpu) and passed(mps)
    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        packages[name] = {"version": dist.version,
                          "license": dist.metadata.get("License-Expression") or dist.metadata.get("License") or "unresolved"}
    return {"schema_version": 1, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": hardware, "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
            "packages": dict(sorted(packages.items())),
            "accelerator": {**backend, "absence_is_hardware_limitation": False,
                            "visibility_note": "unavailable may reflect process restrictions; verify outside sandbox before hardware conclusions"},
            "fallback": {"environment_value": fallback_value, "enabled": not fallback_disabled,
                         "policy": "disabled_required_for_critical_paths"},
            "native_arm64": native, "operations": {"cpu": cpu, "mps": mps},
            "cpu_reference_eligible": passed(cpu), "gpu_claim_eligible": gpu_eligible,
            "scope": "exploratory_gpu" if gpu_eligible else "correctness_only" if passed(cpu) else "blocked",
            "capability_scope": "tiny shapes only; no OAR, checkpoint, shape-headroom or speedup qualification",
            "memory": {"available_bytes": psutil.virtual_memory().available, "process_rss_bytes": psutil.Process().memory_info().rss},
            "discovery_wall_s": time.perf_counter() - start}
