"""Discovery tests protect evidence, RNG isolation and fail-closed approval."""

import hashlib
import json
from pathlib import Path

import pytest
import torch


def mac_fixture():
    # Recorded native command shapes; identifier values are synthetic sentinels.
    return {
        "architecture": "arm64", "python_architecture": "arm64", "os_version": "26.7.1",
        "physical_memory_bytes": 38654705664, "disk_free_bytes": 74000000000,
        "system_profiler": {"SPHardwareDataType": [{"chip_type": "Apple M3 Max",
            "machine_model": "Mac15,11", "serial_number": "SECRET-SERIAL",
            "platform_UUID": "SECRET-UUID", "machine_name": "PERSONAL-NAME"}],
            "SPDisplaysDataType": [{"sppci_model": "Apple M3 Max", "spdisplays_metal": "spdisplays_metal3"}]},
        "serial_number": "SECRET-SERIAL", "unknown": {"secret": "SECRET-UUID"},
        "power": {"source": "ac", "low_power_mode": False, "personal": "PERSONAL-NAME"},
    }


def test_discovery_never_publishes_identifiers():
    from oar.hardware import redact_hardware
    report = redact_hardware(mac_fixture())
    serialized = json.dumps(report).lower()
    for forbidden in ("serial", "uuid", "secret", "personal", "machine_name"):
        assert forbidden not in serialized
    assert report["physical_memory_bytes"] == 38654705664
    assert report["chip"] == "Apple M3 Max"
    assert report["model"] == "Mac15,11"
    assert report["power"] == {"source": "ac", "low_power_mode": False, "mode_code": None}


@pytest.mark.parametrize("available,arch,failed,scope", [
    (False, "arm64", False, "correctness_only"),
    (True, "x86_64", False, "correctness_only"),
    (True, "arm64", True, "correctness_only"),
    (True, "arm64", False, "exploratory_gpu"),
])
def test_environment_classifies_actual_backend(monkeypatch, available, arch, failed, scope):
    from oar import hardware
    raw = mac_fixture()
    raw["python_architecture"] = arch
    monkeypatch.setattr(hardware, "_collect_hardware", lambda: raw)
    monkeypatch.setattr(hardware, "_backend_state", lambda: {"built": True, "available": available})
    monkeypatch.setattr(hardware, "check_operations", lambda device: [
        {"name": "gemm", "critical": True, "status": "unsupported" if failed and device == "mps" else "pass",
         "execution_isolation": "fresh_process", "fallback_environment_before_torch_import": "0"}])
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    report = hardware.discover_environment()
    assert report["scope"] == scope
    assert report["gpu_claim_eligible"] is (scope == "exploratory_gpu")
    assert report["fallback"]["enabled"] is (False if available else None)
    if not available:
        assert report["accelerator"]["available"] is False
        assert report["accelerator"]["absence_is_hardware_limitation"] is False


def test_fallback_enabled_blocks_gpu_claim(monkeypatch):
    from oar import hardware
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    report = hardware.check_operations("mps")
    assert report and all(row["status"] == "blocked" for row in report)


def test_cpu_operations_are_real_and_preserve_model_rng():
    from oar.hardware import check_operations
    torch.manual_seed(302)
    before = torch.get_rng_state().clone()
    rows = {row["name"]: row for row in check_operations("cpu")}
    assert torch.equal(before, torch.get_rng_state())
    for name in ("gemm", "autograd", "adamw", "cpu_gaussian_transfer", "model_rng_restore", "reference_fp64", "qr", "svd"):
        assert rows[name]["status"] == "pass", rows[name]
    assert rows["cpu_gaussian_transfer"]["generation_device"] == "cpu"
    assert rows["cpu_gaussian_transfer"]["independent_streams"] is True
    assert rows["model_rng_restore"]["replayed_equal"] is True
    assert rows["gemm"]["dtype"] == "float32"


def test_unsupported_critical_operation_is_recorded(monkeypatch):
    from oar.hardware import check_operations
    def unsupported(*args, **kwargs):
        raise NotImplementedError("unsupported kernel; private details must not be exported")
    monkeypatch.setattr(torch, "matmul", unsupported)
    rows = {row["name"]: row for row in check_operations("cpu")}
    assert rows["gemm"]["status"] == "unsupported"
    assert rows["gemm"]["critical"] is True
    assert "private" not in json.dumps(rows)


def test_invalid_device_rejected():
    from oar.hardware import check_operations
    with pytest.raises(ValueError, match="device"):
        check_operations("cuda")


def test_file_hash_is_streamed_sha256(tmp_path):
    from oar.provenance import hash_file
    path = tmp_path / "tiny.txt"
    path.write_bytes(b"abc")
    assert hash_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def write_asset(tmp_path, licenses=("MIT",), reviewed=True):
    asset = tmp_path / "tiny.txt"
    asset.write_bytes(b"abc")
    notices = []
    for index, license_id in enumerate(licenses):
        path = tmp_path / f"NOTICE{index}"
        path.write_text(f"Fixture notice: {license_id}\n")
        notices.append({"path": path.name, "license": license_id,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    asset.with_suffix(".txt.notice.json").write_text(json.dumps({
        "source": "https://example.org/tiny", "revision": "a" * 40,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
        "notices": notices, "review_status": "approved" if reviewed else "unreviewed",
        "restrictions": ["preserve notice"],
    }))
    return asset


@pytest.mark.parametrize("licenses,reviewed,status", [
    (("MIT",), True, "approved"), (("MIT", "Apache-2.0"), True, "blocked"),
    ((), True, "blocked"), (("MIT",), False, "blocked"),
])
def test_asset_permission_requires_consistent_reviewed_notices(tmp_path, licenses, reviewed, status):
    from oar.provenance import inspect_asset
    asset = write_asset(tmp_path, licenses, reviewed)
    report = inspect_asset(asset.as_uri(), "a" * 40)
    assert report["approval_status"] == status
    assert report["size_bytes"] == 3
    assert report["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert report["revision"] == "a" * 40
    assert report["restrictions"] == ["preserve notice"]
    for notice in report["notice_evidence"]:
        assert len(notice["sha256"]) == 64


def test_asset_missing_or_changed_revision_never_approved(tmp_path):
    from oar.provenance import inspect_asset
    asset = write_asset(tmp_path)
    assert inspect_asset(str(asset), "b" * 40)["approval_status"] == "blocked"
    asset.with_suffix(".txt.notice.json").unlink()
    assert inspect_asset(str(asset), "a" * 40)["approval_status"] == "blocked"
    asset.unlink()
    assert inspect_asset(str(asset), "a" * 40)["local_status"] == "missing"


def test_remote_asset_is_not_downloaded_or_approved():
    from oar.provenance import inspect_asset
    report = inspect_asset("https://example.org/corpus", "a" * 40)
    assert report["local_status"] == "not_downloaded"
    assert report["approval_status"] == "blocked"
    assert report["sha256"] is None


def test_discovery_config_rejects_hidden_fallback(tmp_path):
    from oar.config import load_discovery_config
    path = tmp_path / "discovery.yaml"
    path.write_text("schema_version: 1\ndevice: auto\nallow_mps_fallback: true\n")
    with pytest.raises(ValueError, match="fallback"):
        load_discovery_config(path)
    path.write_text("schema_version: 1\ndevice: auto\nallow_mps_fallback: false\n")
    assert load_discovery_config(path).device == "auto"


def test_doctor_writes_classification_without_fabricating_gate(tmp_path, monkeypatch, capsys):
    from oar import cli, hardware
    monkeypatch.setattr(hardware, "discover_environment", lambda device="auto": {
        "schema_version": 1, "scope": "correctness_only", "gpu_claim_eligible": False})
    out = tmp_path / "environment.json"
    assert cli.main(["doctor", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["scope"] == "correctness_only"
    assert "correctness_only" in capsys.readouterr().out
    assert not (tmp_path / "gates.jsonl").exists()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual visible MPS; rerun outside sandbox")
def test_mps_explicit_cpu_fp64_decompositions_and_rng_isolation():
    from oar.hardware import check_operations
    before = torch.mps.get_rng_state().clone()
    rows = {row["name"]: row for row in check_operations("mps")}
    assert torch.equal(before, torch.mps.get_rng_state())
    for name in ("qr_cpu_fp64", "svd_cpu_fp64"):
        assert rows[name]["status"] == "pass", rows[name]
        assert rows[name]["execution_device"] == "cpu"
        assert rows[name]["dtype"] == "float64"


def test_changed_asset_bytes_invalidate_notice_review(tmp_path):
    from oar.provenance import inspect_asset
    asset = write_asset(tmp_path)
    sidecar = asset.with_suffix(".txt.notice.json")
    payload = json.loads(sidecar.read_text())
    payload["sha256"] = hashlib.sha256(b"abc").hexdigest()
    sidecar.write_text(json.dumps(payload))
    asset.write_bytes(b"changed")
    assert inspect_asset(str(asset), "a" * 40)["approval_status"] == "blocked"


def test_native_shell_fixture_parses_power_mode_and_redacts(monkeypatch):
    from oar import hardware
    from types import SimpleNamespace
    responses = {
        ("uname", "-m"): "arm64", ("sw_vers", "-productVersion"): "26.7.1",
        ("sysctl", "-n", "hw.memsize"): "38654705664",
        ("system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"): json.dumps(mac_fixture()["system_profiler"]),
        ("pmset", "-g", "custom"): "AC Power:\n powermode 0\n", ("pmset", "-g", "batt"): "Now drawing from 'AC Power'",
        ("vm_stat",): "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 42.",
    }
    def shell(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=responses[tuple(args)])
    monkeypatch.setattr(hardware.subprocess, "run", shell)
    report = hardware.redact_hardware(hardware._collect_hardware())
    assert report["physical_memory_bytes"] == 38654705664
    assert report["power"]["mode_code"] == 0
    assert "SECRET" not in json.dumps(report)


def test_tiny_attention_backward_is_qualified():
    from oar.hardware import check_operations
    rows = {row["name"]: row for row in check_operations("cpu")}
    assert rows["attention_backward"]["status"] == "pass"
    assert rows["attention_backward"]["critical"] is True


def test_changed_notice_bytes_invalidate_review(tmp_path):
    from oar.provenance import inspect_asset
    asset = write_asset(tmp_path)
    (tmp_path / 'NOTICE0').write_text('replaced license')
    assert inspect_asset(str(asset), 'a' * 40)['approval_status'] == 'blocked'


def test_failed_cpu_reference_blocks_all_execution_claims(monkeypatch):
    from oar import hardware
    monkeypatch.setattr(hardware, '_collect_hardware', mac_fixture)
    monkeypatch.setattr(hardware, '_backend_state', lambda: {'built': False, 'available': False})
    monkeypatch.setattr(hardware, 'check_operations', lambda device: [
        {'name': 'reference_fp64', 'critical': True, 'status': 'unsupported'}])
    report = hardware.discover_environment()
    assert report['scope'] == 'blocked'
    assert report['cpu_reference_eligible'] is False


@pytest.mark.parametrize('license_id', ['unresolved', 'unknown', 'NOASSERTION', ''])
def test_unresolved_license_cannot_be_approved(tmp_path, license_id):
    from oar.provenance import inspect_asset
    asset = write_asset(tmp_path, (license_id,))
    assert inspect_asset(str(asset), 'a' * 40)['approval_status'] == 'blocked'


@pytest.mark.parametrize('config_text', [
    'schema_version: 1\ndevice: auto\nallow_mps_fallback: true\n',
    'schema_version: [\n',
])
def test_doctor_invalid_config_returns_usage_error_without_traceback(tmp_path, capsys, config_text):
    from oar.cli import main
    path = tmp_path / 'invalid.yaml'
    path.write_text(config_text)
    out = tmp_path / 'environment.json'
    assert main(['doctor', '--config', str(path), '--out', str(out)]) == 2
    diagnostic = capsys.readouterr().err
    assert 'doctor:' in diagnostic
    assert 'Traceback' not in diagnostic
    assert not out.exists()


def test_mps_import_order_uses_fresh_fallback_disabled_worker():
    import os
    import subprocess
    import sys
    script = '''
import json, os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
def forbidden_parent_rng(*args, **kwargs):
    raise AssertionError('MPS checks must not touch an already imported Torch process')
torch.mps.get_rng_state = forbidden_parent_rng
from oar.hardware import check_operations
print(json.dumps(check_operations('mps')))
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env={**os.environ, 'PYTORCH_ENABLE_MPS_FALLBACK': '1'}, timeout=60)
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    assert rows
    assert all(row['execution_isolation'] == 'fresh_process' for row in rows)
    assert all(row['fallback_environment_before_torch_import'] == '0' for row in rows)
