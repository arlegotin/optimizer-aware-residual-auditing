"""Contracts for resolved inputs and honest, fail-closed evidence records."""

import json
from pathlib import Path

import pytest
import torch
import yaml


FIXTURES = Path(__file__).parent / "fixtures"


def config_payload():
    return yaml.safe_load((FIXTURES / "config_minimal.yaml").read_text())


def write_config(tmp_path, payload):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_canonical_config_mapping_order():
    from oar.config import config_sha256, load_config, resolve_config

    left = load_config(FIXTURES / "config_minimal.yaml")
    right = load_config(FIXTURES / "config_reordered.yaml")
    assert config_sha256(left) == config_sha256(right)
    resolved = resolve_config(left)
    assert resolved["seeds"] == {"construction": 11, "audit": 22, "data": 33, "model": 44}
    assert resolved["paths"]["runs"] == "runs"
    assert resolved["scope"] == "correctness_structural"
    resolved["seeds"]["model"] = 100
    assert resolve_config(left)["seeds"]["model"] == 44


def test_config_hash_changes_with_seed(tmp_path):
    from oar.config import config_sha256, load_config

    payload = config_payload()
    payload["seeds"]["audit"] = 23
    assert config_sha256(load_config(write_config(tmp_path, payload))) != config_sha256(
        load_config(FIXTURES / "config_minimal.yaml")
    )


@pytest.mark.parametrize("section", [None, "seeds", "paths"])
def test_unknown_field_rejected(tmp_path, section):
    from oar.config import load_config

    payload = config_payload()
    (payload if section is None else payload[section])["surprise"] = 1
    with pytest.raises(ValueError, match="unknown"):
        load_config(write_config(tmp_path, payload))


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True), ("phase", "test"),
    ("method", "unknown"), ("device", "cuda"), ("device", "mps:0"),
    ("seeds", {"construction": -1, "audit": 22, "data": 33, "model": 44}),
    ("seeds", {"construction": True, "audit": 22, "data": 33, "model": 44}),
    ("seeds", {"construction": 2**63, "audit": 22, "data": 33, "model": 44}),
    ("paths", {"data": "data", "runs": "", "decisions": "decisions"}),
])
def test_invalid_config_rejected(tmp_path, field, value):
    from oar.config import load_config

    payload = config_payload()
    payload[field] = value
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, payload))


def test_missing_seed_rejected(tmp_path):
    from oar.config import load_config

    payload = config_payload()
    del payload["seeds"]["audit"]
    with pytest.raises(ValueError, match="missing"):
        load_config(write_config(tmp_path, payload))


def test_confirmatory_config_fails_closed(tmp_path):
    from oar.config import load_config

    payload = config_payload()
    payload["phase"] = "confirmatory"
    with pytest.raises(ValueError, match="unresolved.*protocol"):
        load_config(write_config(tmp_path, payload))


def test_duplicate_yaml_keys_rejected(tmp_path):
    from oar.config import load_config

    path = tmp_path / "duplicate.yaml"
    path.write_text((FIXTURES / "config_minimal.yaml").read_text() + "device: mps\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_config(path)


def test_gate_appends_nulls_and_stopped_prerequisite_blocks(tmp_path):
    from oar.gates import GateBlocked, record_gate, require_gate
    from oar.records import GateRecord

    ledger = tmp_path / "decisions" / "gates.jsonl"
    record_gate(GateRecord(package="WP00", status="pass", reason="fixture", next_action="WP01"), ledger)
    before = ledger.read_bytes()
    record_gate(GateRecord(package="WP01", status="stop", reason="invalid state", next_action="repair"), ledger)
    assert ledger.read_bytes().startswith(before)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert rows[1]["schema_version"] == 1
    assert rows[1]["measured_resources"] is None
    assert rows[1]["forecast_resources"] is None
    with pytest.raises(GateBlocked, match="WP01"):
        require_gate("WP02", ledger)


def test_gate_passed_prerequisites_allow_work(tmp_path):
    from oar.gates import record_gate, require_gate
    from oar.records import GateRecord

    ledger = tmp_path / "gates.jsonl"
    for package in ("WP00", "WP01"):
        record_gate(GateRecord(package=package, status="pass", reason="fixture", next_action="continue"), ledger)
    assert require_gate("WP02", ledger) is None


@pytest.mark.parametrize("package", ["WP02", "WP99"])
def test_unknown_or_missing_prerequisite_blocks(tmp_path, package):
    from oar.gates import GateBlocked, require_gate

    with pytest.raises(GateBlocked):
        require_gate(package, tmp_path / "missing.jsonl")


@pytest.mark.parametrize("status", ["stop", "inconclusive", "blocked"])
def test_latest_gate_record_controls_prerequisite(tmp_path, status):
    from oar.gates import GateBlocked, record_gate, require_gate
    from oar.records import GateRecord

    ledger = tmp_path / "gates.jsonl"
    for outcome in ("pass", status):
        record_gate(GateRecord(package="WP00", status=outcome, reason="fixture", next_action="review"), ledger)
    with pytest.raises(GateBlocked):
        require_gate("WP01", ledger)


@pytest.mark.parametrize("line", ['{"schema_version": 9}\n', 'broken\n'])
def test_corrupt_or_unknown_schema_gate_blocks(tmp_path, line):
    from oar.gates import GateBlocked, require_gate

    ledger = tmp_path / "gates.jsonl"
    ledger.write_text(line)
    with pytest.raises(GateBlocked):
        require_gate("WP01", ledger)


def test_step_measurements_are_explicit_nulls():
    from oar.records import Decision, DecisionKind, StepRecord, from_payload, to_payload, to_json

    record = StepRecord(logical_step=3, parameter_version=2, state_version=2,
                        decision=Decision(kind=DecisionKind.STOP_INVALID, reasons=("fixture",)),
                        valid_tokens=17, processed_tokens=34)
    payload = json.loads(to_json(record))
    assert payload["active_wall_s"] is None
    assert payload["checkpoint"] is None
    assert payload["decision"]["a"] is None
    assert payload["decision"]["B"] is None
    assert payload["decision"]["allowance"] is None
    assert from_payload(to_payload(record)) == record


def test_tensor_payload_keeps_tensors_and_json_refuses_them():
    from oar.records import AdamWParamState, PendingUpdate, from_payload, to_json, to_payload

    tensor = torch.tensor([1.25], dtype=torch.float64)
    pending = PendingUpdate(parameter_version=4, state_version=7,
                            candidate_parameters={"mlp.weight": tensor},
                            displacements={"mlp.weight": tensor},
                            optimizer_state={"mlp.weight": AdamWParamState(3, tensor, tensor)})
    restored = from_payload(to_payload(pending))
    assert torch.equal(restored.candidate_parameters["mlp.weight"], tensor)
    assert torch.equal(restored.displacements["mlp.weight"], tensor)
    assert restored.optimizer_state["mlp.weight"].exp_avg.dtype == torch.float64
    assert restored.consumed is False
    with pytest.raises(TypeError, match="tensor|Tensor"):
        to_json(pending)


def test_payload_reader_rejects_unknown_schema_and_fields():
    from oar.records import GateRecord, from_payload, to_payload

    payload = to_payload(GateRecord(package="WP00", status="blocked", reason="fixture", next_action="review"))
    with pytest.raises(ValueError, match="schema"):
        from_payload({**payload, "schema_version": 2})
    with pytest.raises(ValueError, match="unknown"):
        from_payload({**payload, "surprise": 1})


def test_json_rejects_fabricated_nonfinite_measurements():
    from oar.records import GateRecord, to_json

    with pytest.raises(ValueError):
        to_json(GateRecord(package="WP00", status="blocked", reason="fixture", next_action="review",
                           measured_resources={"wall_s": float("nan")}))


def test_cli_help_has_no_unimplemented_subcommands(capsys):
    from oar.cli import main

    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage:" in help_text
    assert "train" not in help_text
    assert "discover" not in help_text
    assert main([]) == 0
    with pytest.raises(SystemExit) as result:
        main(["train"])
    assert result.value.code == 2


def test_replay_context_payload_round_trip_preserves_descriptors_and_rng():
    from oar.records import ReplayableBatch, StepContext, from_payload, to_payload

    batch = ReplayableBatch(source_ids=("article-a",), microbatches=({"offset": 5},),
                            target_masks=((False, True, True),), valid_counts=(2,),
                            pre_cursor={"article": 0}, post_cursor={"article": 1})
    rng = torch.Generator().manual_seed(7).get_state()
    context = StepContext(logical_step=2, parameter_version=1, state_version=1, valid_tokens=2,
                          rng_snapshots={"model": rng}, schedule_position=2, batch=batch,
                          timing_boundary_s=1.5)
    restored = from_payload(to_payload(context))
    assert restored.batch == batch
    assert restored.batch.microbatches == ({"offset": 5},)
    assert restored.batch.target_masks == ((False, True, True),)
    assert restored.valid_tokens == 2
    assert torch.equal(restored.rng_snapshots["model"], rng)


def test_layer_audit_nulls_group_payload_and_decision_json_round_trip():
    from oar.records import AdamWGroup, Decision, DecisionKind, LayerAudit, from_payload, to_json

    audit = LayerAudit(layer_id="mlp.weight", shape=(4, 3), rank=2, basis_version=1,
                       probe_id="probe-1", probe_count=3, allocation=0.01)
    payload = json.loads(to_json(audit))
    for name in ("residual_sketch", "statistic", "bound", "orthogonality", "guardband", "profiling"):
        assert payload[name] is None
    assert from_payload(payload) == audit
    group = AdamWGroup(parameter_ids=("mlp.weight",), lr=0.001,
                       betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    assert from_payload(json.loads(to_json(group))) == group
    decision = Decision(kind=DecisionKind.REJECT_REPLAY, reasons=("unsafe_margin",),
                        a=2.0, B=1.1, allowance=0.2, next_step_changes={"mlp.weight": 4})
    assert from_payload(json.loads(to_json(decision))) == decision


def test_layer_audit_rejects_attached_tensor():
    from oar.records import LayerAudit

    with pytest.raises(ValueError, match="detach"):
        LayerAudit(layer_id="mlp.weight", shape=(4, 3), rank=2, basis_version=1,
                   probe_id="probe-1", probe_count=3, allocation=0.01,
                   residual_sketch=torch.ones(4, 3, requires_grad=True))


def test_gate_rechecks_earlier_stopped_prerequisites(tmp_path):
    from oar.gates import GateBlocked, record_gate, require_gate
    from oar.records import GateRecord

    ledger = tmp_path / "gates.jsonl"
    for index in range(5):
        record_gate(GateRecord(package=f"WP{index:02d}", status="pass", reason="fixture", next_action="continue"), ledger)
    record_gate(GateRecord(package="WP01", status="stop", reason="late defect", next_action="repair"), ledger)
    with pytest.raises(GateBlocked, match="WP01"):
        require_gate("WP05", ledger)


@pytest.mark.parametrize("status", ["unknown", "PASS"])
def test_gate_rejects_invalid_status_before_writing(tmp_path, status):
    from oar.gates import record_gate
    from oar.records import GateRecord

    ledger = tmp_path / "gates.jsonl"
    with pytest.raises(ValueError, match="status"):
        record_gate(GateRecord(package="WP00", status=status, reason="fixture", next_action="review"), ledger)
    assert not ledger.exists()


def test_discovery_and_synthesis_need_no_passed_gate(tmp_path):
    from oar.gates import require_gate

    for package in ("WP00", "WP09"):
        assert require_gate(package, tmp_path / "missing.jsonl") is None


def test_replay_descriptors_are_immutable_snapshots():
    from oar.records import ReplayableBatch, from_payload, to_payload

    descriptor = {"offsets": [3, 7]}
    cursor = {"article": 0}
    batch = ReplayableBatch(source_ids=("article-a",), microbatches=(descriptor,),
                            target_masks=((False, True),), valid_counts=(1,),
                            pre_cursor=cursor, post_cursor={"article": 1})
    descriptor["offsets"][0] = 99
    cursor["article"] = 99
    assert batch.microbatches[0]["offsets"] == (3, 7)
    assert batch.pre_cursor["article"] == 0
    with pytest.raises(TypeError):
        batch.microbatches[0]["offsets"] = (9,)
    restored = from_payload(to_payload(batch))
    with pytest.raises(TypeError):
        restored.pre_cursor["article"] = 8


def test_gate_payload_preserves_integer_resource_counts():
    from oar.records import GateRecord, from_payload, to_payload

    record = GateRecord(package="WP00", status="blocked", reason="fixture", next_action="review",
                        measured_resources={"bytes": 9007199254740993, "wall_s": 0.5})
    restored = from_payload(to_payload(record))
    assert restored.measured_resources["bytes"] == 9007199254740993
    assert type(restored.measured_resources["bytes"]) is int
    assert restored.measured_resources["wall_s"] == 0.5


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_nonfinite_gate_evidence_blocks_prerequisite(tmp_path, constant):
    from oar.gates import GateBlocked, require_gate
    from oar.records import GateRecord, to_payload

    payload = to_payload(GateRecord(package="WP00", status="pass", reason="fixture", next_action="continue",
                                    measured_resources={"wall_s": 0.5}))
    ledger = tmp_path / "gates.jsonl"
    ledger.write_text(json.dumps(payload).replace("0.5", constant) + "\n")
    with pytest.raises(GateBlocked):
        require_gate("WP01", ledger)
