"""Local research commands, added only when their owning modules exist."""

import argparse
import json
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oar", description="Optimizer-aware residual auditing research")
    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser("doctor", help="record redacted environment and tiny capabilities")
    doctor.add_argument("--out", type=Path, required=True)
    doctor.add_argument("--config", type=Path)
    bench = commands.add_parser('bench', help='measure complete bounded operator costs in a fresh worker')
    bench.add_argument('--config', type=Path, required=True)
    oracle = commands.add_parser('audit-checkpoints', help='bounded synthetic structural oracle only')
    oracle.add_argument('--config', type=Path, required=True)
    report = commands.add_parser('report', help='synthesize reached or stopped evidence')
    report.add_argument('--gates', type=Path, required=True)
    report.add_argument('--run-index', type=Path)
    report.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'report':
        from oar.report import build_report
        try:
            path = build_report(args.gates, args.run_index, args.out)
        except (ValueError, TypeError, OSError) as exc:
            print(f'report: invalid evidence: {exc}', file=sys.stderr)
            return 2
        print(f'report: {path}')
        return 0
    if args.command == 'audit-checkpoints':
        from oar.oracle import run_structural_oracle
        from oar.gates import GateBlocked
        try:
            path = run_structural_oracle(args.config)
        except GateBlocked as exc:
            print(f'audit-checkpoints: blocked: {exc}', file=sys.stderr)
            return 3
        except (ValueError, OSError, TypeError) as exc:
            print(f'audit-checkpoints: invalid configuration: {exc}', file=sys.stderr)
            return 2
        except RuntimeError as exc:
            print(f'audit-checkpoints: blocked: {exc}', file=sys.stderr)
            return 3
        print(f'audit-checkpoints: {path}')
        return 0
    if args.command == 'bench':
        from oar.bench import benchmark, load_benchmark_config, write_cost_table
        try:
            config = load_benchmark_config(args.config)
        except (ValueError, OSError, TypeError) as exc:
            print(f'bench: invalid configuration: {exc}', file=sys.stderr)
            return 2
        try:
            table = benchmark(config)
        except RuntimeError as exc:
            print(f'bench: blocked: {exc}', file=sys.stderr)
            return 3
        write_cost_table(table, Path(config.benchmark.output))
        print(f'bench: {len(table.rows)} actual cost rows; {config.benchmark.output}')
        return 0
    if args.command == "doctor":
        from oar.config import DiscoveryConfig, load_discovery_config
        from oar.hardware import discover_environment
        try:
            config = load_discovery_config(args.config) if args.config else DiscoveryConfig()
        except (ValueError, OSError) as exc:
            print(f"doctor: invalid configuration: {exc}", file=sys.stderr)
            return 2
        report = discover_environment(config.device)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(f"doctor: {report['scope']}; gpu_claim_eligible={report['gpu_claim_eligible']}; {args.out}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
