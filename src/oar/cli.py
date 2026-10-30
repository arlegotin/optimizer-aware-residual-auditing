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
    args = parser.parse_args(argv)
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
