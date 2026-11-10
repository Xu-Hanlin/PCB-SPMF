"""Stable paper-facing commands. Run each command in its own Python process."""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="PCB RGB--HSI SPMF: locked preprocessing, training and evaluation"
    )
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--output", default="outputs/package_verification.json")
    v.add_argument("--code-only", action="store_true")
    d = sub.add_parser("prepare-data")
    d.add_argument("--raw-root", required=True)
    d.add_argument("--profile", choices=["main", "paired", "efficiency", "all"], default="all")
    d.add_argument("--output", default="data")
    v = sub.add_parser("verify-raw")
    v.add_argument("--raw-root", required=True)
    v.add_argument("--output", default="outputs/raw_verification.json")
    e = sub.add_parser("evaluate")
    e.add_argument("--profile", choices=["main", "paired"], required=True)
    e.add_argument("--seeds", default="all")
    e.add_argument("--batch", type=int, choices=[16, 8], default=16)
    e.add_argument("--device", default="0")
    e.add_argument("--output", required=True)
    s = sub.add_parser("replay-statistics")
    s.add_argument("--bootstrap", action="store_true")
    s.add_argument("--output", required=True)
    f = sub.add_parser("fit-spectral")
    f.add_argument("--profile", choices=["main", "paired"], required=True)
    f.add_argument("--raw-root", required=True)
    f.add_argument("--output", required=True)
    a = sub.add_parser("apply-spectral")
    a.add_argument("--raw-root", required=True)
    a.add_argument("--transform", default="transforms/paired")
    a.add_argument("--output", required=True)
    a = sub.add_parser("register")
    a.add_argument("--raw-root", required=True)
    a.add_argument("--spectral-dir", required=True)
    a.add_argument("--output", required=True)
    t = sub.add_parser("train-rgb")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--device", default="0")
    t.add_argument("--output", required=True)
    t = sub.add_parser("train-frozen")
    t.add_argument("--profile", choices=["main", "paired"], default="paired")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--device", default="0")
    t.add_argument("--output", required=True)
    t.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify trainer setup and one loss forward; no optimizer updates",
    )
    b = sub.add_parser("benchmark")
    b.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "verify":
        from spmf_pcb.runtime import verify_inputs, save

        n = verify_inputs(include_assets=not args.code_only)
        save(args.output, {"status": "verified", "files": n})
        print("HASHES_VERIFIED", n)
    elif args.command == "prepare-data":
        from spmf_pcb.prepare_data import run

        run(args)
    elif args.command == "evaluate":
        from spmf_pcb.evaluate import run

        run(args)
    elif args.command == "replay-statistics":
        from spmf_pcb.replay import run

        run(args)
    elif args.command in ["fit-spectral", "apply-spectral", "register", "verify-raw"]:
        from spmf_pcb.preprocess import run

        run(args)
    elif args.command.startswith("train-"):
        from spmf_pcb.training import run

        run(args)
    elif args.command == "benchmark":
        from spmf_pcb.benchmark import run

        run(args)


if __name__ == "__main__":
    main()
