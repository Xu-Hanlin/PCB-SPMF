from pathlib import Path
from .runtime import (
    ROOT,
    read,
    save,
    sha,
    activate,
    require_reference_gpu,
    dataset_yaml,
    verify_inputs,
    verify_dataset,
)


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    activate(settings=out / "settings")
    runtime = require_reference_gpu()
    verify_inputs()
    verify_dataset(args.profile)
    from . import native_evaluation as e
    import numpy as np, torch
    import cv2

    cv2.setNumThreads(1)
    e.HSI_ROOT = ROOT / "data/spectral" / args.profile
    p = {
        "data": str(dataset_yaml(out)),
        "val_ids": list(range(43, 54)),
        "device": args.device,
        "final_evaluation": read(ROOT / "configs/evaluation.json"),
    }
    p["final_evaluation"]["batch"] = args.batch
    seeds = (
        list(range(3, 17))
        if args.seeds == "all"
        else [int(x) for x in args.seeds.split(",")]
    )
    if args.profile == "main":
        seeds = [None]
    report = []
    for seed in seeds:
        if seed is None:
            d = out / "selected"
            weights = ROOT / "weights/main"
            rgb = weights / "rgb_prior.pt"
            spmf = weights / "spmf_selected.pt"
            ref = {
                state: read(ROOT / f"results/main/selected/{old}/metrics.json")
                for state, old in [("SPMF", "ON"), ("OFF", "OFF")]
            }
            ref["RGB"] = ref["OFF"]
            ref = {
                k: {"P": v["P"], "R": v["R"], "mAP50": v["AP50"], "mAP50_95": v["AP95"]}
                for k, v in ref.items()
            }
            if args.batch != 16:
                raise ValueError(
                    "The main-checkpoint reference has only batch 16; do not substitute a new geometry."
                )
        else:
            if seed not in range(3, 17):
                raise ValueError("Locked paired seeds are 3--16.")
            d = out / f"seed_{seed:02d}"
            weights = ROOT / f"weights/paired/seed_{seed:02d}"
            rgb = weights / "rgb.pt"
            spmf = weights / "spmf.pt"
            ref = read(ROOT / "reference/paired_results.json")["profiles"][
                str(args.batch)
            ]["per_seed"][str(seed)]
        d.mkdir()
        result = {
            state: e.evaluate_state(seed, state, ck, d / state, p)
            for state, ck in [("RGB", rgb), ("SPMF", spmf), ("OFF", spmf)]
        }
        off = e.compare_off(result["RGB"], result["OFF"])
        save(d / "OFF_check.json", off)
        if not off["passed"]:
            raise AssertionError(
                "OFF does not exactly recover the matched RGB detector."
            )
        for state, r in result.items():
            error = max(abs(r["metrics"][k] - ref[state][k]) for k in r["metrics"])
            if error > 1e-7:
                raise AssertionError((seed, state, "Reference metric mismatch", error))
            if seed is not None:
                old = torch.load(
                    ROOT
                    / f"results/paired/seed_{seed:02d}/batch_{args.batch}/{state}/raw_predictions_fp16.pt",
                    map_location="cpu",
                )
            else:
                old = torch.load(
                    ROOT
                    / f'results/main/selected/{"ON" if state=="SPMF" else "OFF"}/raw_predictions.pt',
                    map_location="cpu",
                )
            if not all(torch.equal(old[k], v) for k, v in r["raw"].items()):
                raise AssertionError((seed, state, "Raw predictions differ"))
        report.append(
            {
                "seed": seed,
                "metrics": {s: r["metrics"] for s, r in result.items()},
                "OFF_exact_RGB": True,
                "reference_raw_predictions_exact": True,
            }
        )
        save(d / "completion.json", report[-1])
        print("PAIR_VERIFIED", seed, flush=True)
    save(
        out / "completion.json",
        {
            "status": "verified",
            "profile": args.profile,
            "batch": args.batch,
            "runtime": runtime,
            "rows": report,
            "new_training": 0,
        },
    )
