"""Evaluate manuscript checkpoints against their paired RGB priors."""
from pathlib import Path
from .runtime import ROOT, read, save, activate, require_reference_gpu, dataset_yaml, verify_inputs, verify_dataset

def run(args):
    reference = read(ROOT / f"reference/{args.profile}_results.json")
    seeds = reference["seeds"] if args.seeds == "all" else [int(x) for x in args.seeds.split(",")]
    if args.batch != 16 or not seeds or len(set(seeds)) != len(seeds) or any(s not in reference["seeds"] for s in seeds):
        raise ValueError("Use the published profile's seeds and batch-16 evaluation.")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    activate(settings=out / "settings")
    runtime = require_reference_gpu()
    verify_inputs()
    verify_dataset(args.profile)
    from . import native_evaluation as e
    from .adaptation import check
    from .statistics import records_from_rows, records_from_json, same_record_arrays
    import cv2, torch
    cv2.setNumThreads(1)
    e.HSI_ROOT = ROOT / "data/spectral" / args.profile
    settings = read(ROOT / "configs/spmf.json")
    p = {"data": str(dataset_yaml(out)), "val_ids": list(range(43, 54)),
         "device": args.device, "final_evaluation": read(ROOT / "configs/evaluation.json")}
    p["final_evaluation"]["batch"] = 16
    report = []
    for seed in seeds:
        d = out / f"seed_{seed:02d}"
        d.mkdir()
        weights = ROOT / f"weights/{args.profile}/seed_{seed:02d}"
        rgb = ROOT / "weights/main/rgb_prior.pt" if args.profile == "main" else weights / "rgb.pt"
        spmf = weights / "spmf.pt"
        ck = torch.load(spmf, map_location="cpu")
        check((ck.get("ema") or ck["model"]).model[20], settings)
        del ck
        result = {state: e.evaluate_state(seed, state, path, d / state, p)
                  for state, path in [("RGB", rgb), ("SPMF", spmf), ("OFF", spmf)]}
        off = e.compare_off(result["RGB"], result["OFF"])
        save(d / "OFF_check.json", off)
        if not off["passed"]:
            raise AssertionError("OFF does not exactly recover its paired RGB detector.")
        for state, value in result.items():
            expected = reference["per_seed"][str(seed)][state]
            error = max(abs(value["metrics"][k] - expected[k]) for k in value["metrics"])
            if error > 1e-7:
                raise AssertionError((seed, state, "Reference metric mismatch", error))
            saved = records_from_json(ROOT / f"results/{args.profile}/seed_{seed:02d}/{state}/stats_records.json")
            actual = records_from_rows(value["records"], f"{seed}/{state}")
            if not same_record_arrays(saved, actual):
                raise AssertionError((seed, state, "Reference native detection records differ"))
        row = {"seed": seed, "metrics": {s: r["metrics"] for s, r in result.items()},
               "OFF_exact_RGB": True, "reference_detection_records_exact": True}
        save(d / "completion.json", row)
        report.append(row)
        print("PAIR_VERIFIED", seed, flush=True)
    save(out / "completion.json", {"status": "verified", "profile": args.profile,
         "batch": 16, "runtime": runtime, "rows": report, "new_training": 0})
