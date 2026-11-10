"""Recompute native AP and seed statistics from archived per-board predictions.
This is statistical replay, not a new network evaluation.
"""

from pathlib import Path
import math
import numpy as np
from scipy.stats import t
from .runtime import ROOT, read, save, verify_inputs
from .statistics import (
    load_native_metric,
    records_from_json,
    pooled_metrics,
    same_record_arrays,
)


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    verify_inputs()
    native, _ = load_native_metric(
        ROOT / "third_party/fusion/ultralytics/utils/metrics.py"
    )
    references = read(ROOT / "reference/paired_results.json")
    profiles = {}
    records = {}
    for b in [16, 8]:
        values = []
        for seed in range(3, 17):
            v = {}
            for arm in ["RGB", "SPMF", "OFF"]:
                path = ROOT / f"results/paired/seed_{seed:02d}/batch_{b}/{arm}"
                rows = records_from_json(path / "stats_records.json")
                m = pooled_metrics(rows, native)
                ref = read(path / "metrics.json")
                assert max(abs(m[k] - ref[k]) for k in m) < 1e-7, (b, seed, arm)
                records[seed, b, arm] = {r["pcb"]: r for r in rows}
                v[arm] = m
            assert same_record_arrays(
                list(records[seed, b, "RGB"].values()),
                list(records[seed, b, "OFF"].values()),
            )
            values.append(v)
        metrics = {}
        for k in ["P", "R", "mAP50", "mAP50_95"]:
            rgb = np.array([v["RGB"][k] for v in values])
            spmf = np.array([v["SPMF"][k] for v in values])
            d = spmf - rgb
            h = t.ppf(0.975, 13) * d.std(ddof=1) / math.sqrt(14)
            metrics[k] = {
                "RGB_mean": float(rgb.mean()),
                "SPMF_mean": float(spmf.mean()),
                "difference": float(d.mean()),
                "sample_SD": float(d.std(ddof=1)),
                "seed_CI95": [float(d.mean() - h), float(d.mean() + h)],
                "positive": int((d > 0).sum()),
            }
            ref = references["profiles"][str(b)]["paired"]["RGB"][k]
            assert abs(metrics[k]["difference"] - ref["mean"]) < 1e-7
        profiles[str(b)] = metrics
    draws = read(ROOT / "results/paired/bootstrap_draws.json")
    if args.bootstrap:
        samples = []
        for idx, ids in enumerate(draws):
            sample = []
            for b in [16, 8]:
                ds = []
                for seed in range(3, 17):
                    vals = {
                        a: pooled_metrics(
                            [records[seed, b, a][pid] for pid in ids], native
                        )
                        for a in ["RGB", "SPMF"]
                    }
                    ds.append(
                        [
                            vals["SPMF"][k] - vals["RGB"][k]
                            for k in ["mAP50", "mAP50_95"]
                        ]
                    )
                sample.append(np.mean(ds, axis=0))
            samples.append(sample)
            if (idx + 1) % 100 == 0:
                print("BOOTSTRAP", idx + 1, flush=True)
        samples = np.asarray(samples)
        np.save(out / "bootstrap_differences.npy", samples)
    else:
        samples = np.load(ROOT / "results/paired/reference_bootstrap.npy")[:, :, 0, :]
    for bi, b in enumerate([16, 8]):
        for ki, k in enumerate(["mAP50", "mAP50_95"]):
            ci = np.quantile(samples[:, bi, ki], [0.025, 0.975]).tolist()
            profiles[str(b)][k]["PCB_CI95"] = ci
            assert (
                np.max(
                    np.abs(
                        np.array(ci)
                        - references["profiles"][str(b)]["paired"]["RGB"][k]["PCB_ci95"]
                    )
                )
                < 1e-7
            )
    save(
        out / "completion.json",
        {
            "status": "verified",
            "AP_recomputed_from_records": True,
            "bootstrap_recomputed": args.bootstrap,
            "bootstrap_replicates": 2000,
            "profiles": profiles,
        },
    )
    print("STATISTICAL_REPLAY_VERIFIED", flush=True)
