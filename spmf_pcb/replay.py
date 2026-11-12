"""Statistical replay of the main and independent-RGB paired evaluations."""
from pathlib import Path
import numpy as np
from scipy.stats import t
from .runtime import ROOT, read, save, verify_inputs
from .statistics import load_native_metric, records_from_json, pooled_metrics, same_record_arrays

def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    verify_inputs()
    native, _ = load_native_metric(ROOT / "third_party/fusion/ultralytics/utils/metrics.py")
    profiles, records, refs = {}, {}, {}
    names = ["main", "paired"]
    metrics = ["P", "R", "mAP50", "mAP50_95"]
    for profile in names:
        ref = refs[profile] = read(ROOT / f"reference/{profile}_results.json")
        values = []
        for seed in ref["seeds"]:
            row = {}
            for state in ["RGB", "SPMF", "OFF"]:
                rows = records_from_json(ROOT / f"results/{profile}/seed_{seed:02d}/{state}/stats_records.json")
                v = pooled_metrics(rows, native)
                assert max(abs(v[k] - ref["per_seed"][str(seed)][state][k]) for k in metrics) < 1e-7
                row[state] = v
                records[profile, seed, state] = {r["pcb"]: r for r in rows}
            assert same_record_arrays(list(records[profile, seed, "RGB"].values()), list(records[profile, seed, "OFF"].values()))
            values.append(row)
        profiles[profile] = {}
        for k in metrics:
            rgb = np.array([v["RGB"][k] for v in values])
            spmf = np.array([v["SPMF"][k] for v in values])
            delta = spmf - rgb
            half = t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
            expected = ref["statistics"][k]
            seed_ci = [float(delta.mean()-half), float(delta.mean()+half)]
            assert abs(100*delta.mean()-expected["mean_delta_pp"]) < 1e-7
            assert np.max(np.abs(100*np.array(seed_ci)-[expected["seed_CI95_low_pp"], expected["seed_CI95_high_pp"]])) < 1e-7
            profiles[profile][k] = {"RGB_mean":float(rgb.mean()), "SPMF_mean":float(spmf.mean()),
                "RGB_SD":float(rgb.std(ddof=1)), "SPMF_SD":float(spmf.std(ddof=1)),
                "difference":float(delta.mean()), "difference_SD":float(delta.std(ddof=1)),
                "seed_CI95":seed_ci, "positive":int((delta>0).sum())}
    draws = read(ROOT / "results/bootstrap_draws.json")
    assert len(draws) == 2000
    if args.bootstrap:
        samples = []
        for idx, ids in enumerate(draws):
            item = []
            for profile in names:
                diffs = []
                for seed in refs[profile]["seeds"]:
                    v = {s: pooled_metrics([records[profile,seed,s][pid] for pid in ids],native) for s in ["RGB","SPMF"]}
                    diffs.append([v["SPMF"][k]-v["RGB"][k] for k in metrics[2:]])
                item.append(np.mean(diffs,axis=0))
            samples.append(item)
            if (idx+1)%100 == 0: print("BOOTSTRAP",idx+1,flush=True)
        samples = np.array(samples)
        np.save(out / "bootstrap_differences.npy",samples)
    else:
        samples = np.load(ROOT / "results/reference_bootstrap.npy")
    assert samples.shape == (2000,2,2)
    for pi,profile in enumerate(names):
        for ki,k in enumerate(metrics[2:]):
            ci = np.quantile(samples[:,pi,ki],[.025,.975])
            expected = refs[profile]["statistics"][k]
            assert np.max(np.abs(100*ci-[expected["PCB_CI95_low_pp"],expected["PCB_CI95_high_pp"]])) < 1e-7
            profiles[profile][k]["PCB_CI95"] = ci.tolist()
    save(out / "completion.json", {"status":"verified", "AP_recomputed_from_records":True,
         "bootstrap_recomputed":args.bootstrap, "bootstrap_replicates":2000, "profiles":profiles,
         "main_seeds_share_one_RGB_prior":True, "paired_seeds_have_independent_RGB_priors":True})
    print("STATISTICAL_REPLAY_VERIFIED",flush=True)
