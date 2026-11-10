"""Original band scoring/LDA and registration, with explicit fitting population."""

from pathlib import Path
import sys, importlib.util, subprocess, json
from .runtime import ROOT, read, save, sha, verify_inputs


def source(name):
    p = ROOT / "spmf_pcb" / name
    s = importlib.util.spec_from_file_location("spectral_reference", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def verify_projection(directory, profile):
    expected = read(ROOT / f"reference/{profile}_unaligned_sha256.json")
    actual = {p.name: sha(p) for p in Path(directory).glob("*.png")}
    assert actual == expected, "Projected images differ from the reference inputs"
    return actual


def run(args):
    if args.command == "verify-raw":
        verify_inputs(include_assets=False)
        raw = Path(args.raw_root).resolve()
        expected = read(ROOT / "reference/raw_input_sha256.json")
        for name, h in expected.items():
            assert sha(raw / name) == h, name
        save(args.output, {"status": "verified", "raw_files": len(expected)})
        return
    verify_inputs(include_assets=False)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw = Path(args.raw_root).resolve()
    if args.command == "register":
        images = {p.name: sha(p) for p in Path(args.spectral_dir).glob("*.png")}
        profiles = [
            profile
            for profile in ["main", "paired"]
            if images == read(ROOT / f"reference/{profile}_unaligned_sha256.json")
        ]
        assert (
            len(profiles) == 1
        ), "Registration requires one of the two verified spectral projections"
        cmd = [
            sys.executable,
            "-B",
            str(ROOT / "spmf_pcb/spatial_registration.py"),
            "--root",
            str(raw),
            "--params",
            str(ROOT / "configs/manual_alignment.json"),
            "--current_pseudo_dir",
            str(Path(args.spectral_dir).resolve()),
            "--refine_ecc",
            "--out_dir",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        summary = read(out / "apply_alignment_summary.json")
        assert summary["num_ok"] == 53 and summary["num_fail"] == 0
        actual = {p.name: sha(p) for p in (out / "pseudo").glob("*.png")}
        prefix = "data/spectral/" + profiles[0] + "/"
        assert actual == {
            Path(k).name: h for k, h in read(ROOT / "reference/dataset_sha256.json").items()
            if k.startswith(prefix)
        }, "Registered images differ from the reference"
        save(
            out / "completion.json",
            {
                "status": "complete",
                "images": actual,
                "image_only_ECC": True,
                "profile": profiles[0],
                "reference_images_exact": True,
            },
        )
        return
    m = source("spectral_construction.py")
    if args.command == "apply-spectral":
        import numpy as np, joblib

        tr = Path(args.transform)
        if not tr.is_absolute():
            tr = ROOT / tr
        meta = read(tr / "metadata.json")
        lda = joblib.load(tr / "lda.joblib")
        for pid in range(1, 54):
            cube = m.spectral.envi.open(
                str(m.find_envi_hdr(raw / "HSI" / f"pcb{pid}"))
            ).open_memmap()
            mask = m.load_mask_auto(m.find_file_by_id(raw / "HSI/PCB_Masks", pid))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            z = m.make_tensor_lda_transform(
                cube,
                (mask > 0).astype(np.uint8),
                meta["picked_bands_0_based"],
                lda,
                np.asarray(meta["lda_qlo"], dtype=np.float32),
                np.asarray(meta["lda_qhi"], dtype=np.float32),
                bg=meta["bg"],
            )
            m.save_png_3ch(z, out / f"{pid:02d}.png")
        profile = "main" if meta["use_all_53"] else "paired"
        actual = verify_projection(out, profile)
        save(
            out / "completion.json",
            {
                "status": "complete",
                "fit_performed": False,
                "semantic_label_access": False,
                "images": actual,
                "reference_images_exact": True,
            },
        )
        return
    cfg = read(ROOT / f"configs/spectral_{args.profile}.json")
    m.TRAIN_IDS = cfg["fit_ids"]
    original = m.find_file_by_id
    seen = []
    input_hashes = {}

    def find(folder, pid, *a, **kw):
        folder = Path(folder)
        if folder.name in ["Monoseg_masks", "General_masks"]:
            assert pid in cfg["fit_ids"], "Validation semantic label access denied"
            seen.append(pid)
            if not folder.exists() and folder.name == "Monoseg_masks":
                folder = folder.with_name("Monoseg_mask")
        p = original(folder, pid, *a, **kw)
        input_hashes[str(p.relative_to(raw))] = sha(p)
        return p

    m.find_file_by_id = find
    import joblib

    Base = m.LinearDiscriminantAnalysis

    class Capture(Base):
        def fit_transform(self, X, y=None):
            z = super().fit_transform(X, y)
            fitted = Base(**self.get_params())
            fitted.__dict__.update(self.__dict__)
            joblib.dump(fitted, out / "lda.joblib")
            return z

    m.LinearDiscriminantAnalysis = Capture
    argv = ["spectral_construction", "--root", str(raw), "--out_dir", str(out)]
    if args.profile == "main":
        argv.append("--use_all_53")
    for k in [
        "label_type",
        "output_mode",
        "save_format",
        "drop_head",
        "drop_tail",
        "k",
        "min_gap",
        "corr_thresh",
        "corr_sample_rows",
        "bg",
        "qlo",
        "qhi",
        "max_per_class_per_pcb",
        "max_bg_per_pcb",
        "cv_folds",
    ]:
        argv += ["--" + k, str(cfg[k])]
    sys.argv = argv
    m.main()
    meta = read(out / "selected_bands.json")
    assert sorted(set(seen)) == cfg["fit_ids"]
    reference = read(ROOT / f"transforms/{args.profile}/metadata.json")
    for key in ["picked_bands_0_based", "lda_qlo", "lda_qhi"]:
        assert meta[key] == reference[key], key
    verify_projection(out, args.profile)
    save(out / "metadata.json", meta)
    save(
        out / "completion.json",
        {
            "status": "complete",
            "profile": args.profile,
            "fit_ids": cfg["fit_ids"],
            "validation_semantic_labels_read": args.profile == "main",
            "read_mask_hashes": input_hashes,
            "lda_sha256": sha(out / "lda.joblib"),
        },
    )
