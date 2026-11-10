"""Build the experiment inputs from the official PCB-Vision download.

No raw or processed dataset images are distributed with this repository.
Existing matching inputs are reused; different inputs are never overwritten.
"""
from pathlib import Path
from types import SimpleNamespace
import shutil
import tempfile

from .runtime import ROOT, read, save, sha, verify_inputs, verify_dataset


def checked_copy(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if sha(source) != expected:
        raise RuntimeError("Unexpected input content: " + str(source))
    if destination.exists():
        if sha(destination) != expected:
            raise RuntimeError("Refusing to overwrite a different input: " + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def build_timing(raw, out, expected):
    import cv2
    import numpy as np
    import spectral
    from .timing_input import normalize_u8, apply_transform

    cfg = read(ROOT / "configs/timing_input.json")
    params = read(ROOT / "configs/manual_alignment.json")
    spectral.settings.envi_support_nonlowercase_params = True
    for pid in cfg["ids"]:
        rel = "data/spectral/efficiency/{}.png".format(pid)
        dest = out / Path(rel).relative_to("data")
        if dest.exists():
            if sha(dest) != expected[rel]:
                raise RuntimeError("Different timing input already exists: " + str(dest))
            continue
        cube = spectral.envi.open(str(raw / "HSI" / ("pcb" + str(pid)) / ("pcb{}.hdr".format(pid)))).read_bands(cfg["bands_zero_based"])
        rgb = cv2.imdecode(np.fromfile(raw / "RGB" / (str(pid) + ".jpg"), dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError("Cannot decode RGB PCB " + str(pid))
        result = cv2.merge([apply_transform(normalize_u8(cube[:, :, i]), params[str(pid)], rgb.shape) for i in range(3)])
        ok, encoded = cv2.imencode(".png", result)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        import hashlib
        if hashlib.sha256(encoded.tobytes()).hexdigest() != expected[rel]:
            raise RuntimeError("Timing input differs from the fixed reference: " + str(pid))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as f:
            f.write(encoded.tobytes())
        print("TIMING_INPUT", pid, flush=True)


def run(args):
    verify_inputs(include_assets=False)
    raw = Path(args.raw_root).resolve()
    if not (raw / "RGB").is_dir() or not (raw / "HSI").is_dir():
        raise ValueError("--raw-root must be the extracted directory containing RGB/ and HSI/.")
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    if out == raw or out in raw.parents or raw in out.parents:
        raise ValueError("Use separate raw-data and prepared-data directories.")
    expected = read(ROOT / "reference/dataset_sha256.json")
    profiles = ["main", "paired", "efficiency"] if args.profile == "all" else [args.profile]
    split = read(ROOT / "configs/dataset_split.json")
    for subset in ("train", "val"):
        for pid in split[subset]:
            rel = "data/images/{}/{}.jpg".format(subset, pid)
            checked_copy(raw / "RGB" / (str(pid) + ".jpg"), out / Path(rel).relative_to("data"), expected[rel])
            rel = "data/labels/{}/{}.txt".format(subset, pid)
            checked_copy(ROOT / "configs/annotations" / subset / (str(pid) + ".txt"), out / Path(rel).relative_to("data"), expected[rel])
    for profile in profiles:
        prefix = "data/spectral/" + profile + "/"
        items = {k: h for k, h in expected.items() if k.startswith(prefix)}
        if not items:
            raise ValueError("Missing dataset profile manifest")
        present = [out / Path(k).relative_to("data") for k in items]
        for p, h in zip(present, items.values()):
            if p.exists() and sha(p) != h:
                raise RuntimeError("Refusing to replace different generated input: " + str(p))
        if all(p.is_file() for p in present):
            print("DATA_PROFILE_READY", profile, flush=True)
            continue
        if profile == "efficiency":
            build_timing(raw, out, expected)
        else:
            from .preprocess import run as preprocess
            work = Path(tempfile.mkdtemp(prefix="build_" + profile + "_", dir=str(out)))
            projected, registered = work / "projected", work / "registered"
            preprocess(SimpleNamespace(command="apply-spectral", raw_root=str(raw), transform=str(ROOT / "transforms" / profile), output=str(projected)))
            preprocess(SimpleNamespace(command="register", raw_root=str(raw), spectral_dir=str(projected), output=str(registered)))
            for k, h in items.items():
                checked_copy(registered / "pseudo" / Path(k).name, out / Path(k).relative_to("data"), h)
        verify_dataset(profile, data_root=out)
        print("DATA_PROFILE_READY", profile, flush=True)
    save(out / "preparation.json", {"profiles": profiles, "raw_root": str(raw), "all_requested_inputs_match": True})
