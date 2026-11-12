from pathlib import Path
import os, sys, json, hashlib

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf8",
    )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def activate(kind="fusion", settings=None):
    sys.dont_write_bytecode = True
    code = ROOT / "third_party" / kind
    sys.path.insert(0, str(code))
    os.environ.update(
        YOLO_AUTOINSTALL="false",
        WANDB_DISABLED="true",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    os.environ["YOLO_CONFIG_DIR"] = str(settings or ROOT / "outputs/settings")
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    import ultralytics

    assert (
        code.resolve() in Path(ultralytics.__file__).resolve().parents
    ), "A different Ultralytics installation was imported"
    if kind == "fusion":
        import ultralytics.nn.tasks as tasks
        from ultralytics.nn.modules.DEA import DEA

        tasks.SPMF = DEA  # identical class object: parser and checkpoint serialization stay compatible
    return code


def require_reference_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Reference evaluation requires CUDA FP16; CPU is not a numerically equivalent replacement."
        )
    if torch.__version__ != "2.0.1+cu117" or torch.backends.cudnn.version() != 8801:
        raise RuntimeError(
            "Reference runtime requires torch 2.0.1+cu117 and cuDNN 8.8.1 (8801). Set the cuDNN library path before Python starts."
        )
    torch.set_num_threads(2)
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "GPU": torch.cuda.get_device_name(0),
    }


def dataset_yaml(destination, data_root=None):
    import yaml

    data = Path(data_root or ROOT / "data").resolve()
    path = Path(destination) / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(data),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "ic", 1: "capacitor", 2: "connector"},
            },
            sort_keys=False,
        ),
        encoding="utf8",
    )
    return path


def verify_inputs(include_assets=True):
    manifest = read(ROOT / "release_manifest.json")
    assets = read(ROOT / "reference/assets_sha256.json")
    checked = 0
    for rel, h in manifest.items():
        if not include_assets and rel in assets:
            continue
        p = ROOT / rel
        if not p.is_file() or sha(p) != h:
            hint = " Extract PCB-SPMF-assets-v1.1.0.zip into the repository root." if rel in assets else ""
            raise RuntimeError("Release input missing or changed: " + rel + hint)
        checked += 1
    return checked


def verify_dataset(profile=None, data_root=None):
    data = Path(data_root or ROOT / "data")
    expected = read(ROOT / "reference/dataset_sha256.json")
    prefixes = ["data/images/", "data/labels/"]
    if profile:
        prefixes.append("data/spectral/" + profile + "/")
    count = 0
    for rel, h in expected.items():
        if not any(rel.startswith(prefix) for prefix in prefixes):
            continue
        p = data / Path(rel).relative_to("data")
        if not p.is_file() or sha(p) != h:
            raise RuntimeError("Dataset input missing or changed: " + rel + ". Run prepare-data using the official PCB-Vision download.")
        count += 1
    return count
