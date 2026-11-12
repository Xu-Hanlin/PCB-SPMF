"""Fixed RGB pretraining and paired frozen adaptation; existing results are never overwritten."""

from pathlib import Path
import json, os, shutil
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
    activate("rgb" if args.command == "train-rgb" else "fusion", out / "settings")
    runtime = require_reference_gpu()
    verify_inputs()
    verify_dataset(None if args.command == "train-rgb" else args.profile)
    import torch
    from ultralytics import YOLO

    data = dataset_yaml(out)
    if args.command == "train-rgb":
        cfg = read(ROOT / "configs/rgb_pretraining.json")
        cfg.update(
            data=str(data),
            model=str(ROOT / "third_party/rgb/yolov8n.pt"),
            seed=args.seed,
            project=str(out),
            name="RGB",
            device=args.device,
        )
        if not Path(cfg["model"]).is_file():
            raise FileNotFoundError("Exact pretrained YOLOv8n file missing")
        save(out / "requested_arguments.json", cfg)
        model = YOLO(cfg["model"])
        model.train(**cfg)
        best = out / "RGB/weights/best.pt"
        save(
            out / "completion.json",
            {
                "status": "complete",
                "runtime": runtime,
                "epochs_requested": 500,
                "seed": args.seed,
                "best_sha256": sha(best),
                "exact_retraining_equivalence_not_assumed": True,
            },
        )
        return
    from ultralytics.models.yolo.detect import DetectionTrainer
    from .paired_data import build_paired_dataset
    from . import frozen_optimizer as f

    if args.profile == "main" and args.seed not in (0, 1, 2):
        raise ValueError("Main adaptation seeds are 0, 1, and 2 (one shared RGB prior).")
    if args.profile == "paired" and args.seed not in range(3, 17):
        raise ValueError("The paired seeds are 3--16.")

    class DryRunComplete(Exception):
        pass

    # Native AMP check resolves this official helper checkpoint against the cwd.
    # Copy the exact helper into the new run, never download a substitute implicitly.
    shutil.copy2(ROOT / "third_party/fusion/yolo11n.pt", out / "yolo11n.pt")
    os.chdir(out)

    class PairedTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            stride = max(
                int(self.model.stride.max()) if self.model is not None else 32, 32
            )
            return build_paired_dataset(
                self.args,
                img_path,
                batch,
                self.data,
                mode=mode,
                rect=mode == "val",
                stride=stride,
                hsi_root=ROOT / "data/spectral" / args.profile,
            )

    from .adaptation import configure, check
    settings = read(ROOT / "configs/spmf.json")
    f.CUSTOM.update(base_lr=settings["base_lr"], gate_lr=settings["gate_lr"],
                    weight_decay=settings["base_weight_decay"], optimizer="AdamW")
    seed = args.seed
    initial_path = (
        ROOT / "weights/main/initial_state.pt"
        if args.profile == "main"
        else ROOT / f"weights/paired/seed_{seed:02d}/initial_state.pt"
    )
    initial = torch.load(initial_path, map_location="cpu")
    model = YOLO(str(ROOT / "configs/rgb_hsi_spmf_p3.yaml"))
    model.model.load_state_dict(initial, strict=True)
    init = out / "initial_checkpoint.pt"
    torch.save({"model": model.model, "ema": None, "updates": 0}, init)
    observer = {"batches": 0, "initial": None, "handles": [], "active": False}

    def frozen(k):
        return int(k.split(".")[1]) not in range(10, 21)

    def setup(t):
        configure(t.model.model[20], settings)
        configure(t.ema.ema.model[20], settings)
        save(out / "spmf_settings.json", settings)
        now = t.model.state_dict()
        assert set(now) == set(initial) and all(
            torch.equal(now[k].cpu(), v.to(now[k].dtype)) for k, v in initial.items()
        )
        observer["initial"] = {
            k: v.detach().cpu().clone() for k, v in now.items() if frozen(k)
        }
        assert (
            len(t.train_loader.dataset) == 42
            and len(t.test_loader.dataset) == 11
            and len(t.train_loader) == 6
        )

        def route(which):
            def hook(_m, inp):
                if observer["active"]:
                    if which == 0:
                        observer["rgb"] = inp[0].detach()
                    else:
                        assert not torch.equal(
                            observer["rgb"], inp[0]
                        ), "RGB/RGB route detected"

            return hook

        observer["handles"] = [
            t.model.model[0].register_forward_pre_hook(route(0)),
            t.model.model[10].register_forward_pre_hook(route(1)),
        ]
        save(
            out / "runtime_setup.json",
            {
                "runtime": runtime,
                "seed": seed,
                "train_batches_per_epoch": len(t.train_loader),
                "initial_state_exact": True,
                "args": vars(t.args),
            },
        )
        if args.dry_run:
            batch = t.preprocess_batch(next(iter(t.train_loader)))
            assert batch["img"].shape[1] == 6
            observer["active"] = True
            with torch.no_grad():
                loss, items = t.model(batch)
            observer["active"] = False
            assert torch.isfinite(loss).all()
            save(
                out / "completion.json",
                {
                    "status": "verified",
                    "initial_state_exact": True,
                    "shape": list(batch["img"].shape),
                    "loss_finite": True,
                    "true_paired_route": True,
                    "optimizer_updates": 0,
                    "dry_run": True,
                },
            )
            raise DryRunComplete()

    def start(t):
        observer["batches"] += 1
        observer["active"] = True
        assert all(
            not p.requires_grad for k, p in t.model.named_parameters() if frozen(k)
        )
        assert all(
            not m.training
            for i, layer in enumerate(t.model.model)
            if i not in range(10, 21)
            for m in layer.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        )

    def end(t):
        observer["active"] = False

    def epoch(t):
        current = t.model.state_dict()
        assert all(
            torch.equal(current[k].cpu(), v) for k, v in observer["initial"].items()
        )
        with (out / "epoch_audit.jsonl").open("a") as h:
            h.write(
                json.dumps(
                    {
                        "epoch": t.epoch + 1,
                        "frozen_model_exact": True,
                        "batches": observer["batches"],
                        "metrics": t.metrics,
                    }
                )
                + "\n"
            )

    for name, fn in [
        ("on_pretrain_routine_end", f.cb_pretrain_routine_end),
        ("on_train_epoch_start", f.cb_train_epoch_start),
        ("on_train_batch_start", f.cb_train_batch_start),
        ("on_pretrain_routine_end", setup),
        ("on_train_batch_start", start),
        ("on_train_batch_end", end),
        ("on_fit_epoch_end", epoch),
    ]:
        model.add_callback(name, fn)
    cfg = read(
        ROOT
        / (
            "configs/main_training_recipe.json"
            if args.profile == "main"
            else "configs/frozen_adaptation.json"
        )
    )
    for k in ["model", "gate_lr"]:
        cfg.pop(k, None)
    cfg.update(
        data=str(data),
        pretrained=str(init),
        seed=seed,
        project=str(out),
        name="SPMF",
        device=args.device,
    )
    try:
        model.train(trainer=PairedTrainer, **cfg)
    except DryRunComplete:
        print("TRAINER_STARTUP_VERIFIED")
        return
    assert observer["batches"] == 720
    best = out / "SPMF/weights/best.pt"
    ck = torch.load(best, map_location="cpu")
    saved_model = ck.get("ema") or ck["model"]
    check(saved_model.model[20], settings)
    sd = saved_model.state_dict()
    assert all(
        torch.equal(sd[k], v.to(sd[k].dtype)) for k, v in observer["initial"].items()
    )
    save(
        out / "completion.json",
        {
            "status": "complete",
            "seed": seed,
            "epochs": 120,
            "microbatches": 720,
            "best_sha256": sha(best),
            "RGB_parameters_and_BN_preserved": True,
            "exact_retraining_equivalence_not_assumed": True,
        },
    )
