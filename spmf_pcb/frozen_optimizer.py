import argparse
import re
import torch
import torch.nn as nn
from ultralytics import YOLO

DEFAULT_MODEL = "configs/rgb_hsi_spmf_p3.yaml"
DEFAULT_PRETRAINED = "initial_checkpoint.pt"
DEFAULT_DATA = "dataset.yaml"

# Stage A: only train HSI backbone (10~19) + SPMF (20)
TRAINABLE_IDXS = set(range(10, 21))
_num_pat = re.compile(r"(?:^|\.)model\.(\d+)\.")

# custom hyperparams kept OUTSIDE Ultralytics cfg
CUSTOM = {
    "gate_lr": 2e-3,
    "base_lr": 3e-4,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
}


def _extract_idx(name: str):
    m = _num_pat.search(name)
    return int(m.group(1)) if m else None


def _is_gate_like(name: str) -> bool:
    keys = [
        ".alpha",
        ".beta",
        ".rel_",
        ".warp_",
        ".spatial_gate",
        ".force_",
        ".mix",
        ".gate",
    ]
    return any(k in name for k in keys)


def apply_freeze(det_model: nn.Module, verbose: bool = False):
    modules = det_model.model
    for idx, m in enumerate(modules):
        flag = idx in TRAINABLE_IDXS
        for p in m.parameters(recurse=True):
            p.requires_grad = flag
        if flag:
            m.train()
        else:
            m.eval()
            for sub in m.modules():
                if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                    sub.eval()

    # safety pass by name
    for name, p in det_model.named_parameters():
        idx = _extract_idx(name)
        if idx is not None:
            p.requires_grad = idx in TRAINABLE_IDXS

    if verbose:
        n_train = sum(p.numel() for p in det_model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in det_model.parameters())
        print(f"[freeze] trainable idxs = {sorted(TRAINABLE_IDXS)}")
        print(
            f"[freeze] trainable params = {n_train}/{n_all} ({100.0*n_train/max(n_all,1):.2f}%)"
        )


def rebuild_optimizer_and_scheduler(
    trainer, base_lr=3e-4, gate_lr=2e-3, wd=1e-4, optimizer_name="AdamW"
):
    model = trainer.model
    gate_params, other_params = [], []
    gate_names, other_names = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        idx = _extract_idx(name)
        if idx is None or idx not in TRAINABLE_IDXS:
            continue
        if _is_gate_like(name):
            gate_params.append(p)
            gate_names.append(name)
        else:
            other_params.append(p)
            other_names.append(name)

    if not other_params and not gate_params:
        raise RuntimeError(
            "No trainable params found in TRAINABLE_IDXS. Check model naming / freeze logic."
        )

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "weight_decay": wd})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": gate_lr, "weight_decay": 0.0})

    if optimizer_name.lower() == "adamw":
        trainer.optimizer = torch.optim.AdamW(
            param_groups, lr=base_lr, betas=(0.9, 0.999)
        )
    else:
        trainer.optimizer = torch.optim.SGD(
            param_groups, lr=base_lr, momentum=0.937, nesterov=True
        )

    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lr_lambda=lambda epoch: 1.0
    )

    print(
        f"[opt] rebuilt optimizer={optimizer_name} base_lr={base_lr} gate_lr={gate_lr} wd={wd}"
    )
    print(f"[opt] other_trainable_tensors = {len(other_params)}")
    print(f"[opt] gate_trainable_tensors  = {len(gate_params)}")
    print(f"[opt] sample other params: {other_names[:8]}")
    print(f"[opt] sample gate params : {gate_names[:8]}")


def cb_pretrain_routine_end(trainer):
    apply_freeze(trainer.model, verbose=True)
    rebuild_optimizer_and_scheduler(
        trainer,
        base_lr=float(CUSTOM["base_lr"]),
        gate_lr=float(CUSTOM["gate_lr"]),
        wd=float(CUSTOM["weight_decay"]),
        optimizer_name=str(CUSTOM["optimizer"]),
    )


def cb_train_epoch_start(trainer):
    apply_freeze(trainer.model, verbose=False)


def cb_train_batch_start(trainer):
    apply_freeze(trainer.model, verbose=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default="SPMF")
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=3e-4)
    p.add_argument("--gate_lr", type=float, default=2e-3)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=float, default=3.0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--project", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--close_mosaic", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    CUSTOM["gate_lr"] = float(args.gate_lr)
    CUSTOM["base_lr"] = float(args.lr0)
    CUSTOM["weight_decay"] = float(args.weight_decay)
    CUSTOM["optimizer"] = str(args.optimizer)

    model = YOLO(args.model)
    model.add_callback("on_pretrain_routine_end", cb_pretrain_routine_end)
    model.add_callback("on_train_epoch_start", cb_train_epoch_start)
    model.add_callback("on_train_batch_start", cb_train_batch_start)

    # IMPORTANT: only pass valid Ultralytics args here
    train_kwargs = dict(
        data=args.data,
        pretrained=args.pretrained,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        seed=args.seed,
        close_mosaic=args.close_mosaic,
    )
    if args.project:
        train_kwargs["project"] = args.project

    print("[info] Stage A hard-freeze + hard-optimizer training")
    print("[info] only modules 10~20 are trainable; gate params use higher LR")
    print(f"[info] model      = {args.model}")
    print(f"[info] pretrained = {args.pretrained}")
    print(f"[info] custom gate_lr = {CUSTOM['gate_lr']}")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
