"""Original RGB-to-dual-branch tensor mapping for a new, explicitly named run.

Reference reproduction uses the archived initial states. Calling this function
creates a new initialization; it does not certify historical checkpoint equality.
"""

from pathlib import Path
from .runtime import ROOT, activate, save, sha


def initialize_from_rgb(rgb_checkpoint, output, seed):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    activate(settings=output / "settings")
    import torch
    from ultralytics import YOLO

    torch.manual_seed(seed)
    checkpoint = torch.load(rgb_checkpoint, map_location="cpu")
    rgb_model = checkpoint.get("ema") or checkpoint["model"]
    rgb = rgb_model.state_dict()
    model = YOLO(str(ROOT / "configs/rgb_hsi_spmf_p3.yaml")).model
    initial = model.state_dict()
    detect = lambda m: max(
        i for i, layer in enumerate(m.model) if type(layer).__name__ == "Detect"
    )
    offset = detect(model) - detect(rgb_model)
    for key, value in rgb.items():
        if key in initial and initial[key].shape == value.shape:
            initial[key] = value.detach().clone()
    for key, value in rgb.items():
        parts = key.split(".")
        if len(parts) < 3 or parts[0] != "model":
            continue
        target = "model." + str(int(parts[1]) + offset) + "." + ".".join(parts[2:])
        if target in initial and initial[target].shape == value.shape:
            initial[target] = value.detach().clone()
    model.load_state_dict(initial, strict=True)
    torch.save(initial, output / "initial_state.pt")
    torch.save(
        {"model": model, "ema": None, "updates": 0}, output / "initial_checkpoint.pt"
    )
    save(
        output / "completion.json",
        {
            "seed": seed,
            "RGB_sha256": sha(Path(rgb_checkpoint)),
            "state_sha256": sha(output / "initial_state.pt"),
            "head_offset": offset,
            "new_initialization_not_historical_retraining_proof": True,
        },
    )
