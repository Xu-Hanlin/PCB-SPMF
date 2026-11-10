"""The recorded Table 11 timing boundary and alternating order; latency is remeasured."""

from pathlib import Path
import time, statistics, hashlib
from .runtime import (
    ROOT,
    read,
    save,
    sha,
    activate,
    require_reference_gpu,
    verify_inputs,
    verify_dataset,
)


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    activate(settings=out / "settings")
    env = require_reference_gpu()
    verify_inputs()
    verify_dataset("efficiency")
    import torch, numpy as np, cv2
    from ultralytics import YOLO

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def transform(im):
        h, w = im.shape[:2]
        ratio = 640 / max(h, w)
        nw, nh = round(w * ratio), round(h * ratio)
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dh, dw = (640 - nh) / 2, (640 - nw) / 2
        im = cv2.copyMakeBorder(
            im,
            round(dh - 0.1),
            round(dh + 0.1),
            round(dw - 0.1),
            round(dw + 0.1),
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))

    inputs = []
    for pid in range(43, 54):
        parts = [
            torch.from_numpy(transform(cv2.imread(str(p)))).cuda().half() / 255
            for p in [
                ROOT / f"data/images/val/{pid}.jpg",
                ROOT / f"data/spectral/efficiency/{pid}.png",
            ]
        ]
        assert not torch.equal(*parts)
        inputs.append(torch.cat(parts).unsqueeze(0))
    hashes = [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in inputs]
    models = {}
    names = ["joint", "frozen"]
    for name, path in [
        ("joint", ROOT / "weights/efficiency/spmf_joint.pt"),
        ("frozen", ROOT / "weights/main/spmf_selected.pt"),
    ]:
        reference = read(ROOT / f"results/efficiency/{name}.json")
        assert sha(path) == reference["checkpoint_sha256"]
        assert hashes == reference["input_tensor_hashes"]
        model = YOLO(str(path)).model.cuda().float().eval()
        model.fuse()
        model.half().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        models[name] = model
    samples = {k: [] for k in names}
    with torch.inference_mode():
        for k in range(55):
            for n in names:
                models[n](inputs[k % 11])
        for rnd in range(6):
            for n in names if rnd % 2 == 0 else list(reversed(names)):
                for x in inputs:
                    models[n](x)
                for rep in range(5):
                    for i, x in enumerate(inputs):
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        models[n](x)
                        torch.cuda.synchronize()
                        samples[n].append(
                            {
                                "round": rnd,
                                "repeat": rep,
                                "pcb": 43 + i,
                                "ms": (time.perf_counter() - t) * 1000,
                            }
                        )
    result = {
        n: {
            "samples": samples[n],
            "FPS": 1000 / statistics.mean(x["ms"] for x in samples[n]),
        }
        for n in names
    }
    save(
        out / "completion.json",
        {
            "status": "complete",
            "runtime": env,
            "results": result,
            "input_shape": [1, 6, 640, 640],
            "network_forward_only": True,
            "preprocessing_transfer_NMS_excluded": True,
            "historical_FPS": {"frozen": 52.85919435197814, "joint": 52.554},
            "timing_is_not_bitwise_reproducible": True,
        },
    )
