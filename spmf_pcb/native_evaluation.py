from pathlib import Path
import hashlib, json, types
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
import ultralytics.engine.validator as native_validator
from .paired_data import build_paired_dataset

CODE = Path(__file__).resolve().parents[1] / "third_party/fusion"
HSI_ROOT = None
EVALUATOR_VERSION = "paper_named_native_evaluator_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(
            value, handle, indent=2, ensure_ascii=False, allow_nan=False, default=str
        )


def array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value)


def tensor_sha(value):
    return hashlib.sha256(array(value).tobytes()).hexdigest()


def raw_tensor(predictions):
    # The locked Detect inference API returns (decoded predictions, feature maps).
    result = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    assert isinstance(result, torch.Tensor), type(result)
    return result


def evaluate_state(seed, name, checkpoint, destination, protocol):
    destination.mkdir(exist_ok=False)
    evidence = {"expected": None, "routes": [], "frames": [], "raw": {}, "records": []}
    model = YOLO(str(checkpoint))
    bn_before = sum(
        isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        for m in model.model.modules()
    )
    assert bn_before > 0, "Expected an unfused source checkpoint"
    handles = []
    if name == "OFF":
        dea = model.model.model[20]
        assert "DEA" in type(dea).__name__, type(dea)

        def off_forward(self, values):
            assert isinstance(values, (tuple, list)) and len(values) == 2
            return values[0]

        dea.forward = types.MethodType(off_forward, dea)

    class Validator(DetectionValidator):
        def build_dataset(self, img_path, mode="val", batch=None):
            return build_paired_dataset(
                self.args,
                img_path,
                batch,
                self.data,
                mode=mode,
                rect=True,
                stride=self.stride,
                hsi_root=HSI_ROOT,
            )

        def preprocess(self, batch):
            assert batch["img"].ndim == 4 and batch["img"].shape[1] == 6
            batch = super().preprocess(batch)
            paired = batch["img"]
            assert paired.dtype == torch.float16 and self.device.type == "cuda"
            rgb, hsi = paired[:, :3], paired[:, 3:]
            pids = [int(Path(path).stem) for path in batch["im_file"]]
            evidence["pids"] = pids
            evidence["expected"] = (rgb, hsi)
            evidence["branch_calls"] = {0: 0, 1: 0}
            for i, pid in enumerate(pids):
                difference = (rgb[i].float() - hsi[i].float()).abs()
                assert bool(
                    torch.any(difference > 0)
                ), f"RGB/HSI input equality on PCB {pid}"
                evidence["frames"].append(
                    {
                        "pcb": pid,
                        "batch_index": self.batch_i,
                        "batch_position": i,
                        "paired_shape": list(paired.shape),
                        "dtype": str(paired.dtype),
                        "RGB_sha256": tensor_sha(rgb[i]),
                        "HSI_sha256": tensor_sha(hsi[i]),
                        "HSI_file": str(HSI_ROOT / f"{pid}.png"),
                        "HSI_file_sha256": sha(HSI_ROOT / f"{pid}.png"),
                        "RGB_equals_HSI": False,
                        "paired_mean_abs_difference": float(difference.mean()),
                        "paired_different_fraction": float(
                            (difference > 0).float().mean()
                        ),
                        "model_input_channels": 3 if name == "RGB" else 6,
                    }
                )
            if name == "RGB":
                batch["img"] = rgb
            return batch

        def postprocess(self, preds):
            decoded = raw_tensor(preds)
            assert decoded.dtype == torch.float16, decoded.dtype
            assert decoded.shape[0] == len(evidence["pids"])
            assert bool(torch.isfinite(decoded).all()), "Nonfinite raw predictions"
            for i, pid in enumerate(evidence["pids"]):
                assert pid not in evidence["raw"], f"Repeated PCB {pid}"
                evidence["raw"][pid] = decoded[i].detach().cpu().clone()
            return super().postprocess(preds)

        def update_metrics(self, preds, batch):
            assert evidence["branch_calls"][0] == 1
            assert evidence["branch_calls"][1] == (0 if name == "RGB" else 1)
            evidence["expected"] = None
            # Observe exactly what the native evaluator accumulates, including
            # its float64 empty arrays on images with no predictions.  Do not
            # repeat matching or manufacture arrays from post-NMS predictions.
            pids = [int(Path(path).stem) for path in batch["im_file"]]
            assert pids == evidence["pids"] and len(preds) == len(pids)
            original_update_stats = self.metrics.update_stats
            captured = 0
            prior_record_count = len(evidence["records"])

            def capture_update_stats(stats):
                nonlocal captured
                assert captured < len(pids), "More than one native stats update per PCB"
                fields = {
                    key: np.array(stats[key], copy=True)
                    for key in ["tp", "conf", "pred_cls", "target_cls"]
                }
                assert fields["tp"].dtype == np.dtype(bool)
                assert fields["tp"].shape == (len(fields["conf"]), 10)
                assert len(fields["pred_cls"]) == len(fields["conf"])
                evidence["records"].append(
                    {
                        "pcb": pids[captured],
                        **{key: values.tolist() for key, values in fields.items()},
                        "dtypes": {
                            key: str(values.dtype) for key, values in fields.items()
                        },
                        "shapes": {
                            key: list(values.shape) for key, values in fields.items()
                        },
                    }
                )
                captured += 1
                return original_update_stats(stats)

            self.metrics.update_stats = capture_update_stats
            try:
                result = super().update_metrics(preds, batch)
            finally:
                self.metrics.update_stats = original_update_stats
            assert captured == len(pids), "Missing native per-PCB stats updates"
            assert len(evidence["records"]) - prior_record_count == len(pids)
            return result

    def branch_hook(branch):
        def check(_module, inputs):
            # The native backend warmup is excluded; all actual dataset batches
            # are checked against their explicit paired loader tensors.
            expected = evidence["expected"]
            if expected is None:
                return
            value = inputs[0]
            assert torch.equal(
                value, expected[branch]
            ), f"Incorrect branch {branch} route"
            evidence["branch_calls"][branch] += 1
            for i, pid in enumerate(evidence["pids"]):
                evidence["routes"].append(
                    {
                        "pcb": pid,
                        "branch": "RGB" if branch == 0 else "HSI",
                        "shape": list(value[i].shape),
                        "dtype": str(value.dtype),
                        "sha256": tensor_sha(value[i]),
                        "exact_expected_input": True,
                    }
                )

        return check

    handles.append(model.model.model[0].register_forward_pre_hook(branch_hook(0)))
    if name != "RGB":
        handles.append(model.model.model[10].register_forward_pre_hook(branch_hook(1)))

    original_backend = native_validator.AutoBackend
    backend_calls = []

    class UnfusedBackend(original_backend):
        def __init__(self, *args, **kwargs):
            # AutoBackend defaults fuse=True; BaseValidator does not specify it.
            kwargs["fuse"] = False
            backend_calls.append({"fuse": False, "fp16_requested": kwargs.get("fp16")})
            super().__init__(*args, **kwargs)
            assert self.fp16 and self.device.type == "cuda"

    native_validator.AutoBackend = UnfusedBackend
    try:
        metrics = model.val(
            data=protocol["data"],
            imgsz=640,
            rect=True,
            batch=protocol["final_evaluation"]["batch"],
            half=True,
            device=protocol.get("device", 0),
            conf=0.001,
            iou=0.7,
            max_det=300,
            split="val",
            augment=False,
            single_cls=False,
            classes=None,
            workers=0,
            cache=False,
            copy_paste=0.0,
            compile=False,
            save_json=False,
            save_txt=False,
            plots=False,
            verbose=False,
            project=str(destination),
            name="native",
            exist_ok=False,
            validator=Validator,
        )
    finally:
        native_validator.AutoBackend = original_backend
        for handle in handles:
            handle.remove()
        # Preserve diagnostic evidence even if an assertion in the native loop fails.
        save(destination / "input_frames.json", evidence["frames"])
        save(destination / "input_routes.json", evidence["routes"])
        save(destination / "stats_records.json", evidence["records"])
        torch.save(evidence["raw"], destination / "raw_predictions_fp16.pt")

    expected_ids = list(protocol["val_ids"])
    assert sorted(evidence["raw"]) == expected_ids
    assert sorted(row["pcb"] for row in evidence["records"]) == expected_ids
    assert len(evidence["routes"]) == len(expected_ids) * (1 if name == "RGB" else 2)
    assert sum(len(row["target_cls"]) for row in evidence["records"]) == 108
    bn_after = sum(
        isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        for m in model.model.modules()
    )
    assert bn_after == bn_before and len(backend_calls) == 1
    row = {
        "P": float(metrics.box.mp),
        "R": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    save(destination / "metrics.json", row)
    save(
        destination / "evaluation_receipt.json",
        {
            "evaluator_version": EVALUATOR_VERSION,
            "seed": seed,
            "state": name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "evaluation": protocol["final_evaluation"],
            "actual_workers": 0,
            "actual_backend_calls": backend_calls,
            "BN_count_before": bn_before,
            "BN_count_after": bn_after,
            "actual_dataset_images": len(evidence["records"]),
            "target_boxes": 108,
            "source": str(CODE),
            "raw_prediction_dtype": "torch.float16",
            "warmup_excluded_from_dataset_input_routes": True,
        },
    )
    del model
    torch.cuda.empty_cache()
    return {
        "metrics": row,
        "raw": evidence["raw"],
        "frames": evidence["frames"],
        "records": evidence["records"],
    }


def compare_off(rgb, off):
    rgb_frames = {row["pcb"]: row for row in rgb["frames"]}
    off_frames = {row["pcb"]: row for row in off["frames"]}
    rows = []
    for pid, reference in rgb["raw"].items():
        observed = off["raw"][pid]
        same_frames = all(
            rgb_frames[pid][key] == off_frames[pid][key]
            for key in [
                "RGB_sha256",
                "HSI_sha256",
                "paired_shape",
                "batch_index",
                "batch_position",
            ]
        )
        same_shape = list(reference.shape) == list(observed.shape)
        same_dtype = reference.dtype == observed.dtype == torch.float16
        exact = same_shape and same_dtype and torch.equal(reference, observed)
        diff = (reference.float() - observed.float()).abs() if same_shape else None
        rows.append(
            {
                "pcb": pid,
                "same_preprocessed_frame_and_batch": same_frames,
                "reference_shape": list(reference.shape),
                "OFF_shape": list(observed.shape),
                "both_FP16": same_dtype,
                "raw_predictions_exact": exact,
                "max_abs_difference": float(diff.max()) if diff is not None else None,
                "different_values": int((diff > 0).sum()) if diff is not None else None,
                "reference_sha256": tensor_sha(reference),
                "OFF_sha256": tensor_sha(observed),
            }
        )
    metrics_equal = rgb["metrics"] == off["metrics"]
    records_equal = sorted(rgb["records"], key=lambda x: x["pcb"]) == sorted(
        off["records"], key=lambda x: x["pcb"]
    )
    passed = all(
        r["same_preprocessed_frame_and_batch"] and r["raw_predictions_exact"]
        for r in rows
    )
    passed = passed and metrics_equal and records_equal
    return {
        "passed": passed,
        "comparison": "Exact native FP16 raw predictions on the same frames and batches",
        "metrics_exact": metrics_equal,
        "stats_records_exact": records_equal,
        "boards": rows,
    }
