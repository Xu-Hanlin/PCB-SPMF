# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for Ultralytics YOLO models
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.
    """

    def __init__(self, *args, data: Optional[dict] = None, task: str = "detect", **kwargs):
        """Initialize the YOLODataset."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)
        self.im_files2 = None
        self._dual = False

        k2 = None
        p = str(self.img_path).replace("\\", "/")
        if "train" in p and isinstance(self.data, dict) and self.data.get("train2", None) is not None:
            k2 = "train2"
        if "val" in p and isinstance(self.data, dict) and self.data.get("val2", None) is not None:
            k2 = "val2"

        if k2 is not None:
            from pathlib import Path
            p2 = self.data[k2]
            if isinstance(p2, (list, tuple)):
                p2 = p2[0]
            root = Path(self.data.get("path", Path(self.img_path).parent))
            p2 = Path(p2)
            if not p2.is_absolute():
                p2 = (root / p2).resolve()

            im_files2_raw = self.get_img_files(str(p2))
            map2 = {Path(x).stem: x for x in im_files2_raw}

            aligned = []
            missing = 0
            for f in self.im_files:
                s = Path(f).stem
                g = map2.get(s, None)
                if g is None:
                    missing += 1
                    continue
                aligned.append(g)

            if missing != 0 or len(aligned) != len(self.im_files):
                raise FileNotFoundError(f"dual stream mismatch: missing={missing}, rgb={len(self.im_files)}, msi={len(aligned)}")

            self.im_files2 = aligned
            self._dual = True



        # Prefill mosaic buffer to avoid empty buffer crash on first batch
        # Some Ultralytics versions expect dataset.buffer non-empty when Mosaic is enabled.
        try:
            n = len(getattr(self, 'im_files', []))
            if not hasattr(self, 'buffer') or len(getattr(self, 'buffer', [])) == 0:
                # use a set for cheap membership, Mosaic will cast to list()
                self.buffer = set(range(n))
        except Exception:
            pass


        # -----------------------------
    # Multimodal load_image override
    # - If data['channels'] >= 6: pack RGB (first 3 ch) + MSI/HSI (last 3 ch) into a single 6-channel tensor.
    # - Optional ablations controlled by env vars (set them in train.py BEFORE importing ultralytics):
    #     os.environ["ZERO_MSI"] = "1"      # force MSI/HSI channels to 0
    #     os.environ["MSI_SCALE"] = "0.0"   # scale MSI/HSI channels (e.g., 0.2)
    #     os.environ["MM_DEBUG"] = "1"      # print one-time statistics for sanity-check
    # -----------------------------
    def _mm_infer_msi_path(self, rgb_path: str) -> str:
        """
        Infer the paired MSI/HSI image path from an RGB image path.

        Strategy:
          1) Replace folder name: train -> msi_train, val -> msi_val (and a few common variants).
          2) Keep the same filename stem.
          3) Try multiple suffixes (original suffix first, then common alternatives).
          4) If nothing exists, return the most likely path (same suffix as RGB) so the caller can handle missing files.
        """
        from pathlib import Path

        p = Path(rgb_path)

        # Replace common RGB folders with MSI folders
        parts = list(p.parts)
        repl = {
            "train": "msi_train",
            "val": "msi_val",
            "images_train": "msi_train",
            "images_val": "msi_val",
            "rgb_train": "msi_train",
            "rgb_val": "msi_val",
            "train_rgb": "msi_train",
            "val_rgb": "msi_val",
        }
        parts2 = [repl.get(x, x) for x in parts]
        p_base = Path(*parts2)

        # Candidate suffixes (try original first)
        orig_suf = p_base.suffix.lower()
        suf_candidates = []
        if orig_suf:
            suf_candidates.append(orig_suf)
        suf_candidates += [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

        base_no_ext = p_base.with_suffix("")
        for suf in suf_candidates:
            cand = base_no_ext.with_suffix(suf)
            if cand.exists():
                return str(cand)

        # Fallback: keep same suffix as RGB if present, otherwise use .png
        if orig_suf:
            return str(p_base)
        return str(base_no_ext.with_suffix(".png"))

    def _mm_read_msi(self, msi_path: str, hw: Tuple[int, int], dtype) -> np.ndarray:
        """Read MSI/HSI image, convert to 3-channel HWC, resize to hw, and cast dtype. Missing -> zeros."""
        h, w = hw
        msi = cv2.imread(msi_path, cv2.IMREAD_UNCHANGED)
        if msi is None:
            return np.zeros((h, w, 3), dtype=dtype)

        if msi.ndim == 2:
            msi = cv2.cvtColor(msi, cv2.COLOR_GRAY2BGR)
        elif msi.ndim == 3:
            # drop alpha or extra bands if any
            if msi.shape[2] > 3:
                msi = msi[..., :3]

        if msi.shape[:2] != (h, w):
            msi = cv2.resize(msi, (w, h), interpolation=cv2.INTER_LINEAR)

        if msi.dtype != dtype:
            msi = msi.astype(dtype, copy=False)

        return msi

    def _mm_apply_switch(self, im: np.ndarray) -> np.ndarray:
        """Apply ZERO_MSI / MSI_SCALE switches in-place on a packed 6-channel image."""
        if im.ndim != 3 or im.shape[2] < 6:
            return im

        # cache config once per dataset instance
        if not hasattr(self, "_mm_cfg"):
            zero = os.getenv("ZERO_MSI", "0").lower() in ("1", "true", "yes", "y", "on")
            debug = os.getenv("MM_DEBUG", "0").lower() in ("1", "true", "yes", "y", "on")
            try:
                scale = float(os.getenv("MSI_SCALE", "1.0"))
            except Exception:
                scale = 1.0
            self._mm_cfg = {"zero": zero, "scale": scale, "debug": debug}

        cfg = self._mm_cfg
        if cfg["scale"] != 1.0:
            m = im[..., 3:6].astype(np.float32) * float(cfg["scale"])
            im[..., 3:6] = m.astype(im.dtype)

        if cfg["zero"]:
            im[..., 3:6] = 0

        if cfg["debug"] and not getattr(self, "_mm_printed", False):
            self._mm_printed = True
            m = im[..., 3:6]
            # avoid non-ascii chars to prevent Windows console encoding issues
            print(f"[MM_DEBUG] ZERO_MSI={cfg['zero']} MSI_SCALE={cfg['scale']} msi(min/max/mean)={m.min():.3f}/{m.max():.3f}/{m.mean():.3f} shape={m.shape} dtype={m.dtype}", flush=True)
            if cfg["zero"] and (m.max() != 0 or m.min() != 0):
                raise RuntimeError("ZERO_MSI is enabled but MSI channels are not all zero. Please check your pipeline.")

        return im

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, (h0, w0), hw_resized)."""
        im = self.imgs[i] if hasattr(self, "imgs") else None
        f = self.im_files[i]
        fn = self.npy_files[i] if hasattr(self, "npy_files") else Path(f).with_suffix(".npy")

        if im is None:  # not cached in RAM
            if fn.exists():
                im = np.load(fn)
            else:
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found: {f}")
                if im.ndim == 2:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

            # pack multimodal if requested
            want_msi = getattr(self, "channels", 3) >= 6
            if want_msi and (im.ndim == 3) and (im.shape[2] < 6):
                msi_path = self._mm_infer_msi_path(f)
                msi = self._mm_read_msi(msi_path, im.shape[:2], im.dtype)
                im = np.concatenate((im, msi), axis=2)

            im = self._mm_apply_switch(im)

            h0, w0 = im.shape[:2]
            r = self.imgsz / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)), interpolation=interp)

            return im, (h0, w0), im.shape[:2]

        # cached path
        im = self._mm_apply_switch(im)
        return im, self.im_hw0[i], self.im_hw[i]

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """Cache dataset labels."""
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect.")

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],
                            "bboxes": lb[:, 1:],
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> List[dict]:
        """Return dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(f"No valid images found in {cache_path}.")
        self.im_files = [lb["im_file"] for lb in labels]

        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning("Box and segment counts should be equal... only boxes will be used.")
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}.")
        return labels

    def build_transforms(self, hyp: Optional[dict] = None) -> Compose:
        """Build and append transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    def close_mosaic(self, hyp: dict) -> None:
        """Disable mosaic, copy_paste, mixup and cutmix augmentations."""
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label: dict) -> dict:
        """Update label format for different tasks."""
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """Collate data samples into batches."""
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """Dataset class for multi-modal support."""

    def __init__(self, *args, data: Optional[dict] = None, task: str = "detect", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: dict) -> dict:
        labels = super().update_labels_info(label)
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp: Optional[dict] = None) -> Compose:
        transforms = super().build_transforms(hyp)
        if self.augment:
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}

    @property
    def category_freq(self):
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> List[str]:
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class GroundingDataset(YOLODataset):
    """Dataset class for grounding tasks."""

    def __init__(self, *args, task: str = "detect", json_file: str = "", max_samples: int = 80, **kwargs):
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        self.max_samples = max_samples
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path: str) -> list:
        return []

    def verify_labels(self, labels: List[Dict[str, Any]]) -> None:
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }
        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' has {instance_count} instances, expected {count}."
                return
        LOGGER.warning(f"Skipping instance count verification for unrecognized dataset '{self.json_file}'")

    def cache_labels(self, path: Path = Path("./labels.cache")) -> Dict[str, Any]:
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0]: t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
            lb = np.array(lb, dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> List[dict]:
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.json_file)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, _ = self.cache_labels(cache_path), False
        [cache.pop(k) for k in ("hash", "version")]
        labels = cache["labels"]
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp: Optional[dict] = None) -> Compose:
        transforms = super().build_transforms(hyp)
        if self.augment:
            transform = RandomLoadText(
                max_samples=min(self.max_samples, 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> List[str]:
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOConcatDataset(ConcatDataset):
    """Dataset as a concatenation of multiple datasets."""

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: dict) -> None:
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


class ClassificationDataset:
    """Dataset class for image classification tasks."""

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        import torchvision

        if TORCHVISION_0_18:
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        if augment and args.fraction < 1.0:
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"
        self.samples = self.verify_images()
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]
        scale = (1.0 - args.scale, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )
    

    def __getitem__(self, i: int) -> dict:
        f, j, fn, im = self.samples[i]
        if self.cache_ram:
            if im is None:
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:
            im = cv2.imread(f)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        if getattr(self, "_dual", False) and self.im_files2 is not None:
            import cv2
            import numpy as np
            from pathlib import Path

            f2 = self.im_files2[i]
            im2 = cv2.imread(f2, cv2.IMREAD_UNCHANGED)
            if im2 is None:
                raise FileNotFoundError(f"read failed: {f2}")

            if im2.ndim == 2:
                im2 = np.stack([im2, im2, im2], axis=2)
            elif im2.shape[2] == 1:
                im2 = np.repeat(im2, 3, axis=2)
            elif im2.shape[2] > 3:
                im2 = im2[:, :, :3]

            # labels["img"] is torch.Tensor (3,H,W) after format
            # We need to apply same resize/letterbox to im2 before formatting.
            # The simplest safe way here: reuse final shape from labels["img"].
            import torch
            im2 = cv2.resize(im2, (labels["img"].shape[2], labels["img"].shape[1]), interpolation=cv2.INTER_LINEAR)
            im2 = im2[:, :, ::-1].copy()  # BGR->RGB
            im2 = torch.from_numpy(im2).permute(2, 0, 1).contiguous()
            im2 = im2.to(labels["img"].dtype) / 255.0

            labels["img"] = [labels["img"], im2]

        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        return len(self.samples)

    def verify_images(self) -> List[tuple]:
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")

        try:
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)
            cache = load_dataset_cache_file(path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([x[0] for x in self.samples])
            nf, nc, n, samples = cache.pop("results")
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """Semantic Segmentation Dataset."""

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()