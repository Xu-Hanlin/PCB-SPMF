"""Isolated paired adapter preserving the native RGB training path.

No edits to the historical repository. Model construction still uses ch=3 per
backbone; only the image batch is explicitly RGB(3)+HSI(3). Missing HSI fails.
"""

import math, random
from pathlib import Path
import cv2, numpy as np, torch
import ultralytics.data.augment as aug
from ultralytics.data.dataset import YOLODataset


class SixChannelCV2:
    """Use the same spatial operation and matrix for each three-channel image."""

    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        fn = getattr(self.original, name)
        if name not in {"resize", "warpAffine", "warpPerspective", "copyMakeBorder"}:
            return fn

        def call(image, *args, **kwargs):
            if image.ndim == 3 and image.shape[2] == 6:
                return np.concatenate(
                    [
                        fn(np.ascontiguousarray(image[:, :, :3]), *args, **kwargs),
                        fn(np.ascontiguousarray(image[:, :, 3:]), *args, **kwargs),
                    ],
                    axis=2,
                )
            return fn(image, *args, **kwargs)

        return call


def install_spatial_adapter():
    if not isinstance(aug.cv2, SixChannelCV2):
        aug.cv2 = SixChannelCV2(aug.cv2)


class RGBPhotometric:
    def __init__(self, original):
        self.original = original
        if isinstance(original, aug.Albumentations) and original.transform is not None:
            assert (
                not original.contains_spatial
            ), "HSI geometry must not silently diverge"

    def __call__(self, labels):
        image = labels["img"]
        assert image.ndim == 3 and image.shape[2] == 6
        hsi = image[:, :, 3:].copy()
        labels["img"] = image[:, :, :3].copy()
        labels = self.original(labels)
        assert labels["img"].shape == hsi.shape
        labels["img"] = np.concatenate([labels["img"], hsi], axis=2)
        return labels


class PairedFormat(aug.Format):
    def _format_img(self, img):
        assert img.ndim == 3 and img.shape[2] == 6
        rgb = img[:, :, :3].transpose(2, 0, 1)
        hsi = img[:, :, 3:].transpose(2, 0, 1)
        # Native Format consumes exactly one draw, including bgr=0.
        if random.uniform(0, 1) > self.bgr:
            rgb = rgb[::-1]
            hsi = hsi[::-1]
        return torch.from_numpy(
            np.ascontiguousarray(np.concatenate([rgb, hsi], axis=0))
        )


def adapt_transforms(t):
    if isinstance(t, aug.Compose):
        t.transforms = [adapt_transforms(x) for x in t.transforms]
    elif isinstance(t, (aug.RandomHSV, aug.Albumentations)):
        return RGBPhotometric(t)
    elif isinstance(t, aug.Format):
        t.__class__ = PairedFormat
    return t


class PairedNativeDataset(YOLODataset):
    def __init__(self, *args, hsi_root=None, probe_duplicate=False, **kwargs):
        assert kwargs.get("cache") in [
            None,
            False,
        ], "This adapter intentionally refuses ambiguous cached image formats"
        assert (
            kwargs.get("data", {}).get("channels", 3) == 3
        ), "Each backbone must retain its original three-channel construction"
        self.hsi_root = Path(hsi_root) if hsi_root is not None else None
        self.probe_duplicate = probe_duplicate
        assert self.hsi_root is not None or probe_duplicate
        install_spatial_adapter()
        super().__init__(*args, **kwargs)
        assert not any(
            Path(f).with_suffix(".npy").exists() for f in self.im_files
        ), "Raw native .npy image caches are not accepted"
        if not probe_duplicate:
            for f in self.im_files:
                assert (self.hsi_root / (Path(f).stem + ".png")).is_file(), f

    def build_transforms(self, hyp=None):
        assert (
            hyp.copy_paste == 0
        ), "Instance copy-paste is outside this locked historical detection protocol"
        return adapt_transforms(super().build_transforms(hyp))

    def load_image(self, i, rect_mode=True):
        rgb, hw0, hw = super().load_image(i, rect_mode)
        assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
        if self.probe_duplicate:
            hsi = rgb.copy()
        else:
            p = self.hsi_root / (Path(self.im_files[i]).stem + ".png")
            hsi = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            assert (
                hsi is not None
                and hsi.shape == tuple(hw0) + (3,)
                and hsi.dtype == np.uint8
            ), (str(p), None if hsi is None else (hsi.shape, str(hsi.dtype)), hw0)
            r = self.imgsz / max(hw0)
            if r != 1:
                hsi = cv2.resize(
                    hsi,
                    (hw[1], hw[0]),
                    interpolation=(
                        cv2.INTER_LINEAR if self.augment or r > 1 else cv2.INTER_AREA
                    ),
                )
            assert hsi.shape == rgb.shape
        return np.concatenate([rgb, hsi], axis=2), hw0, hw


def build_paired_dataset(
    cfg,
    img_path,
    batch,
    data,
    mode="train",
    rect=False,
    stride=32,
    hsi_root=None,
    probe_duplicate=False,
):
    return PairedNativeDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=mode + ": ",
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        hsi_root=hsi_root,
        probe_duplicate=probe_duplicate,
    )
