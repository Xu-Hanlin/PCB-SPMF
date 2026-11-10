import re
import json
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import spectral

spectral.settings.envi_support_nonlowercase_params = True


# =========================
# utils
# =========================
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def get_file_id(name: str):
    digits = "".join(filter(str.isdigit, name))
    return str(int(digits)) if digits else None


def find_file_by_id(folder: Path, pid: str, exts=None, prefer_hdr=True):
    """
    Robust id matching:
    - supports mono1 / mono1.hdr / gen1_mask.png / pcb1_mask.png ...
    - prefers .hdr over extensionless ENVI data file
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    cands = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue

        name_for_match = p.stem.lower() if p.suffix else p.name.lower()
        if re.search(rf"(\D|^){int(pid)}(\D|$)", name_for_match):
            cands.append(p)

    if not cands:
        raise FileNotFoundError(f"File for id={pid} not found in {folder}")

    def sort_key(p: Path):
        # 1) prefer .hdr
        hdr_rank = 0 if (prefer_hdr and p.suffix.lower() == ".hdr") else 1
        # 2) then prefer normal image files
        img_rank = (
            0
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            else 1
        )
        # 3) then natural name sort
        return (hdr_rank, img_rank, natural_key(p.name))

    cands = sorted(cands, key=sort_key)
    return cands[0]


def load_image_auto(path: Path):
    """
    Supports:
    - ENVI .hdr
    - extensionless ENVI data file (auto redirect to sibling .hdr if exists)
    - normal images (.png/.jpg/...)
    """
    path = Path(path)

    # case 1: explicit .hdr
    if path.suffix.lower() == ".hdr":
        img = spectral.envi.open(str(path))
        arr = np.asarray(img.open_memmap())
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return arr

    # case 2: extensionless ENVI data file -> try sibling .hdr
    if path.suffix == "":
        hdr_path = path.with_suffix(".hdr")
        if hdr_path.exists():
            img = spectral.envi.open(str(hdr_path))
            arr = np.asarray(img.open_memmap())
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            return arr

    # case 3: normal image
    return np.array(Image.open(path))


def save_image_auto(arr: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray(arr).save(out_path)
    else:
        Image.fromarray(arr).save(out_path)


def normalize_u8(img):
    img = img.astype(np.float32)
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip(img, p2, p98)
    if p98 - p2 < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - p2) / (p98 - p2) * 255.0).astype(np.uint8)


def preprocess_for_alignment(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return mag.astype(np.float32)


def ensure_u8_3ch(arr: np.ndarray):
    """
    For ECC refinement visualization/registration only.
    """
    if arr.ndim == 2:
        arr = normalize_u8(arr)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if arr.ndim == 3 and arr.shape[2] == 3:
        if arr.dtype != np.uint8:
            out = np.zeros_like(arr, dtype=np.uint8)
            for c in range(3):
                out[:, :, c] = normalize_u8(arr[:, :, c])
            return out
        return arr

    # fallback
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    out = np.zeros(arr.shape[:2] + (3,), dtype=np.uint8)
    for c in range(min(3, arr.shape[2])):
        out[:, :, c] = normalize_u8(arr[:, :, c])
    return out


# =========================
# manual transform
# =========================
def apply_manual_transform(img, params, target_shape, is_mask=False):
    """
    Reuse the exact semantics of your manual tuner:
      scale -> rotate+perspective -> paste with translation onto RGB-sized canvas
    """
    h_t, w_t = target_shape[:2]
    h_s, w_s = img.shape[:2]

    scale_x = params["sx"] / 1000.0
    scale_y = params["sy"] / 1000.0
    rotation = (params.get("rot", 500) - 500) / 10.0
    dx = params["dx"] - 2000
    dy = params["dy"] - 2000
    px = (params.get("px", 500) - 500) / 10000.0
    py = (params.get("py", 500) - 500) / 10000.0

    if scale_x <= 0:
        scale_x = 0.001
    if scale_y <= 0:
        scale_y = 0.001

    interp_resize = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    interp_warp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    # 1) resize
    new_w = max(1, int(round(w_s * scale_x)))
    new_h = max(1, int(round(h_s * scale_y)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp_resize)

    # 2) rotation + tiny perspective
    center = (new_w / 2.0, new_h / 2.0)
    M_rot = cv2.getRotationMatrix2D(center, rotation, 1.0)

    H = np.eye(3, dtype=np.float32)
    H[:2] = M_rot
    H[2, 0] = px
    H[2, 1] = py

    if resized.ndim == 2:
        warped = cv2.warpPerspective(
            resized,
            H,
            (new_w, new_h),
            flags=interp_warp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        warped = cv2.warpPerspective(
            resized,
            H,
            (new_w, new_h),
            flags=interp_warp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    # 3) paste onto target canvas
    if warped.ndim == 2:
        canvas = np.zeros((h_t, w_t), dtype=warped.dtype)
    else:
        canvas = np.zeros((h_t, w_t, warped.shape[2]), dtype=warped.dtype)

    x_start = int(round(dx))
    y_start = int(round(dy))

    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = new_w, new_h
    dst_x1, dst_y1 = x_start, y_start
    dst_x2, dst_y2 = x_start + new_w, y_start + new_h

    if dst_x1 < 0:
        src_x1 -= dst_x1
        dst_x1 = 0
    if dst_y1 < 0:
        src_y1 -= dst_y1
        dst_y1 = 0
    if dst_x2 > w_t:
        src_x2 -= dst_x2 - w_t
        dst_x2 = w_t
    if dst_y2 > h_t:
        src_y2 -= dst_y2 - h_t
        dst_y2 = h_t

    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = warped[src_y1:src_y2, src_x1:src_x2]

    return canvas


# =========================
# optional ECC refinement
# =========================
def refine_ecc_affine(rgb_bgr: np.ndarray, coarse_bgr: np.ndarray):
    """
    Small affine refinement after manual coarse alignment.
    Returns warp that maps coarse -> rgb.
    """
    tpl = preprocess_for_alignment(rgb_bgr)
    mov = preprocess_for_alignment(coarse_bgr)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)

    cc, warp = cv2.findTransformECC(
        tpl, mov, warp, cv2.MOTION_AFFINE, criteria, None, 5
    )
    return float(cc), warp


def apply_affine(img, warp2x3, out_shape, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border = 0 if img.ndim == 2 else (0, 0, 0)

    return cv2.warpAffine(
        img,
        warp2x3,
        (out_shape[1], out_shape[0]),
        flags=interp + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


# =========================
# main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="PCBDataset root")
    ap.add_argument(
        "--params",
        type=str,
        default="alignment_params.json",
        help="Path to alignment_params.json",
    )

    ap.add_argument(
        "--current_pseudo_dir",
        type=str,
        required=True,
        help="Current pseudoRGB folder to be aligned",
    )
    ap.add_argument("--rgb_dir", type=str, default="RGB")

    ap.add_argument("--align_monoseg", action="store_true")
    ap.add_argument("--align_general", action="store_true")
    ap.add_argument("--align_pcbmask", action="store_true")

    ap.add_argument(
        "--refine_ecc",
        action="store_true",
        help="After manual coarse alignment, do ECC affine refinement",
    )
    ap.add_argument("--out_dir", type=str, default="registered_spectral")
    args = ap.parse_args()

    root = Path(args.root)
    params_path = (
        root / args.params if not Path(args.params).is_absolute() else Path(args.params)
    )

    current_pseudo_dir = (
        root / args.current_pseudo_dir
        if not Path(args.current_pseudo_dir).is_absolute()
        else Path(args.current_pseudo_dir)
    )
    rgb_dir = (
        root / args.rgb_dir
        if not Path(args.rgb_dir).is_absolute()
        else Path(args.rgb_dir)
    )

    hsi_dir = root / "HSI"
    monoseg_dir = hsi_dir / "Monoseg_masks"
    general_dir = hsi_dir / "General_masks"
    pcbmask_dir = hsi_dir / "PCB_Masks"

    out_root = root / args.out_dir
    out_pseudo = out_root / "pseudo"
    out_monoseg = out_root / "Monoseg_masks"
    out_general = out_root / "General_masks"
    out_pcbmask = out_root / "PCB_Masks"
    out_overlay = out_root / "overlays"
    out_root.mkdir(parents=True, exist_ok=True)
    out_pseudo.mkdir(parents=True, exist_ok=True)
    out_overlay.mkdir(parents=True, exist_ok=True)

    if args.align_monoseg:
        out_monoseg.mkdir(parents=True, exist_ok=True)
    if args.align_general:
        out_general.mkdir(parents=True, exist_ok=True)
    if args.align_pcbmask:
        out_pcbmask.mkdir(parents=True, exist_ok=True)

    with open(params_path, "r", encoding="utf-8") as f:
        params_all = json.load(f)

    rgb_files = (
        list(rgb_dir.glob("*.jpg"))
        + list(rgb_dir.glob("*.jpeg"))
        + list(rgb_dir.glob("*.png"))
    )
    rgb_map = {get_file_id(p.stem): p for p in rgb_files if get_file_id(p.stem)}

    pseudo_files = (
        list(current_pseudo_dir.glob("*.png"))
        + list(current_pseudo_dir.glob("*.jpg"))
        + list(current_pseudo_dir.glob("*.jpeg"))
    )
    pseudo_map = {get_file_id(p.stem): p for p in pseudo_files if get_file_id(p.stem)}

    common_ids = sorted(
        list(set(rgb_map.keys()) & set(pseudo_map.keys()) & set(params_all.keys())),
        key=int,
    )

    print(f"Found {len(common_ids)} images to align.")

    rows = []

    for i, fid in enumerate(common_ids, 1):
        print(f"[{i:02d}/{len(common_ids):02d}] ID={fid}")

        row = {"id": fid, "status": "OK", "ecc_cc": None}

        try:
            rgb_path = rgb_map[fid]
            pseudo_path = pseudo_map[fid]
            params = params_all[fid]

            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"Failed to read RGB: {rgb_path}")

            pseudo_arr = np.array(Image.open(pseudo_path))
            if pseudo_arr.ndim == 2:
                pseudo_arr = cv2.cvtColor(pseudo_arr, cv2.COLOR_GRAY2RGB)

            # manual coarse alignment to RGB canvas
            pseudo_coarse = apply_manual_transform(
                pseudo_arr, params, rgb.shape, is_mask=False
            )

            # optional ECC refinement
            warp = None
            pseudo_final = pseudo_coarse.copy()
            if args.refine_ecc:
                rgb_bgr = rgb
                coarse_bgr = cv2.cvtColor(pseudo_coarse, cv2.COLOR_RGB2BGR)
                cc, warp = refine_ecc_affine(rgb_bgr, coarse_bgr)
                row["ecc_cc"] = cc
                pseudo_final_bgr = apply_affine(
                    coarse_bgr, warp, rgb.shape[:2], is_mask=False
                )
                pseudo_final = cv2.cvtColor(pseudo_final_bgr, cv2.COLOR_BGR2RGB)

            # save pseudo
            save_image_auto(pseudo_final.astype(np.uint8), out_pseudo / f"{fid}.png")

            # overlay
            pseudo_bgr_vis = cv2.cvtColor(
                pseudo_final.astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            overlay = cv2.addWeighted(rgb, 0.6, pseudo_bgr_vis, 0.4, 0)
            cv2.imwrite(str(out_overlay / f"{fid}.png"), overlay)

            # align masks using same transforms
            if args.align_monoseg:
                src = find_file_by_id(monoseg_dir, fid, prefer_hdr=True)
                arr = load_image_auto(src).astype(np.uint8)
                arr = apply_manual_transform(arr, params, rgb.shape, is_mask=True)
                if warp is not None:
                    arr = apply_affine(arr, warp, rgb.shape[:2], is_mask=True)
                save_image_auto(arr, out_monoseg / f"{fid}.png")

            if args.align_general:
                src = find_file_by_id(general_dir, fid, prefer_hdr=True)
                arr = load_image_auto(src).astype(np.uint8)
                arr = apply_manual_transform(arr, params, rgb.shape, is_mask=True)
                if warp is not None:
                    arr = apply_affine(arr, warp, rgb.shape[:2], is_mask=True)
                save_image_auto(arr, out_general / f"{fid}.png")

            if args.align_pcbmask:
                src = find_file_by_id(pcbmask_dir, fid, prefer_hdr=True)
                arr = load_image_auto(src).astype(np.uint8)
                arr = apply_manual_transform(arr, params, rgb.shape, is_mask=True)
                if warp is not None:
                    arr = apply_affine(arr, warp, rgb.shape[:2], is_mask=True)
                save_image_auto(arr, out_pcbmask / f"{fid}.png")

            row["rgb_file"] = rgb_path.name
            row["pseudo_file"] = pseudo_path.name

        except Exception as e:
            row["status"] = "FAIL"
            row["error"] = repr(e)
            print(f"[WARN] ID={fid} failed: {e}")

        rows.append(row)

    # save log
    csv_path = out_root / "apply_alignment_log.csv"
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_total": len(rows),
        "num_ok": sum(1 for r in rows if r.get("status") == "OK"),
        "num_fail": sum(1 for r in rows if r.get("status") == "FAIL"),
        "refine_ecc": bool(args.refine_ecc),
        "current_pseudo_dir": str(current_pseudo_dir),
        "out_dir": str(out_root),
    }
    with open(out_root / "apply_alignment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Saved:")
    print(" -", out_pseudo)
    if args.align_monoseg:
        print(" -", out_monoseg)
    if args.align_general:
        print(" -", out_general)
    if args.align_pcbmask:
        print(" -", out_pcbmask)
    print(" -", out_overlay)
    print(" -", csv_path)
    print(" -", out_root / "apply_alignment_summary.json")


if __name__ == "__main__":
    main()
