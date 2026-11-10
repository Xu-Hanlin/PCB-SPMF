import re
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import spectral

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score

spectral.settings.envi_support_nonlowercase_params = True

# Optional: paper train split (only used if --use_all_53 is NOT set)
TRAIN_IDS = list(range(1, 43))
EPS = 1e-8


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_envi_hdr(pcb_dir: Path) -> Path:
    hdrs = sorted(pcb_dir.glob("*.hdr"), key=lambda p: natural_key(p.name))
    if not hdrs:
        raise FileNotFoundError(f"No .hdr found in {pcb_dir}")
    return hdrs[0]


def load_mask_auto(path: Path) -> np.ndarray:
    """Load ENVI .hdr mask or normal image mask."""
    if path.suffix.lower() == ".hdr":
        img = spectral.envi.open(str(path))
        arr = np.asarray(img.open_memmap())
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return arr
    return np.array(Image.open(path))


def rgb_mask_to_labels(rgb: np.ndarray) -> np.ndarray:
    """Nearest-color RGB mask -> {0,1,2,3}."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb_mask_to_labels expects RGB image")
    rgb3 = rgb[:, :, :3].astype(np.int16)
    ref = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.int16)
    d = ((rgb3[:, :, None, :] - ref[None, None, :, :]) ** 2).sum(axis=-1)
    return d.argmin(axis=2).astype(np.uint8)


def ensure_label_ids(lab: np.ndarray) -> np.ndarray:
    """
    Ensure labels are in {0,1,2,3}. If not, map by sorted unique values:
      background = 0 if exists else most frequent value;
      next 3 unique values -> 1..3.
    """
    lab = lab.astype(np.int32)
    u = np.unique(lab)
    if set(u.tolist()).issubset({0, 1, 2, 3}):
        return lab.astype(np.uint8)

    if 0 in u:
        bg = 0
    else:
        vals, cnt = np.unique(lab, return_counts=True)
        bg = int(vals[np.argmax(cnt)])

    non_bg = sorted([v for v in u.tolist() if v != bg])
    mapping = {bg: 0}
    for i, v in enumerate(non_bg[:3], start=1):
        mapping[v] = i

    out = np.zeros_like(lab, dtype=np.uint8)
    for v, nv in mapping.items():
        out[lab == v] = nv
    return out


# Resolve the dataset's native foreground and semantic mask filenames.
def find_file_by_id(folder: Path, pid: int, prefer_hdr=True) -> Path:
    cands = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if re.search(rf"(\D|^){pid}(\D|$)", p.stem.lower()):
            cands.append(p)

    if not cands:
        raise FileNotFoundError(f"File for id={pid} not found in {folder}")

    cands = sorted(
        cands,
        key=lambda p: (
            0 if (prefer_hdr and p.suffix.lower() == ".hdr") else 1,
            natural_key(p.name),
        ),
    )
    return cands[0]


# ---------- scoring (for band selection) ----------
def score_min_pair_effect_size(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    score(b) = min_{i<j} |mu_i-mu_j| / sqrt(var_i+var_j)
    y should contain only classes {1,2,3} for selection.
    """
    classes = np.array(sorted(np.unique(y).tolist()))
    B = X.shape[1]
    scores = np.zeros(B, dtype=np.float64)
    idxs = {c: np.where(y == c)[0] for c in classes}

    for b in range(B):
        min_d = np.inf
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                c1, c2 = classes[i], classes[j]
                x1 = X[idxs[c1], b]
                x2 = X[idxs[c2], b]
                if x1.size < 2 or x2.size < 2:
                    continue
                mu1, mu2 = float(x1.mean()), float(x2.mean())
                var1, var2 = float(x1.var(ddof=1)), float(x2.var(ddof=1))
                d = abs(mu1 - mu2) / (np.sqrt(var1 + var2) + EPS)
                min_d = min(min_d, d)
        scores[b] = 0.0 if not np.isfinite(min_d) else min_d
    return scores


# ---------- redundancy control: min_gap + correlation ----------
def build_corr_cache(X_use: np.ndarray, max_rows: int, rng: np.random.Generator):
    N = X_use.shape[0]
    m = min(int(max_rows), N)
    idx = rng.choice(N, m, replace=False) if m < N else np.arange(N)
    Xs = X_use[idx].astype(np.float32)  # (m,B)
    mean = Xs.mean(axis=0, keepdims=True)
    Xc = Xs - mean
    std = Xc.std(axis=0, ddof=1)
    return Xc, std


def corr_abs(Xc: np.ndarray, std: np.ndarray, b: int, s: int) -> float:
    sb = float(std[b])
    ss = float(std[s])
    if sb < 1e-12 or ss < 1e-12:
        return 1.0
    cov = float((Xc[:, b] * Xc[:, s]).mean())
    c = cov / (sb * ss + 1e-12)
    if not np.isfinite(c):
        return 1.0
    return abs(c)


def select_bands(
    scores: np.ndarray,
    k: int,
    min_gap: int,
    corr_thresh: float,
    X_use_for_corr: np.ndarray,
    corr_sample_rows: int,
    rng: np.random.Generator,
):
    order = np.argsort(scores)[::-1]
    picked = []
    Xc, std = build_corr_cache(X_use_for_corr, corr_sample_rows, rng)

    for b in order:
        b = int(b)
        ok = True
        for s in picked:
            if abs(b - s) < min_gap:
                ok = False
                break
            if corr_abs(Xc, std, b, s) > corr_thresh:
                ok = False
                break
        if ok:
            picked.append(b)
            if len(picked) == k:
                break
    return picked


# ---------- global stretch for visualization ----------
def compute_global_stretch(X: np.ndarray, picked_bands: list[int], qlo=0.01, qhi=0.99):
    stretches = []
    for b in picked_bands:
        lo = float(np.quantile(X[:, b], qlo))
        hi = float(np.quantile(X[:, b], qhi))
        stretches.append((lo, hi))
    return stretches


# ---------- outputs ----------
def make_tensor_bands(
    memmap_hwb,
    pcb_mask: np.ndarray,
    bands: list[int],
    mode: str,
    stretches=None,
    bg: float = 0.0,
) -> np.ndarray:
    """
    Return (H,W,K) float32 in [0,1]
      mode=physical: reflectance*mask (no stretch)
      mode=global: global stretch then mask
    """
    H, W, _ = memmap_hwb.shape
    K = len(bands)
    out = np.full((H, W, K), bg, dtype=np.float32)
    m = pcb_mask.astype(bool)

    for ch, b in enumerate(bands):
        band = np.asarray(memmap_hwb[:, :, b], dtype=np.float32)
        band = np.clip(band, 0.0, 1.0)

        if mode == "global":
            lo, hi = stretches[ch]
            band = (band - lo) / (hi - lo + 1e-6)
            band = np.clip(band, 0.0, 1.0)

        out[:, :, ch][m] = band[m]
    return out


# Three-channel LDA projection.
def make_tensor_lda_transform(
    memmap_hwb,
    pcb_mask: np.ndarray,
    bands: list[int],
    lda: LinearDiscriminantAnalysis,
    qlo: np.ndarray,
    qhi: np.ndarray,
    bg: float = 0.0,
) -> np.ndarray:
    """
    Return (H,W,3) LDA projection features, then global quantile scaling to [0,1].
    """
    H, W, _ = memmap_hwb.shape
    m = pcb_mask.astype(bool)

    # read only selected K bands
    Xk = np.stack([np.asarray(memmap_hwb[:, :, b], np.float32) for b in bands], axis=-1)
    Xk = np.clip(Xk, 0.0, 1.0)

    out = np.full((H, W, 3), bg, dtype=np.float32)

    valid = Xk[m]  # (N,K)
    if valid.size > 0:
        z = lda.transform(valid)  # (N,3)
        z = (z - qlo) / (qhi - qlo + 1e-8)
        z = np.clip(z, 0.0, 1.0)
        out[m] = z
    return out


def save_png_3ch(rgb01: np.ndarray, out_path: Path):
    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError("save_png_3ch expects (H,W,3)")
    rgb8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb8).save(out_path)


def save_npy(arr: np.ndarray, out_path: Path):
    np.save(out_path, arr.astype(np.float32))


def decide_format(save_format: str, channels: int) -> str:
    if save_format == "auto":
        return "png" if channels == 3 else "npy"
    return save_format


def lda_cv_report(X: np.ndarray, y: np.ndarray, picked_bands: list[int], folds=5):
    print("\n" + "=" * 50)
    print(">>> Sanity Check: LDA with Stratified CV (on classes 1/2/3 only) <<<")
    Xs = X[:, picked_bands]
    lda = LinearDiscriminantAnalysis()
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    pred = cross_val_predict(lda, Xs, y, cv=cv)
    acc = accuracy_score(y, pred)
    print(f"CV accuracy (fold={folds}): {acc:.4f}")
    print(
        classification_report(
            y,
            pred,
            labels=[1, 2, 3],
            target_names=["IC (1)", "Capacitor (2)", "Connectors (3)"],
            zero_division=0,
        )
    )
    print("=" * 50 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument(
        "--label_type", type=str, default="Monoseg", choices=["Monoseg", "General"]
    )

    # selection scope
    ap.add_argument(
        "--use_all_53", action="store_true", help="Use all 53 for band selection"
    )
    ap.add_argument("--max_per_class_per_pcb", type=int, default=5000)
    ap.add_argument(
        "--max_bg_per_pcb",
        type=int,
        default=5000,
        help="for LDA-transform: sample class 0",
    )

    # band range control
    ap.add_argument("--drop_head", type=int, default=20)
    ap.add_argument("--drop_tail", type=int, default=20)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--min_gap", type=int, default=15)
    ap.add_argument("--corr_thresh", type=float, default=0.98)
    ap.add_argument("--corr_sample_rows", type=int, default=20000)

    # output
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument(
        "--bands_mode", type=str, default="physical", choices=["physical", "global"]
    )
    ap.add_argument("--bg", type=float, default=0.5)
    ap.add_argument("--qlo", type=float, default=0.01)
    ap.add_argument("--qhi", type=float, default=0.99)

    ap.add_argument(
        "--output_mode",
        type=str,
        default="lda_transform",
        choices=["bands", "lda_transform"],
        help="bands: output (H,W,K) band tensor; lda_transform: output (H,W,3) LDA projection",
    )
    ap.add_argument(
        "--save_format", type=str, default="auto", choices=["auto", "png", "npy"]
    )
    ap.add_argument("--cv_folds", type=int, default=5)

    args = ap.parse_args()

    root = Path(args.root)
    hsi_dir = root / "HSI"
    label_dir = hsi_dir / f"{args.label_type}_masks"
    pcb_mask_dir = hsi_dir / "PCB_Masks"

    out_dir = Path(args.out_dir) if args.out_dir else (root / "spectral_projection")
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = list(range(1, 54)) if args.use_all_53 else TRAIN_IDS
    rng = np.random.default_rng(0)

    # -------------------------------
    # 1) Build pooled pixels for band selection (classes 1/2/3 only)
    #    and pooled pixels for LDA transform (classes 0/1/2/3 if needed)
    # -------------------------------
    print(
        f"Extracting spectra... ids={len(ids)} ({'ALL53' if args.use_all_53 else 'TRAIN'})"
    )

    X_sel, y_sel = [], []  # only 1/2/3 for band scoring
    X_lda, y_lda = [], []  # 0/1/2/3 for LDA transform training (if enabled)

    for pid in ids:
        pcb_dir = hsi_dir / f"pcb{pid}"
        hdr = find_envi_hdr(pcb_dir)
        cube = spectral.envi.open(str(hdr)).open_memmap()  # (H,W,B)

        lab_path = find_file_by_id(label_dir, pid, prefer_hdr=True)
        pcbm_path = find_file_by_id(pcb_mask_dir, pid, prefer_hdr=True)

        lab_raw = load_mask_auto(lab_path)
        pcbm_raw = load_mask_auto(pcbm_path)
        if pcbm_raw.ndim == 3:
            pcbm_raw = pcbm_raw[:, :, 0]
        pcbm = pcbm_raw > 0

        if lab_raw.ndim == 2:
            lab = ensure_label_ids(lab_raw)
        else:
            lab = rgb_mask_to_labels(lab_raw)

        H, W, B = cube.shape
        if lab.shape[:2] != (H, W) or pcbm.shape != (H, W):
            raise ValueError(
                f"Size mismatch pid={pid}: cube={(H,W,B)}, lab={lab.shape}, pcbm={pcbm.shape}"
            )

        # ----- selection sampling (1/2/3) -----
        for cls in [1, 2, 3]:
            m = (lab == cls) & pcbm
            coords = np.argwhere(m)
            if coords.size == 0:
                continue
            if coords.shape[0] > args.max_per_class_per_pcb:
                coords = coords[
                    rng.choice(
                        coords.shape[0], args.max_per_class_per_pcb, replace=False
                    )
                ]
            spectra = np.asarray(cube[coords[:, 0], coords[:, 1], :], dtype=np.float32)
            spectra = np.clip(spectra, 0.0, 1.0)
            X_sel.append(spectra)
            y_sel.append(np.full((spectra.shape[0],), cls, dtype=np.uint8))

        # ----- LDA sampling (0/1/2/3) only if needed -----
        if args.output_mode == "lda_transform":
            # class 0 = unlabeled PCB surface inside pcb mask
            for cls in [0, 1, 2, 3]:
                m = (lab == cls) & pcbm
                coords = np.argwhere(m)
                if coords.size == 0:
                    continue
                limit = args.max_bg_per_pcb if cls == 0 else args.max_per_class_per_pcb
                if coords.shape[0] > limit:
                    coords = coords[rng.choice(coords.shape[0], limit, replace=False)]
                spectra = np.asarray(
                    cube[coords[:, 0], coords[:, 1], :], dtype=np.float32
                )
                spectra = np.clip(spectra, 0.0, 1.0)
                X_lda.append(spectra)
                y_lda.append(np.full((spectra.shape[0],), cls, dtype=np.uint8))

    X_sel = np.concatenate(X_sel, axis=0)
    y_sel = np.concatenate(y_sel, axis=0)

    # -------------------------------
    # 2) Band search range: drop_head + drop_tail
    # -------------------------------
    drop_h = int(args.drop_head)
    drop_t = int(args.drop_tail)
    B_total = X_sel.shape[1]
    start = drop_h
    end = B_total - drop_t if drop_t > 0 else B_total
    if not (0 <= start < end <= B_total):
        raise ValueError(
            f"Invalid band range: start={start}, end={end}, B_total={B_total}"
        )

    X_use = X_sel[:, start:end]
    print(f"Band search range: [{start}, {end}) => B_use={X_use.shape[1]}")

    # -------------------------------
    # 3) Score + select K bands (min-pair) + correlation redundancy control
    # -------------------------------
    scores = score_min_pair_effect_size(X_use, y_sel)
    picked_local = select_bands(
        scores=scores,
        k=int(args.k),
        min_gap=int(args.min_gap),
        corr_thresh=float(args.corr_thresh),
        X_use_for_corr=X_use,
        corr_sample_rows=int(args.corr_sample_rows),
        rng=rng,
    )
    picked = [p + start for p in picked_local]
    print(f"[OK] picked bands (0-based): {picked}")

    # sanity check on 1/2/3 separability with selected bands
    lda_cv_report(
        X_sel[:, picked], y_sel, list(range(len(picked))), folds=int(args.cv_folds)
    )

    # global stretches for bands_mode=global
    stretches = compute_global_stretch(
        X_sel, picked, qlo=float(args.qlo), qhi=float(args.qhi)
    )

    # -------------------------------
    # 4) Train LDA transform if requested
    #    Need 4 classes (0/1/2/3) so that n_components=3 is possible.
    # -------------------------------
    lda_final = None
    lda_qlo = None
    lda_qhi = None

    if args.output_mode == "lda_transform":
        if len(picked) < 3:
            raise ValueError("lda_transform requires at least k>=3 selected bands.")
        X_lda = np.concatenate(X_lda, axis=0)
        y_lda = np.concatenate(y_lda, axis=0)

        # fit LDA to produce 3 components (4 classes -> max 3)
        lda_final = LinearDiscriminantAnalysis(n_components=3)
        Z = lda_final.fit_transform(X_lda[:, picked], y_lda)  # (N,3)

        lda_qlo = np.quantile(Z, float(args.qlo), axis=0)
        lda_qhi = np.quantile(Z, float(args.qhi), axis=0)

    # save metadata
    meta = {
        "label_type": args.label_type,
        "ids_used_for_selection": ids,
        "use_all_53": bool(args.use_all_53),
        "drop_head": drop_h,
        "drop_tail": drop_t,
        "k": args.k,
        "min_gap": args.min_gap,
        "corr_thresh": args.corr_thresh,
        "corr_sample_rows": args.corr_sample_rows,
        "picked_bands_0_based": picked,
        "picked_bands_local": picked_local,
        "bands_mode": args.bands_mode,
        "output_mode": args.output_mode,
        "save_format": args.save_format,
        "bg": args.bg,
        "stretch_quantiles": [args.qlo, args.qhi],
        "global_stretches": stretches,
        "lda_qlo": lda_qlo.tolist() if lda_qlo is not None else None,
        "lda_qhi": lda_qhi.tolist() if lda_qhi is not None else None,
        "note": "Using all 53 for selection may leak information if you later do train/test evaluation.",
    }
    with open(out_dir / "selected_bands.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved: {out_dir / 'selected_bands.json'}")

    # -------------------------------
    # 5) Generate outputs for ALL 53 (always)
    # -------------------------------
    print(f"Generating outputs for ALL 53... output_mode={args.output_mode}")
    for pid in range(1, 54):
        pcb_dir = hsi_dir / f"pcb{pid}"
        hdr = find_envi_hdr(pcb_dir)
        cube = spectral.envi.open(str(hdr)).open_memmap()

        pcbm_path = find_file_by_id(pcb_mask_dir, pid, prefer_hdr=True)
        pcbm_raw = load_mask_auto(pcbm_path)
        if pcbm_raw.ndim == 3:
            pcbm_raw = pcbm_raw[:, :, 0]
        pcbm = (pcbm_raw > 0).astype(np.uint8)

        if args.output_mode == "bands":
            tensor = make_tensor_bands(
                cube,
                pcbm,
                picked,
                mode=args.bands_mode,
                stretches=stretches,
                bg=float(args.bg),
            )
            fmt = decide_format(args.save_format, tensor.shape[2])
            if fmt == "png":
                if tensor.shape[2] != 3:
                    raise ValueError(
                        "PNG only supports 3-channel. Use --save_format npy for k!=3."
                    )
                save_png_3ch(tensor, out_dir / f"{pid:02d}.png")
            else:
                save_npy(tensor, out_dir / f"{pid:02d}.npy")

        else:  # lda_transform
            tensor = make_tensor_lda_transform(
                cube,
                pcbm,
                picked,
                lda=lda_final,
                qlo=lda_qlo,
                qhi=lda_qhi,
                bg=float(args.bg),
            )
            fmt = decide_format(args.save_format, 3)
            if fmt == "png":
                save_png_3ch(tensor, out_dir / f"{pid:02d}.png")
            else:
                save_npy(tensor, out_dir / f"{pid:02d}.npy")

    print(f"[OK] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
