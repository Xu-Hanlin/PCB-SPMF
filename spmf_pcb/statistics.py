from __future__ import annotations
import ast, copy, hashlib, json, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np

BOARDS = tuple(range(43, 54))
NAMES = {0: "ic", 1: "capacitor", 2: "connector"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_native_metric(path: Path):
    """Load only the three verified math functions, retaining their native code."""
    text = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(text, filename=str(path))
    wanted = ("smooth", "compute_ap", "ap_per_class")
    nodes = {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in wanted
    }
    require(set(nodes) == set(wanted), f"Missing metric functions in {path}")
    for name, node in nodes.items():
        require(
            not node.decorator_list,
            f"Unexpected decorator on {name}; inspect this metric source",
        )
    # Verify the return layout used below rather than silently guessing a version.
    returns = [n for n in ast.walk(nodes["ap_per_class"]) if isinstance(n, ast.Return)]
    require(
        len(returns) == 1 and isinstance(returns[0].value, ast.Tuple),
        "Unexpected ap_per_class return structure",
    )
    elems = returns[0].value.elts
    for index, name in [(2, "p"), (3, "r"), (5, "ap")]:
        require(
            len(elems) > index
            and isinstance(elems[index], ast.Name)
            and elems[index].id == name,
            f"Unsupported ap_per_class return slot {index}",
        )

    def check_version(current, required):
        # This is the one compatibility branch used by the inspected native
        # compute_ap: NumPy >=2 uses trapezoid; older NumPy uses trapz.
        require(
            required == ">=2.0",
            f"Unexpected checks.check_version requirement {required!r}",
        )
        return np.lib.NumpyVersion(str(current)) >= np.lib.NumpyVersion("2.0.0")

    namespace = {
        "np": np,
        "Path": Path,
        "checks": SimpleNamespace(check_version=check_version),
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(
        body=[future] + [copy.deepcopy(nodes[n]) for n in wanted], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    hashes = {
        name: hashlib.sha256(
            ast.get_source_segment(text, node).encode("utf-8")
        ).hexdigest()
        for name, node in nodes.items()
    }
    return namespace["ap_per_class"], {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "function_source_sha256": hashes,
        "extraction": "Verbatim AST functions smooth, compute_ap, ap_per_class; no ranking or integration edits",
        "plot": False,
    }


def records_from_rows(rows, source_label):
    require(
        isinstance(rows, list) and rows,
        f"Expected a nonempty record list: {source_label}",
    )
    records = []
    seen = set()
    for item in rows:
        pid = item["pcb"]
        require(
            isinstance(pid, int) and not isinstance(pid, bool) and pid not in seen,
            f"Invalid/duplicate PCB: {pid}",
        )
        seen.add(pid)
        require(
            isinstance(item.get("dtypes"), dict),
            f"Missing native per-PCB dtype metadata: PCB {pid}",
        )
        native_dtypes = {
            key: np.dtype(item["dtypes"][key])
            for key in ["tp", "conf", "pred_cls", "target_cls"]
        }
        require(native_dtypes["tp"].kind == "b", f"Native TP must be bool: PCB {pid}")
        require(
            native_dtypes["conf"].kind == "f"
            and native_dtypes["conf"].itemsize in [2, 4, 8],
            f"Unsupported confidence dtype: PCB {pid}",
        )
        for key in ["pred_cls", "target_cls"]:
            require(
                native_dtypes[key].kind in "fiu"
                and native_dtypes[key].itemsize in [1, 2, 4, 8],
                f"Unsupported native {key} dtype: PCB {pid}",
            )
        raw_tp = np.asarray(item["tp"])
        require(
            raw_tp.size == 0
            or (raw_tp.dtype.kind in "biu" and np.isin(raw_tp, [0, 1]).all()),
            f"Invalid boolean TP records: PCB {pid}",
        )
        require(
            raw_tp.size == 0 or (raw_tp.ndim == 2 and raw_tp.shape[1] == 10),
            f"Expected predictions by ten IoU thresholds: PCB {pid}",
        )
        tp = raw_tp.astype(native_dtypes["tp"]).reshape(-1, 10)
        arrays = {}
        for key in ["conf", "pred_cls", "target_cls"]:
            original_values = np.asarray(item[key], dtype=np.float64)
            require(
                original_values.ndim == 1, f"Expected one-dimensional {key}: PCB {pid}"
            )
            value = np.asarray(item[key], dtype=native_dtypes[key])
            require(
                np.array_equal(original_values, value.astype(np.float64)),
                f"Serialized {key} does not round-trip exactly to its declared native dtype: PCB {pid}",
            )
            arrays[key] = value
        conf, pc, tc = arrays["conf"], arrays["pred_cls"], arrays["target_cls"]
        require(
            len(tp) == len(conf) == len(pc), f"Unequal prediction lengths: PCB {pid}"
        )
        require(
            np.isfinite(conf).all() and ((conf >= 0) & (conf <= 1)).all(),
            f"Invalid confidence: PCB {pid}",
        )
        for cls in (pc, tc):
            require(
                np.isfinite(cls).all() and np.isin(cls, [0, 1, 2]).all(),
                f"Invalid class IDs: PCB {pid}",
            )
        records.append(
            {"pcb": pid, "tp": tp, "conf": conf, "pred_cls": pc, "target_cls": tc}
        )
    require(
        seen == set(BOARDS),
        f"Expected exactly PCB43--53, received {sorted(seen)} in {source_label}",
    )
    return records


def records_from_json(path: Path):
    return records_from_rows(read_json(path), str(path))


def pooled_metrics(records, native_ap):
    """Pool all detections/targets before native sorting, matching and AP integration.

    tp is already the evaluator's per-image matching result. Repeating a PCB
    repeats its detections, targets and matching as a complete resampled cluster.
    No IoU rematching across images or averaging of per-board AP is performed.
    """
    tp = np.concatenate([x["tp"] for x in records], axis=0)
    conf = np.concatenate([x["conf"] for x in records])
    pred_cls = np.concatenate([x["pred_cls"] for x in records])
    target_cls = np.concatenate([x["target_cls"] for x in records])
    require(
        len(target_cls) > 0,
        "A pooled sample has no ground-truth targets; no AP is defined",
    )
    values = native_ap(tp, conf, pred_cls, target_cls, plot=False, names=NAMES)
    ap = np.asarray(values[5])
    require(
        ap.ndim == 2 and ap.shape[1] == 10 and len(ap), "Unexpected native AP array"
    )
    out = {
        "P": float(np.mean(values[2])),
        "R": float(np.mean(values[3])),
        "mAP50": float(ap[:, 0].mean()),
        "mAP50_95": float(ap.mean()),
    }
    require(
        all(math.isfinite(v) and 0 <= v <= 1 for v in out.values()),
        "Nonfinite/out-of-range pooled metrics",
    )
    return out


def same_record_arrays(left, right):
    a = {x["pcb"]: x for x in left}
    b = {x["pcb"]: x for x in right}
    return all(
        np.array_equal(a[i][key], b[i][key])
        for i in BOARDS
        for key in ["tp", "conf", "pred_cls", "target_cls"]
    )
