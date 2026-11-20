from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


def _flatten_dict(d: Dict[str, Any], prefix: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], Any]:
    """Flatten nested dict into {path_tuple: leaf_value}."""
    out: Dict[Tuple[str, ...], Any] = {}
    for k, v in d.items():
        path = prefix + (str(k),)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, path))
        else:
            out[path] = v
    return out


def summarize_list_of_dicts(
    rows: List[Dict[str, Any]],
    *,
    sep: str = ".",                 # path joiner in output keys
    ddof: int = 0,                  # 0 for population std, 1 for sample std
) -> Dict[str, Dict[str, float]]:
    """
    Summarize numeric leaves across a list of nested dicts.

    Output keys are flattened paths, e.g. "rouge.rougeL.precision".
    For each numeric leaf, compute: min, q1, median, q3, max, mean, std.
    """
    if not rows:
        return {}

    # Collect values for each flattened path
    values_by_path: Dict[Tuple[str, ...], List[float]] = {}

    for r in rows:
        flat = _flatten_dict(r)
        for path, v in flat.items():
            if isinstance(v, (int, float, np.number, bool)):
                values_by_path.setdefault(path, []).append(float(v))

    out: Dict[str, Dict[str, float]] = {}

    for path, vals in values_by_path.items():
        if not vals:
            continue

        arr = np.asarray(vals, dtype=float)

        if arr.size == 0:
            continue

        q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
        key = sep.join(path)

        out[key] = {
            "min": float(np.min(arr)),
            "q1": float(q1),
            "median": float(med),
            "q3": float(q3),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=ddof)),
        }

    return out