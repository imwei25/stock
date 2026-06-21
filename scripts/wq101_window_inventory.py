"""Static AST scan of factors/wq101.py for window literals in window-bearing ops.

Output CSV columns: alpha_id, op, window, count_in_alpha, category, transformable.

`transformable=False` flags alphas with non-literal window args; those cannot
be auto-rewritten by generate_wq101_variants.py.
"""
from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WQ101_SRC = REPO_ROOT / "src" / "stockpool" / "factors" / "wq101.py"

# op_name -> 0-indexed position of window arg
WINDOW_OPS = {
    "ts_sum": 1, "ts_mean": 1, "ts_min": 1, "ts_max": 1,
    "ts_argmin": 1, "ts_argmax": 1, "ts_rank": 1,
    "ts_std": 1, "ts_product": 1,
    "delta": 1, "delay": 1, "decay_linear": 1,
    "correlation": 2, "covariance": 2,
    "_adv": 1,
}


def _categorize(w: int) -> str:
    if w <= 10:
        return "short"
    if w <= 30:
        return "medium"
    if w >= 60:
        return "long"
    return "other"


def _is_alpha_class(node: ast.ClassDef) -> str | None:
    """Return alpha_NNN if node decorated with @_wq(N, ...), else None."""
    for dec in node.decorator_list:
        if (isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "_wq"):
            if dec.args and isinstance(dec.args[0], ast.Constant):
                num = dec.args[0].value
                if isinstance(num, int):
                    return f"alpha_{num:03d}"
    return None


def _scan_compute_body(compute_fn: ast.FunctionDef):
    """Yield (op_name, window_value_or_None, is_literal) for each whitelisted call."""
    for sub in ast.walk(compute_fn):
        if not isinstance(sub, ast.Call):
            continue
        # match ops.<op_name>(...) or _adv(...)
        op_name = None
        if isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name):
            if sub.func.value.id == "ops":
                op_name = sub.func.attr
        elif isinstance(sub.func, ast.Name):
            if sub.func.id == "_adv":
                op_name = "_adv"
        if op_name not in WINDOW_OPS:
            continue
        pos = WINDOW_OPS[op_name]
        if len(sub.args) <= pos:
            continue
        arg = sub.args[pos]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            yield op_name, arg.value, True
        else:
            yield op_name, None, False


def scan_file(path: Path) -> list[dict]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    rows: list[dict] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        alpha_id = _is_alpha_class(node)
        if alpha_id is None:
            continue
        # find compute method
        compute = next(
            (n for n in node.body
             if isinstance(n, ast.FunctionDef) and n.name == "compute"),
            None,
        )
        if compute is None:
            continue
        # collect (op, window) pairs
        literal_pairs: list[tuple[str, int]] = []
        has_non_literal = False
        for op_name, w, is_lit in _scan_compute_body(compute):
            if is_lit:
                literal_pairs.append((op_name, w))
            else:
                has_non_literal = True
        # de-duplicate + count
        counts = Counter(literal_pairs)
        for (op_name, w), cnt in sorted(counts.items()):
            rows.append({
                "alpha_id": alpha_id,
                "op": op_name,
                "window": w,
                "count_in_alpha": cnt,
                "category": _categorize(w),
                "transformable": not has_non_literal,
            })
        # if no literals at all, still emit one row with window blank
        if not counts:
            rows.append({
                "alpha_id": alpha_id, "op": "", "window": "",
                "count_in_alpha": 0, "category": "",
                "transformable": False,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="reports/wq101_window_inventory.csv")
    args = ap.parse_args()
    rows = scan_file(WQ101_SRC)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "alpha_id", "op", "window", "count_in_alpha",
            "category", "transformable",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
