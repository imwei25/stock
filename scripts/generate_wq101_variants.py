"""Generate factors/wq101_variants.py: top-N WQ101 alphas x 3 window rules.

Reads a baseline factor_analysis JSON (must contain abs_ic_mean dict), picks
the top-N wq101 names, AST-rewrites each alpha's compute method body with
three rules (_compress / _rev_short / _expand_long), and emits a Python file
with the generated classes already decorated for registration.

Transformation rules (operate on literal integer windows only):
  * compress     : N -> max(2, ceil(N * 0.5))           (applied to all literals)
  * rev_short    : N <= 10 -> max(2, ceil(N * 0.5))     (preserves med/long)
  * expand_long  : N >= 60 -> ceil(N * 1.5)             (preserves short/med)

Alphas whose compute body has a non-literal window argument are skipped
(see scripts/wq101_window_inventory.py to inventory them).

Until the Phase 0 re-baseline JSON exists, point --baseline at
reports/factor_analysis/2026-06-20.json; Task 7 re-runs with the new baseline.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import textwrap
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


def _transform(w: int, rule: str) -> int:
    if rule == "compress":
        return max(2, math.ceil(w * 0.5))
    if rule == "rev_short":
        return max(2, math.ceil(w * 0.5)) if w <= 10 else w
    if rule == "expand_long":
        return math.ceil(w * 1.5) if w >= 60 else w
    raise ValueError(rule)


class _WindowRewriter(ast.NodeTransformer):
    def __init__(self, rule: str):
        self.rule = rule

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        op_name = None
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "ops":
                op_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            if node.func.id == "_adv":
                op_name = "_adv"
        if op_name not in WINDOW_OPS:
            return node
        pos = WINDOW_OPS[op_name]
        if len(node.args) <= pos:
            return node
        arg = node.args[pos]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            new_w = _transform(arg.value, self.rule)
            node.args[pos] = ast.Constant(value=new_w)
        return node


def _alpha_num(node: ast.ClassDef) -> int | None:
    for dec in node.decorator_list:
        if (isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "_wq"
                and dec.args
                and isinstance(dec.args[0], ast.Constant)):
            return int(dec.args[0].value)
    return None


def _is_transformable(compute: ast.FunctionDef) -> bool:
    """No non-literal window args anywhere in window-bearing op calls."""
    for sub in ast.walk(compute):
        if not isinstance(sub, ast.Call):
            continue
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
        if not (isinstance(sub.args[pos], ast.Constant)
                and isinstance(sub.args[pos].value, int)):
            return False
    return True


def _pick_top_n_wq101(baseline_path: Path, top_n: int) -> list[int]:
    """Return list of alpha numbers (1..101) ordered by abs_ic_mean desc."""
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    abs_ic = data["abs_ic_mean"]
    pairs = []
    for name, v in abs_ic.items():
        if not name.startswith("alpha_"):
            continue
        try:
            n = int(name.split("_")[1])
        except (ValueError, IndexError):
            continue
        if v is None or v != v:  # NaN
            continue
        pairs.append((float(v), n))
    pairs.sort(reverse=True)
    return [n for _, n in pairs[:top_n]]


def generate(baseline: Path, top_n: int, output: Path) -> None:
    src = WQ101_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Map: alpha_num -> ast.ClassDef
    classes: dict[int, ast.ClassDef] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            num = _alpha_num(node)
            if num is not None:
                classes[num] = node

    top_nums = _pick_top_n_wq101(baseline, top_n)
    skipped: list[int] = []
    emitted: list[tuple[int, str]] = []  # (num, rule)

    header_lines = [
        '"""Auto-generated WQ101 variants for A-share localization (Round 1).',
        "",
        "Do not edit by hand; regenerate via:",
        "    python scripts/generate_wq101_variants.py \\",
        "        --baseline reports/factor_analysis/<NEW>.json --top-n 30",
        '"""',
        "from __future__ import annotations",
        "",
        "import numpy as np  # noqa: F401",
        "import pandas as pd  # noqa: F401",
        "",
        "from stockpool.factors import ops  # noqa: F401",
        "from stockpool.factors.base import Factor  # noqa: F401",
        "from stockpool.factors.registry import register",
        "from stockpool.factors.wq101 import (",
        "    WqAlpha, _ret, _vwap, _adv, _nan_like, _indneutralize,",
        ")",
        "",
    ]
    body_chunks: list[str] = []

    for num in top_nums:
        cls_node = classes.get(num)
        if cls_node is None:
            skipped.append(num)
            continue
        compute = next(
            (n for n in cls_node.body
             if isinstance(n, ast.FunctionDef) and n.name == "compute"),
            None,
        )
        if compute is None or not _is_transformable(compute):
            skipped.append(num)
            continue
        for rule in ("compress", "rev_short", "expand_long"):
            # Deep-copy compute via parse(unparse(...)) so AST mutations are isolated.
            new_compute = _WindowRewriter(rule).visit(
                ast.parse(ast.unparse(compute))
            ).body[0]
            ast.fix_missing_locations(new_compute)
            variant_cls_name = f"Alpha{num:03d}_{rule}"
            variant_factor_name = f"alpha_{num:03d}_{rule}"
            description = (
                f"WQ101 alpha_{num:03d} with rule={rule} applied to its window "
                "literals (A-share localization, Round 1)."
            )
            compute_src = ast.unparse(new_compute)
            indented_compute = textwrap.indent(compute_src, "    ")
            body_chunks.append(
                f'@register("{variant_factor_name}",\n'
                f'          sources=("wq101", "wq101_localized"),\n'
                f'          types=("cross_sectional",),\n'
                f'          description={description!r})\n'
                f"class {variant_cls_name}(WqAlpha):\n"
                f"    NUM = {num}\n"
                f"    @property\n"
                f"    def name(self):\n"
                f'        return "{variant_factor_name}"\n'
                f"\n"
                f"{indented_compute}\n"
            )
            emitted.append((num, rule))

    out = output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header_lines) + "\n" + "\n\n".join(body_chunks) + "\n",
                   encoding="utf-8")
    print(f"Emitted {len(emitted)} variants from {len(top_nums)} top alphas;"
          f" skipped {len(skipped)} non-transformable: {skipped}")
    print(f"Wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path,
                    help="factor_analysis JSON with abs_ic_mean dict")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    generate(args.baseline, args.top_n, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
