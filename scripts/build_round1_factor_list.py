"""Build reports/wq101_round1_factors.json from a baseline + generated variants.

Output is {"factors": [top-30 wq101 baseline names + all _compress/_rev_short/_expand_long variants of those names]}.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from stockpool.factors import list_factors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    d = json.loads(args.baseline.read_text(encoding="utf-8"))
    abs_ic = d["abs_ic_mean"]
    pairs = sorted(
        ((float(v), n) for n, v in abs_ic.items()
         if n.startswith("alpha_") and v is not None and v == v),
        reverse=True,
    )
    top_names = [n for _, n in pairs[:args.top_n]]

    all_registered = set(list_factors())
    factors: list[str] = []
    for base in top_names:
        factors.append(base)
        for rule in ("compress", "rev_short", "expand_long"):
            v = f"{base}_{rule}"
            if v in all_registered:
                factors.append(v)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"factors": factors}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {len(factors)} factor names to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
