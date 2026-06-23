"""Pick WQ101 variants that beat their baseline on both walk-forward halves.

Criteria per spec §7.1:
  h1 & h2 each: Δabs_ic ≥ 0.02 AND Δ|ir| ≥ 0.1 AND degenerate ≤ 0.10.
  Pick variant maximizing min(abs_ic_h1, abs_ic_h2).
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _passes(base_row, var_row, *,
            min_dabs=0.02, min_dir=0.10, max_degen=0.10) -> bool:
    if var_row["degen"] > max_degen:
        return False
    if (var_row["abs"] - base_row["abs"]) < min_dabs:
        return False
    if (abs(var_row["ir"]) - abs(base_row["ir"])) < min_dir:
        return False
    return True


def _table(j: dict) -> dict[str, dict]:
    abs_ic = j["abs_ic_mean"]; ir = j["ic_ir"]; degen = j.get("degenerate_day_ratio", {})
    return {n: {
        "abs": float(abs_ic[n]) if abs_ic[n] is not None else float("nan"),
        "ir": float(ir[n]) if ir[n] is not None else float("nan"),
        "degen": float(degen.get(n, 0.0)) if degen.get(n) is not None else 0.0,
    } for n in j["factor_names"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1", required=True, type=Path)
    ap.add_argument("--h2", required=True, type=Path)
    ap.add_argument("--current-selection", required=True, type=Path)
    ap.add_argument("--winners-csv", required=True, type=Path)
    ap.add_argument("--output-selection", required=True, type=Path)
    ap.add_argument("--baseline-top-n", type=int, default=30,
                    help="Only consider this many baseline wq101 alphas, "
                         "ranked by min(abs_ic_h1, abs_ic_h2).")
    args = ap.parse_args()

    h1, h2 = _table(_load(args.h1)), _table(_load(args.h2))
    cur = _load(args.current_selection)
    # Baseline candidate set: alphas present in both tables, non-variant suffixed.
    suffixes = ("_compress", "_rev_short", "_expand_long")
    baselines = sorted(
        n for n in h1
        if n.startswith("alpha_")
        and not any(n.endswith(s) for s in suffixes)
        and n in h2
    )
    # Rank baselines by min(abs_ic_h1, abs_ic_h2) descending; take top N.
    baselines.sort(key=lambda n: -min(h1[n]["abs"], h2[n]["abs"]))
    baselines = baselines[:args.baseline_top_n]

    winners: list[dict] = []
    for base in baselines:
        candidates = []
        for s in suffixes:
            var = base + s
            if var not in h1 or var not in h2:
                continue
            if not _passes(h1[base], h1[var]):
                continue
            if not _passes(h2[base], h2[var]):
                continue
            candidates.append((min(h1[var]["abs"], h2[var]["abs"]), var))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        chosen = candidates[0][1]
        winners.append({
            "baseline_alpha": base,
            "chosen_variant": chosen,
            "abs_ic_baseline_h1": h1[base]["abs"],
            "abs_ic_winner_h1": h1[chosen]["abs"],
            "delta_abs_ic_h1": h1[chosen]["abs"] - h1[base]["abs"],
            "delta_ir_h1": abs(h1[chosen]["ir"]) - abs(h1[base]["ir"]),
            "abs_ic_baseline_h2": h2[base]["abs"],
            "abs_ic_winner_h2": h2[chosen]["abs"],
            "delta_abs_ic_h2": h2[chosen]["abs"] - h2[base]["abs"],
            "delta_ir_h2": abs(h2[chosen]["ir"]) - abs(h2[base]["ir"]),
        })

    args.winners_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.winners_csv.open("w", newline="", encoding="utf-8") as fh:
        if winners:
            w = csv.DictWriter(fh, fieldnames=list(winners[0].keys()))
            w.writeheader(); w.writerows(winners)
        else:
            fh.write("baseline_alpha,chosen_variant\n")

    # Build the new selection: swap baselines for winners.
    swap = {w_["baseline_alpha"]: w_["chosen_variant"] for w_ in winners}
    new_factors = [swap.get(f, f) for f in cur["factors"]]
    args.output_selection.parent.mkdir(parents=True, exist_ok=True)
    args.output_selection.write_text(
        json.dumps({"factors": new_factors}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{len(winners)} winners written to {args.winners_csv}")
    print(f"Updated selection written to {args.output_selection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
