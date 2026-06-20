"""Run `factors analyze --universe all` in chunks across fresh subprocesses.

Why: a single in-process analyze across all ~167 registered factors with
universe=all (4357 stocks) crashes with Windows ACCESS_VIOLATION
(rc=3221225477) after ~30-90 factors. Streaming the factor compute
(2026-06-20 refactor of analyze_factors) only delayed it. Root cause is
C-level state buildup across iterations -- not BLAS thread count
(verified: thread cap regressed crash to factor 17), not held factor
panels (verified: streaming dropped peak panel memory from 3 GB to 17 MB).

Each chunk runs `python -m stockpool factors analyze --factors <subset>`
in a brand new Python process, so allocator state is reset between
chunks. The driver merges the per-chunk JSONs and recomputes the
cross-factor ic_correlation matrix (the only inter-factor quantity --
mean_ic / ic_ir / half_life / regime_ic are all per-factor).

Usage:
    .venv/Scripts/python.exe scripts/factors_analyze_chunked.py
    .venv/Scripts/python.exe scripts/factors_analyze_chunked.py --chunk-size 20
    .venv/Scripts/python.exe scripts/factors_analyze_chunked.py \\
        --output reports/factor_analysis --universe all
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from stockpool.factors import list_factors
from stockpool.factors_analysis import FactorAnalysisResult
from stockpool.factors_analysis_report import render_factor_analysis_report

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("analyze_chunked")


def run_chunk(
    factors: list[str],
    universe: str,
    horizon: int,
    chunks_dir: Path,
    chunk_idx: int,
    n_chunks: int,
) -> Path:
    """Run one chunk in a fresh subprocess; skip if its JSON already exists."""
    chunk_json = chunks_dir / f"chunk_{chunk_idx:02d}_{factors[0]}_{factors[-1]}.json"
    if chunk_json.exists():
        log.info(
            "chunk %d/%d: SKIP (exists) %s..%s",
            chunk_idx + 1, n_chunks, factors[0], factors[-1],
        )
        return chunk_json

    # The CLI writes <today>.json into --output; use a per-chunk staging dir
    # to avoid filename collisions, then move/rename to the persistent name.
    staging = chunks_dir / f"_staging_{chunk_idx:02d}"
    staging.mkdir(exist_ok=True)
    cmd = [
        PYTHON, "-m", "stockpool", "factors", "analyze",
        "--universe", universe,
        "--horizon", str(horizon),
        "--output", str(staging),
        "--factors", *factors,
    ]
    log.info(
        "chunk %d/%d: RUN %d factors (%s..%s)",
        chunk_idx + 1, n_chunks, len(factors), factors[0], factors[-1],
    )
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        raise RuntimeError(
            f"chunk {chunk_idx + 1}/{n_chunks} failed rc={rc}; "
            f"factors={factors[0]}..{factors[-1]}"
        )
    stamp = date.today().isoformat()
    src = staging / f"{stamp}.json"
    if not src.exists():
        raise RuntimeError(f"chunk produced no JSON at {src}")
    src.replace(chunk_json)
    # tidy staging
    for p in staging.iterdir():
        try:
            p.unlink()
        except Exception:
            pass
    try:
        staging.rmdir()
    except Exception:
        pass
    return chunk_json


def _safe_float(v) -> float:
    if v is None or pd.isna(v):
        return float("nan")
    return float(v)


def merge(chunk_jsons: list[Path], final_json: Path) -> FactorAnalysisResult:
    """Union per-factor metrics, recompute cross-factor ic_correlation."""
    daily_ic: dict[str, pd.Series] = {}
    mean_ic: dict[str, float] = {}
    ic_ir: dict[str, float] = {}
    abs_ic_mean: dict[str, float] = {}
    half_life: dict[str, float] = {}
    regime_buckets: dict[str, dict[str, float]] = {}
    factor_order: list[str] = []

    horizon = ic_window = n_stocks = n_days = None
    start_date = end_date = None

    for j in chunk_jsons:
        res = FactorAnalysisResult.from_json(j)
        for name in res.factor_names:
            if name in daily_ic:
                continue
            factor_order.append(name)
            daily_ic[name] = res.daily_ic[name]
            mean_ic[name] = _safe_float(res.mean_ic.get(name))
            ic_ir[name] = _safe_float(res.ic_ir.get(name))
            abs_ic_mean[name] = _safe_float(res.abs_ic_mean.get(name))
            half_life[name] = _safe_float(res.half_life.get(name))
            for regime, series in res.regime_ic.items():
                regime_buckets.setdefault(regime, {})
                if name in series.index:
                    regime_buckets[regime][name] = _safe_float(series[name])
        horizon = res.horizon
        ic_window = res.ic_window
        n_stocks = res.n_stocks
        n_days = res.n_days
        start_date = res.start_date if start_date is None else min(start_date, res.start_date)
        end_date = res.end_date if end_date is None else max(end_date, res.end_date)

    ic_df = pd.DataFrame(daily_ic)[factor_order]
    ic_correlation = ic_df.corr(method="pearson").fillna(0.0)
    for i, name in enumerate(factor_order):
        ic_correlation.iloc[i, i] = 1.0

    result = FactorAnalysisResult(
        factor_names=factor_order,
        daily_ic=daily_ic,
        mean_ic=pd.Series(mean_ic),
        ic_ir=pd.Series(ic_ir),
        abs_ic_mean=pd.Series(abs_ic_mean),
        half_life=pd.Series(half_life),
        ic_correlation=ic_correlation,
        regime_ic={r: pd.Series(d) for r, d in regime_buckets.items()},
        horizon=horizon,
        ic_window=ic_window,
        n_stocks=n_stocks,
        n_days=n_days,
        start_date=start_date,
        end_date=end_date,
    )
    result.to_json(final_json)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/factor_analysis")
    parser.add_argument("--universe", choices=["pool", "all"], default="all")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument(
        "--chunk-size", type=int, default=10,
        help="Factors per subprocess (default 10; lower for safer, higher for less overhead).",
    )
    parser.add_argument(
        "--factors", nargs="*", default=None,
        help="Restrict to a specific factor subset (default: every registered factor).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_names = list(args.factors) if args.factors else list_factors()
    n_chunks = (len(factor_names) + args.chunk_size - 1) // args.chunk_size
    log.info(
        "%d factors -> %d chunks of <= %d (universe=%s, horizon=%d)",
        len(factor_names), n_chunks, args.chunk_size, args.universe, args.horizon,
    )

    stamp = date.today().isoformat()
    # Persistent chunks dir keyed by run date; surviving across reruns so
    # crashes only re-do the failing chunk, not everything before it.
    chunks_dir = out_dir / "chunks" / stamp
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_jsons: list[Path] = []
    for i in range(0, len(factor_names), args.chunk_size):
        chunk = factor_names[i : i + args.chunk_size]
        j = run_chunk(chunk, args.universe, args.horizon, chunks_dir,
                      i // args.chunk_size, n_chunks)
        chunk_jsons.append(j)

    final_json = out_dir / f"{stamp}.json"
    result = merge(chunk_jsons, final_json)
    log.info(
        "merged %d chunks -> %s (%d factors, %d days)",
        len(chunk_jsons), final_json,
        len(result.factor_names), result.n_days,
    )

    html_path = out_dir / f"{stamp}.html"
    render_factor_analysis_report(result, html_path)
    latest = out_dir / "latest.html"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.write_bytes(html_path.read_bytes())
    log.info("wrote %s and %s", html_path, latest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
