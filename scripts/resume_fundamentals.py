"""Resume baostock fundamentals fetches with retry + cooldown.

By default only fetches **priority tables** — those that back at least one
registered factor (see src/stockpool/factors/fundamentals.py):

    growth  -> revenue_yoy
    balance -> pb

profit covers the bulk (roe / roa / gross_margin / net_margin / pe /
market_cap / log_market_cap) and is normally already complete.

`cash_flow` and `dupont` currently have no registered factors and are
omitted by default — pass them explicitly via --tables to fetch them.

Retry policy: on dead-session or login-rate-limit (loader raises and
returns empty DataFrame; partial parquet is preserved), sleep
COOLDOWN minutes and re-attempt the **same** table. This avoids the
prior behaviour where a single failure cascaded across all tables
because each load_or_build_fundamentals call burned a fresh login
within seconds of the last.

Usage:
    # priority tables (growth, balance) with default 45-min cooldown
    .venv/Scripts/python.exe scripts/resume_fundamentals.py

    # also fetch low-priority tables
    .venv/Scripts/python.exe scripts/resume_fundamentals.py \\
        --tables growth balance cash_flow dupont

    # longer cooldown if baostock is being aggressive
    .venv/Scripts/python.exe scripts/resume_fundamentals.py --cooldown-min 90
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from stockpool.fundamentals_loader import load_or_build_fundamentals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("resume_fundamentals")

# Tables that back at least one registered factor today.
PRIORITY_TABLES = ["growth", "balance"]
# No registered factor as of 2026-06-19; included for documentation only.
LOW_PRIORITY_TABLES = ["cash_flow", "dupont"]


def _partial_codes(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(pd.read_parquet(path)["code"].nunique())
    except Exception:
        return 0


def resume_one(
    table: str,
    codes: list[str],
    cache_dir: Path,
    cooldown_sec: int,
    max_retries: int,
) -> bool:
    """Fetch one table with retry. True iff final cache reached completion."""
    final = cache_dir / f"fundamentals_{table}.parquet"
    partial = cache_dir / f"fundamentals_{table}.partial.parquet"

    # Seed partial from final so the loader's resume path can skip prior codes.
    if final.exists() and not partial.exists():
        shutil.copy2(final, partial)
        log.info(
            "[%s] seeded partial from final (%d codes)",
            table, _partial_codes(partial),
        )

    for attempt in range(1, max_retries + 1):
        before = _partial_codes(partial)
        log.info(
            "[%s] attempt %d/%d -- starting from %d codes",
            table, attempt, max_retries, before,
        )
        df = load_or_build_fundamentals(
            table, codes=codes, cache_dir=cache_dir, force_refresh=True,
        )
        # Loader removes the partial only on full success (final written).
        if not partial.exists() and final.exists():
            n_rows = len(df)
            n_codes = df["code"].nunique() if not df.empty else 0
            log.info(
                "[%s] OK done: %d rows / %d codes",
                table, n_rows, n_codes,
            )
            return True

        after = _partial_codes(partial)
        gained = after - before
        log.warning(
            "[%s] attempt %d incomplete -- gained %d codes "
            "(partial now %d). loader returned %d rows.",
            table, attempt, gained, after,
            len(df) if df is not None else 0,
        )
        if attempt < max_retries:
            log.info(
                "[%s] cooldown %d min before retry...",
                table, cooldown_sec // 60,
            )
            time.sleep(cooldown_sec)

    log.error(
        "[%s] FAIL exhausted %d retries; %d codes in partial",
        table, max_retries, _partial_codes(partial),
    )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables", nargs="*", default=PRIORITY_TABLES,
        help=(
            f"Tables to resume. Default: priority tables only ({PRIORITY_TABLES}). "
            f"Low-priority (no registered factor): {LOW_PRIORITY_TABLES}."
        ),
    )
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--universe", default="data/universe.parquet")
    parser.add_argument(
        "--cooldown-min", type=int, default=45,
        help="Minutes to sleep after a failed attempt (default 45).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Maximum attempts per table (default 3).",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    codes = (
        pd.read_parquet(args.universe)["code"]
        .astype(str).str.zfill(6).tolist()
    )
    log.info("universe: %d codes", len(codes))
    log.info(
        "tables=%s cooldown=%dmin max_retries=%d",
        args.tables, args.cooldown_min, args.max_retries,
    )

    cooldown_sec = args.cooldown_min * 60
    results: dict[str, bool] = {}
    for table in args.tables:
        results[table] = resume_one(
            table, codes, cache_dir, cooldown_sec, args.max_retries,
        )

    log.info("=" * 60)
    log.info("summary:")
    for tbl, ok in results.items():
        log.info("  %s %s", "OK  " if ok else "FAIL", tbl)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
