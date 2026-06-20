"""Generate tests/fixtures/{ops_panel,ops_snapshot}.parquet.

Snapshot the pandas-oracle output of every registered factor on a
deterministic 100-stock × 250-day OHLCV slice. The two parquet files
are committed to git and become the contract Rust ops in PR-2..5
must reproduce element-wise within atol=1e-9, rtol=1e-7.

Selection rules:
  * codes  = alphabetical-first 100 codes from data/universe.parquet
             that also have data/<code>_daily.parquet present
  * window = last 250 trading days of the union date index across
             those 100 codes (with whatever the panel.build cap yields)

Regenerate after:
  * adding or removing a registered factor
  * intentionally changing a pandas oracle implementation
DO NOT regenerate because Rust diverged; Rust must match this snapshot.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Belt-and-braces: even before PR-2 lands, set the env var so a future
# Rust import path is forced off when this generator runs.
os.environ.setdefault("STOCKPOOL_USE_PYTHON_OPS", "1")

import pandas as pd
from tqdm import tqdm

from stockpool.factors import list_factors, make_factor
from stockpool.factors.context import set_sector_map
from stockpool.industry_map import load_or_build_industry_map
from stockpool.panel import build_panel_from_cache

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data"
UNIVERSE_FILE = CACHE_DIR / "universe.parquet"
FIXTURES = ROOT / "tests" / "fixtures"
PANEL_OUT = FIXTURES / "ops_panel.parquet"
SNAPSHOT_OUT = FIXTURES / "ops_snapshot.parquet"

N_STOCKS = 100
N_DAYS = 250

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("gen_ops_snapshot")


def select_codes() -> list[str]:
    if not UNIVERSE_FILE.exists():
        log.error("%s missing; run `python -m stockpool fetch-universe` first",
                  UNIVERSE_FILE)
        sys.exit(2)
    all_codes = sorted(
        pd.read_parquet(UNIVERSE_FILE)["code"].astype(str).str.zfill(6).tolist()
    )
    codes = [c for c in all_codes
             if (CACHE_DIR / f"{c}_daily.parquet").exists()][:N_STOCKS]
    if len(codes) < N_STOCKS:
        log.warning("only %d codes available (wanted %d)", len(codes), N_STOCKS)
    return codes


def panel_to_long(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack OHLCV wide frames into one long DataFrame for parquet storage."""
    parts = []
    for field, wide in panel.items():
        # TODO(pandas>=2.1): switch to future_stack=True once we pin pandas minor.
        long = wide.stack(dropna=False).rename(field).reset_index()
        long.columns = ["date", "code", field]
        long = long.set_index(["date", "code"])
        parts.append(long)
    out = pd.concat(parts, axis=1).reset_index()
    # Stable column order regardless of dict iteration order
    return out[["date", "code", "open", "high", "low", "close", "volume"]]


def main() -> int:
    codes = select_codes()
    log.info("selected %d codes (first=%s last=%s)", len(codes), codes[0], codes[-1])

    sector_map = load_or_build_industry_map(CACHE_DIR, source="auto")
    set_sector_map(sector_map or {})
    if not sector_map:
        log.warning(
            "sector_map is EMPTY -- indneutralize-based alphas will demean "
            "globally instead of per-sector. The snapshot will encode this "
            "fallback semantics, so REGENERATE later when baostock is up."
        )
    log.info("sector_map size=%d", len(sector_map or {}))

    panel = build_panel_from_cache(codes, history_days=N_DAYS, cache_dir=CACHE_DIR)
    log.info("panel shape: %d × %d", panel["close"].shape[0], panel["close"].shape[1])

    FIXTURES.mkdir(parents=True, exist_ok=True)
    panel_long = panel_to_long(panel)
    panel_long.to_parquet(PANEL_OUT, compression="snappy", index=False)
    log.info("wrote %s (%.1f MB, %d rows)",
             PANEL_OUT, PANEL_OUT.stat().st_size / 1e6, len(panel_long))

    factor_names = list_factors()
    log.info("computing %d factors...", len(factor_names))

    snapshot_parts: list[pd.DataFrame] = []
    for name in tqdm(factor_names, desc="factors", unit="factor"):
        f = make_factor(name)
        wide = f.compute(panel)
        # TODO(pandas>=2.1): switch to future_stack=True once we pin pandas minor.
        long = wide.stack(dropna=False).rename("value").reset_index()
        long.columns = ["date", "code", "value"]
        long["factor"] = name
        snapshot_parts.append(long[["factor", "date", "code", "value"]])

    snapshot = pd.concat(snapshot_parts, ignore_index=True)
    snapshot.to_parquet(SNAPSHOT_OUT, compression="snappy", index=False)
    log.info("wrote %s (%.1f MB, %d rows)",
             SNAPSHOT_OUT, SNAPSHOT_OUT.stat().st_size / 1e6, len(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
