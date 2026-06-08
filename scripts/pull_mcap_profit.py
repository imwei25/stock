"""Targeted mcap puller: latest totalShare per A-share, from baostock.

baostock query_profit_data is serial and ~0.7s/call today, so pulling the full
16-quarter profit table for ~4357 stocks (~70k calls) would take >12h. For
market-cap neutralization we only need a size proxy, so this puller fetches the
**latest available** totalShare per stock (~1 call/stock, fallback chain over a
few recent quarters) and writes a one-row-per-stock snapshot.

Output: data/mcap_shares.parquet  (cols: code, totalShare, pubDate, statDate)

Downstream (strategy_factory.build_log_mcap_panel) broadcasts totalShare
statically across the backtest window: mcap_t = close_t × totalShare_latest.
The daily mcap variation (price) stays PIT-correct; only the slowly-varying
share count is static — a documented approximation for a neutralization
regressor.

Run: .venv/Scripts/python.exe scripts/pull_mcap_profit.py
"""
from __future__ import annotations

import logging
import sys
import time

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("scripts/pull_mcap_profit.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pull_mcap")

# Most-recent-first fallback chain. Stop at the first quarter that returns a
# non-empty totalShare for the stock. As of 2026-06, 2026Q1 is published for
# most names; the tail covers stocks that report late or were recently listed.
_QUARTERS = [(2026, 1), (2025, 4), (2025, 3), (2025, 2), (2025, 1)]


def _to_bs_code(code: str) -> str:
    if code.startswith(("60", "68")):
        return f"sh.{code}"
    if code.startswith(("00", "30")):
        return f"sz.{code}"
    if code.startswith(("8", "43")):
        return f"bj.{code}"
    return f"sh.{code}"


def main() -> int:
    import baostock as bs
    from stockpool.fetcher import list_universe

    universe = list_universe()
    codes = universe["code"].astype(str).str.zfill(6).tolist()
    log.info("universe: %d codes; pulling latest totalShare each", len(codes))

    lg = bs.login()
    if lg.error_code != "0":
        log.error("baostock login failed: %s", lg.error_msg)
        return 1

    rows: list[dict] = []
    t0 = time.time()
    misses = 0
    try:
        for i, code in enumerate(codes, 1):
            bs_code = _to_bs_code(code)
            got = False
            for year, q in _QUARTERS:
                try:
                    rs = bs.query_profit_data(code=bs_code, year=year, quarter=q)
                    if rs.error_code != "0":
                        continue
                    rec = None
                    while rs.next():
                        rec = dict(zip(rs.fields, rs.get_row_data()))
                    if rec and str(rec.get("totalShare", "")).strip():
                        rows.append({
                            "code": code,
                            "totalShare": rec.get("totalShare"),
                            "pubDate": rec.get("pubDate"),
                            "statDate": rec.get("statDate"),
                        })
                        got = True
                        break
                except Exception as e:  # noqa: BLE001
                    log.warning("%s %dQ%d failed: %s", bs_code, year, q, e)
            if not got:
                misses += 1
            if i % 500 == 0:
                rate = (time.time() - t0) / i
                eta = rate * (len(codes) - i) / 60
                log.info(
                    "progress %d/%d (%.0f%%) hits=%d misses=%d ETA=%.0fmin",
                    i, len(codes), 100 * i / len(codes), len(rows), misses, eta,
                )
    finally:
        bs.logout()

    if not rows:
        log.error("no totalShare rows pulled")
        return 1

    df = pd.DataFrame(rows)
    df["totalShare"] = pd.to_numeric(df["totalShare"], errors="coerce")
    for c in ("pubDate", "statDate"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["totalShare"]).reset_index(drop=True)
    out = "data/mcap_shares.parquet"
    df.to_parquet(out, index=False)
    log.info(
        "DONE in %.0fs: %d/%d codes with totalShare -> %s",
        time.time() - t0, len(df), len(codes), out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
