"""Hardened full-PIT fundamentals puller for baostock's 5 quarterly tables.

The library `fundamentals_loader.load_or_build_fundamentals` does the same pull
but has NO per-query timeout and NO resume — at baostock's ~0.5s/call serial
rate a full pull (5 tables × 12 quarters × ~4375 stocks) takes ~1 day and WILL
hang on a stuck socket (the mcap pull did, at stock ~500). This script adds:

  * socket.setdefaulttimeout — hung reads raise instead of freezing
  * re-login on query error
  * incremental checkpoint save every 250 stocks → restart resumes
    (skips codes already in data/fundamentals_<table>.parquet)

Output (long-form, one row per stock-quarter, matching the schema the
fundamentals factors expect — see docs/handoff/2026-05-31-baostock-fundamentals-schema.md):
  data/fundamentals_<table>.parquet  cols: code, pubDate, statDate, <fields...>

Run (all 5 tables): .venv/Scripts/python.exe scripts/pull_fundamentals.py
Subset:             .venv/Scripts/python.exe scripts/pull_fundamentals.py growth balance
"""
from __future__ import annotations

import logging
import socket
import sys
import time
from pathlib import Path

import pandas as pd

socket.setdefaulttimeout(20)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("scripts/pull_fundamentals.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pull_fund")

_TABLE_TO_BS_FN = {
    "profit": "query_profit_data",
    "growth": "query_growth_data",
    "balance": "query_balance_data",
    "cash_flow": "query_cash_flow_data",
    "dupont": "query_dupont_data",
}
_N_QUARTERS = 12  # ~3 years, covers the 500-bar backtest window with PIT lookback


def _to_bs_code(code: str) -> str:
    if code.startswith(("60", "68")):
        return f"sh.{code}"
    if code.startswith(("00", "30")):
        return f"sz.{code}"
    if code.startswith(("8", "43")):
        return f"bj.{code}"
    return f"sh.{code}"


def _recent_quarters(n: int) -> list[tuple[int, int]]:
    today = pd.Timestamp.today()
    out: list[tuple[int, int]] = []
    y, q = today.year, ((today.month - 1) // 3) + 1
    for _ in range(n):
        out.append((y, q))
        q -= 1
        if q == 0:
            q, y = 4, y - 1
    return out


def _save(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    for c in ("pubDate", "statDate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["pubDate"]).reset_index(drop=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def pull_table(table: str, codes: list[str], quarters: list[tuple[int, int]]) -> None:
    import baostock as bs

    out = Path("data") / f"fundamentals_{table}.parquet"
    fn_name = _TABLE_TO_BS_FN[table]

    rows: list[dict] = []
    done: set[str] = set()
    if out.exists():
        prev = pd.read_parquet(out)
        prev["code"] = prev["code"].astype(str).str.zfill(6)
        rows = prev.to_dict("records")
        done = set(prev["code"])
    todo = [c for c in codes if c not in done]
    log.info("[%s] %d codes, %d done, %d to pull", table, len(codes), len(done), len(todo))
    if not todo:
        log.info("[%s] already complete (%d rows)", table, len(rows))
        return

    def _login_retry(attempts: int = 6, base_delay: float = 20.0) -> bool:
        """Login with exponential backoff. baostock throttles after long pulls
        ('网络接收错误'); wait it out rather than skipping the whole table."""
        for a in range(1, attempts + 1):
            try:
                bs.logout()
            except Exception:
                pass
            lg = bs.login()
            if lg.error_code == "0":
                return True
            delay = min(base_delay * (2 ** (a - 1)), 600.0)
            log.warning("[%s] login failed (%s); attempt %d/%d, sleeping %.0fs",
                        table, lg.error_msg, a, attempts, delay)
            time.sleep(delay)
        return False

    if not _login_retry():
        log.error("[%s] login unrecoverable; skipping table (resume later)", table)
        _save(rows, out)
        return
    fn = getattr(bs, fn_name)
    t0 = time.time()
    consec_fail = 0  # consecutive query failures (exception OR bad error_code)
    try:
        for i, code in enumerate(todo, 1):
            bs_code = _to_bs_code(code)
            for year, q in quarters:
                try:
                    rs = fn(code=bs_code, year=year, quarter=q)
                    if rs.error_code != "0":
                        # Not an exception — baostock returns a bad code when the
                        # session degrades. Count it so we can back off (the old
                        # bare `continue` let 875 stocks fly by producing nothing).
                        consec_fail += 1
                        raise RuntimeError(f"error_code={rs.error_code} {rs.error_msg}")
                    consec_fail = 0
                    while rs.next():
                        rec = dict(zip(rs.fields, rs.get_row_data()))
                        rec["code"] = code
                        rows.append(rec)
                except Exception as e:  # noqa: BLE001
                    if not isinstance(e, RuntimeError):
                        consec_fail += 1
                    # Sustained failure → baostock throttled; save, back off, relogin.
                    if consec_fail >= 5:
                        log.warning("[%s] %d consecutive failures (last: %s); "
                                    "saving + backing off", table, consec_fail, e)
                        _save(rows, out)
                        if not _login_retry():
                            log.error("[%s] still down after backoff at code %s; "
                                      "aborting table (resume later)", table, code)
                            _save(rows, out)
                            return
                        fn = getattr(bs, fn_name)
                        consec_fail = 0
            if i % 250 == 0:
                rate = (time.time() - t0) / i
                eta = rate * (len(todo) - i) / 60
                log.info("[%s] %d/%d (%.0f%%) rows=%d ETA=%.0fmin",
                         table, i, len(todo), 100 * i / len(todo), len(rows), eta)
                _save(rows, out)
    finally:
        try:
            bs.logout()
        except Exception:
            pass
        _save(rows, out)
    log.info("[%s] DONE in %.0fs: %d rows -> %s", table, time.time() - t0, len(rows), out)


def main(argv: list[str]) -> int:
    from stockpool.fetcher import list_universe

    tables = argv if argv else ["profit", "growth", "balance", "cash_flow", "dupont"]
    bad = [t for t in tables if t not in _TABLE_TO_BS_FN]
    if bad:
        log.error("unknown table(s): %s; valid: %s", bad, list(_TABLE_TO_BS_FN))
        return 2

    codes = list_universe()["code"].astype(str).str.zfill(6).tolist()
    quarters = _recent_quarters(_N_QUARTERS)
    log.info("pulling tables=%s | %d codes × %d quarters", tables, len(codes), len(quarters))
    for table in tables:
        pull_table(table, codes, quarters)
    log.info("ALL DONE: %s", tables)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
