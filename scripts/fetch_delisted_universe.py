"""Fetch daily bars for delisted A-share stocks (survivorship-bias fix).

The mootdx universe cache (``fetch-universe``) only lists *currently listed*
stocks, so every 15-yr backtest silently drops the ~258 stocks delisted inside
the window — a structural survivorship bias. Eastmoney (via akshare
``stock_zh_a_hist``) still serves full daily history for delisted codes up to
their delist date, so we backfill them into the same cache.

Semantics kept consistent with the mootdx cache:
  * unadjusted prices (mootdx bars are 不复权)
  * volume in 手 (eastmoney daily volume unit is 手, same as mootdx)
  * schema: date / open / high / low / close / volume

Output:
  * ``data/<code>_daily.parquet`` for each delisted code (skipped if present)
  * ``data/universe_delisted.parquet`` — code / name / list_date / delist_date
    manifest. Deliberately NOT merged into ``universe.parquet``: production
    name_map / ab-pool stay clean; research paths pick the bars up through
    ``load_universe_cache`` (which globs ``*_daily.parquet``).

Usage:
  python scripts/fetch_delisted_universe.py [--start 20100101] [--sleep 0.3]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

_COLMAP = {
    "日期": "date", "开盘": "open", "最高": "high",
    "最低": "low", "收盘": "close", "成交量": "volume",
}


def build_delist_manifest() -> pd.DataFrame:
    """SSE + SZSE delisted lists via akshare, filtered to 60/00/30 codes."""
    import akshare as ak

    sh = ak.stock_info_sh_delist().rename(columns={
        "公司代码": "code", "公司简称": "name",
        "上市日期": "list_date", "暂停上市日期": "delist_date",
    })
    sz = ak.stock_info_sz_delist(symbol="终止上市公司").rename(columns={
        "证券代码": "code", "证券简称": "name",
        "上市日期": "list_date", "终止上市日期": "delist_date",
    })
    cols = ["code", "name", "list_date", "delist_date"]
    dl = pd.concat([sh[cols], sz[cols]], ignore_index=True)
    dl["code"] = dl["code"].astype(str).str.zfill(6)
    dl["list_date"] = pd.to_datetime(dl["list_date"], errors="coerce")
    dl["delist_date"] = pd.to_datetime(dl["delist_date"], errors="coerce")
    dl = dl.dropna(subset=["delist_date"])
    dl = dl[dl["code"].str.match(r"^(60|00|30)")]
    return dl.drop_duplicates(subset="code").reset_index(drop=True)


def fetch_one(code: str, start: str) -> pd.DataFrame | None:
    import akshare as ak

    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                            start_date=start, end_date="20991231", adjust="")
    if df is None or df.empty:
        return None
    df = df.rename(columns=_COLMAP)[list(_COLMAP.values())]
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Basic OHLCV sanity: drop unusable bars (suspension placeholders etc.).
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    df = df.sort_values("date").reset_index(drop=True)
    return df if len(df) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20100101")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--window-start", default="2011-04-01",
                    help="only fetch stocks delisted on/after this date")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    dl = build_delist_manifest()
    in_win = dl[dl["delist_date"] >= pd.Timestamp(args.window_start)]
    print(f"delisted total={len(dl)}, in-window={len(in_win)}")

    ok, empty, failed, skipped = 0, 0, 0, 0
    for i, row in enumerate(in_win.itertuples(), 1):
        out_path = DATA_DIR / f"{row.code}_daily.parquet"
        if out_path.exists():
            skipped += 1
            continue
        try:
            df = fetch_one(row.code, args.start)
        except Exception as e:  # noqa: BLE001 — per-stock isolation
            print(f"[{i}/{len(in_win)}] {row.code} {row.name}: FAILED {e}")
            failed += 1
            time.sleep(args.sleep)
            continue
        if df is None:
            print(f"[{i}/{len(in_win)}] {row.code} {row.name}: no data")
            empty += 1
        else:
            df.to_parquet(out_path)
            ok += 1
            if ok % 25 == 0:
                print(f"[{i}/{len(in_win)}] {ok} fetched "
                      f"(last: {row.code} {row.name}, {len(df)} bars)")
        time.sleep(args.sleep)

    in_win.to_parquet(DATA_DIR / "universe_delisted.parquet")
    print(f"done: ok={ok} empty={empty} failed={failed} skipped={skipped}")
    print(f"manifest: {DATA_DIR / 'universe_delisted.parquet'} ({len(in_win)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
