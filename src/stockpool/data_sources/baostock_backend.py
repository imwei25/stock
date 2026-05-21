"""baostock 后端 (无 token, 收盘后约 18:00 更新当日数据)。"""
from __future__ import annotations

import logging
import time

import pandas as pd

log = logging.getLogger(__name__)

_RETRY_DELAYS = [2, 4, 8]
_logged_in = False


def _ensure_login() -> None:
    global _logged_in
    if _logged_in:
        return
    import baostock as bs
    rs = bs.login()
    if rs.error_code != "0":
        raise RuntimeError(f"baostock login failed: {rs.error_msg}")
    _logged_in = True


def _bs_stock_code(code: str) -> str:
    """605589 → sh.605589; 000528 → sz.000528."""
    if code.startswith(("6", "5", "9")):
        return f"sh.{code}"
    return f"sz.{code}"


def _bs_index_code(symbol: str) -> str:
    """'sh000001' → 'sh.000001'; 'sz399001' → 'sz.399001'."""
    if symbol[:2].lower() in ("sh", "sz") and "." not in symbol:
        return f"{symbol[:2].lower()}.{symbol[2:]}"
    return symbol


def _query(bs_code: str, start: str, fields: str) -> pd.DataFrame:
    import baostock as bs
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(
        bs_code, fields,
        start_date=start, end_date=end,
        frequency="d", adjustflag="2",  # 2 = 前复权
    )
    if rs.error_code != "0":
        raise RuntimeError(f"baostock query failed for {bs_code}: {rs.error_msg}")
    rows: list[list[str]] = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        raise RuntimeError(f"baostock returned empty for {bs_code}")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _query_with_retry(bs_code: str, start: str, fields: str) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            _ensure_login()
            return _query(bs_code, start, fields)
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("baostock attempt %d/%d for %s failed: %s",
                        attempt, len(_RETRY_DELAYS), bs_code, e)
            if attempt < len(_RETRY_DELAYS):
                global _logged_in
                _logged_in = False  # 强制重新登录
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def fetch_stock(code: str, start: str | None = None) -> pd.DataFrame:
    bs_code = _bs_stock_code(code)
    start_date = pd.to_datetime(start).strftime("%Y-%m-%d") if start else "1990-01-01"
    return _query_with_retry(bs_code, start_date, "date,open,high,low,close,volume")


def fetch_index(symbol: str) -> pd.DataFrame:
    bs_code = _bs_index_code(symbol)
    # 指数 volume 字段可能为 0;保留即可
    return _query_with_retry(bs_code, "1990-01-01", "date,open,high,low,close,volume")
