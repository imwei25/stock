"""baostock 后端 (无 token, 收盘后约 18:00 更新当日数据)。

线程安全说明 (P2-19): baostock 底层是全局单 socket,登录状态也是全局的。
并发查询会导致请求/响应交错(轻则报错,重则 A 股票拿到 B 的数据)。
因此用模块级 _LOCK 把"确保登录 + 单次查询 + 取完所有行"整段做成原子临界区。
"""
from __future__ import annotations

import logging
import threading
import time

import pandas as pd

log = logging.getLogger(__name__)

_RETRY_DELAYS = [2, 4, 8]
_LOCK = threading.Lock()
_logged_in = False

_STOCK_FIELDS = "date,open,high,low,close,volume,tradestatus"  # tradestatus: "1"=正常交易 "0"=停牌
_INDEX_FIELDS = "date,open,high,low,close,volume"  # 指数查询不支持 tradestatus
_OUT_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _ensure_login() -> None:
    """调用方必须已持有 _LOCK。"""
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


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "open": pd.Series(dtype="float64"),
        "high": pd.Series(dtype="float64"),
        "low": pd.Series(dtype="float64"),
        "close": pd.Series(dtype="float64"),
        "volume": pd.Series(dtype="float64"),
    })


def _query(bs_code: str, start: str, fields: str, allow_empty: bool) -> pd.DataFrame:
    """单次查询 + 取行。调用方必须已持有 _LOCK(单 socket,不可重入)。

    allow_empty=True(增量场景,调用方给定了 start)时空结果返回空 DataFrame;
    否则空结果 raise(全量空 = 代码错误或退市,应 fail loud)。
    """
    import baostock as bs
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(
        bs_code, fields,
        start_date=start, end_date=end,
        frequency="d", adjustflag="1",  # 1 = 后复权(锚在上市日,历史不变,增量追加自洽)
    )
    if rs.error_code != "0":
        raise RuntimeError(f"baostock query failed for {bs_code}: {rs.error_msg}")
    rows: list[list[str]] = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        if allow_empty:
            return _empty_frame()
        raise RuntimeError(f"baostock returned empty for {bs_code}")
    df = pd.DataFrame(rows, columns=rs.fields)
    if "tradestatus" in df.columns:
        # P2-20: 停牌日 baostock 返回 volume=0 填充行,mootdx 停牌日无行;
        # 过滤掉非正常交易行,统一跨源时序形态。
        df = df[df["tradestatus"] == "1"].drop(columns=["tradestatus"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
    if df.empty:
        if allow_empty:
            return _empty_frame()
        raise RuntimeError(f"baostock returned empty for {bs_code}")
    return df[_OUT_COLUMNS].reset_index(drop=True)


def _query_with_retry(bs_code: str, start: str, fields: str,
                      allow_empty: bool = False) -> pd.DataFrame:
    global _logged_in
    last_err: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            with _LOCK:
                _ensure_login()
                return _query(bs_code, start, fields, allow_empty)
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("baostock attempt %d/%d for %s failed: %s",
                        attempt, len(_RETRY_DELAYS), bs_code, e)
            if attempt < len(_RETRY_DELAYS):
                with _LOCK:
                    _logged_in = False  # 强制重新登录
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def fetch_stock(code: str, start: str | None = None) -> pd.DataFrame:
    bs_code = _bs_stock_code(code)
    start_date = pd.to_datetime(start).strftime("%Y-%m-%d") if start else "1990-01-01"
    # P2-21: 增量场景(给定 start)空结果合法 —— IPO 不足 history_days / 周末无新数据,
    # 返回空 DataFrame 由上游 fetcher 走"无新数据"合并路径;全量空仍 fail loud。
    return _query_with_retry(bs_code, start_date, _STOCK_FIELDS,
                             allow_empty=start is not None)


def fetch_index(symbol: str) -> pd.DataFrame:
    bs_code = _bs_index_code(symbol)
    # 指数 volume 字段可能为 0;保留即可
    return _query_with_retry(bs_code, "1990-01-01", _INDEX_FIELDS)
