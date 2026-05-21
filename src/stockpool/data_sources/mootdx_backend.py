"""mootdx 后端 (通达信行情服务器, TCP 协议)。

特点:
- 不依赖 HTTP 爬虫,稳定性高;
- 支持拉到当日盘中/收盘数据(几分钟级延迟);
- 单次请求最多 800 根 K 线,我们按需要 offset。
"""
from __future__ import annotations

import logging
import threading
import time

import pandas as pd

log = logging.getLogger(__name__)

_RETRY_DELAYS = [2, 4, 8]
_MAX_BARS_PER_CALL = 800  # mootdx 单次上限
_FREQ_DAILY = 9            # mootdx frequency 编码: 9 = 日线

# Per-thread Quotes client. mootdx 的 TCP 连接非线程安全,共享会互相打架;
# 每线程一个 client 才能稳定并发 bulk-fetch 全市场。
_local = threading.local()


def _get_client(force_new: bool = False):
    if force_new or getattr(_local, "client", None) is None:
        from mootdx.quotes import Quotes
        _local.client = Quotes.factory(market="std")
    return _local.client


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # mootdx 不同版本可能同时返回 vol 和 volume,优先用 volume
    if "volume" not in out.columns and "vol" in out.columns:
        out = out.rename(columns={"vol": "volume"})
    if "date" not in out.columns and "datetime" in out.columns:
        out = out.rename(columns={"datetime": "date"})
    # 去重列(若 rename 后仍有同名列)
    out = out.loc[:, ~out.columns.duplicated()]
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out[["date", "open", "high", "low", "close", "volume"]]
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # 丢弃 TDX "未开盘"占位行(末根 bar 成交量近零 & OHLC 同价)
    if len(out) > 1:
        last = out.iloc[-1]
        if last["volume"] < 1 and last["open"] == last["high"] == last["low"] == last["close"]:
            out = out.iloc[:-1].reset_index(drop=True)
    return out


def _offset_for_start(start: str | None) -> int:
    if start is None:
        return _MAX_BARS_PER_CALL
    days_back = (pd.Timestamp.today().normalize() - pd.to_datetime(start)).days
    # 日历日 → 交易日的粗估 (~0.7 倍) + 缓冲
    est_bars = int(days_back * 0.75) + 10
    return max(5, min(est_bars, _MAX_BARS_PER_CALL))


def _call_with_retry(method_name: str, *args, **kwargs):
    """Retry helper. Re-resolves the (per-thread) client on each retry so that
    a failed connection picks a fresh server node next time."""
    last_err: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            client = _get_client(force_new=(attempt > 1))
            fn = getattr(client, method_name)
            raw = fn(*args, **kwargs)
            if raw is None or raw.empty:
                raise RuntimeError("empty result")
            return raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("mootdx %s attempt %d/%d failed: %s",
                        method_name, attempt, len(_RETRY_DELAYS), e)
            if attempt < len(_RETRY_DELAYS):
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def fetch_stock(code: str, start: str | None = None) -> pd.DataFrame:
    """拉 A 股日线 (前复权语义未处理: mootdx bars 返回不复权价)。

    Note:
        mootdx 的 bars 默认返回不复权价。本项目使用价格做技术指标(MA/MACD/...)
        和回测,在没有除权事件的票上影响有限;若需精确复权,后续可加 xdxr 处理。
    """
    offset = _offset_for_start(start)
    raw = _call_with_retry("bars", symbol=code, frequency=_FREQ_DAILY, offset=offset)
    out = _normalize(raw)
    if start is not None:
        out = out[out["date"] >= pd.to_datetime(start)]
    return out.reset_index(drop=True)


def fetch_index(symbol: str) -> pd.DataFrame:
    """拉指数日线。symbol 形如 'sh000001' / 'sz399001'。"""
    sym = symbol[2:] if symbol[:2].lower() in ("sh", "sz") else symbol
    raw = _call_with_retry("index", symbol=sym, frequency=_FREQ_DAILY, offset=_MAX_BARS_PER_CALL)
    return _normalize(raw)


# 通达信行业指数代码 (88xxxx 系列)。新增映射时直接在这里加一行即可。
# 完整列表参考: 通达信 -> 板块涨幅排名 -> 行业 (右键查看代码)
_TDX_INDUSTRY_CODES: dict[str, str] = {
    "化工": "880305",
    "半导体": "880491",
    "工程机械": "880324",
    "通用机械": "880335",
    "装修装饰": "880482",
    "电力": "880350",
    # 常见行业(便于扩展):
    "石油": "880301",
    "钢铁": "880318",
    "煤炭": "880320",
    "有色金属": "880421",
    "银行": "880471",
    "证券": "880472",
    "保险": "880473",
    "房地产": "880451",
    "食品饮料": "880380",
    "白酒": "880387",
    "医药": "880400",
    "汽车": "880430",
    "家电": "880440",
}


def _resolve_sector_code(name_or_code: str) -> str:
    """支持两种输入: 行业名(查表) 或 6 位 TDX 代码 (88xxxx)。"""
    s = name_or_code.strip()
    if s.isdigit() and len(s) == 6 and s.startswith("88"):
        return s
    code = _TDX_INDUSTRY_CODES.get(s)
    if code is None:
        raise KeyError(
            f"未找到行业 '{s}' 对应的通达信代码。请在 "
            f"stockpool.data_sources.mootdx_backend._TDX_INDUSTRY_CODES 中补充,"
            f"或直接在 config.yaml 的 stocks[].sector 填 6 位 TDX 代码 (如 880305)。"
        )
    return code


_MARKET_SZ = 0
_MARKET_SH = 1

# 主板 / 中小板 / 创业板。排除 688*(科创)和 8*/4*(北交所)。
_SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")
_SH_PREFIXES = ("600", "601", "603", "605")


def _is_a_share(code: str, market: int) -> bool:
    if market == _MARKET_SH:
        return any(code.startswith(p) for p in _SH_PREFIXES)
    return any(code.startswith(p) for p in _SZ_PREFIXES)


def list_a_shares() -> pd.DataFrame:
    """List all A-share stocks (excluding ST, 科创板 688*, 北交所 8*/4*).

    Returns DataFrame with columns: code, name, market ('SH'/'SZ').

    Note: mootdx 返回的 name 字段是双重编码乱码,无法可靠按中文过滤(如"退"字)。
    本函数仅按 ASCII 子串过滤 ST/\\*ST(覆盖大部分风险警示股),其余依赖代码前缀。
    """
    parts: list[pd.DataFrame] = []
    for market, label in [(_MARKET_SZ, "SZ"), (_MARKET_SH, "SH")]:
        raw = _call_with_retry("stocks", market=market)
        df = raw.copy()
        df["market"] = label
        df = df[df["code"].apply(lambda c: _is_a_share(c, market))]
        # Filter ST/*ST via ASCII substring (name encoding is mangled but ASCII survives)
        name_upper = df["name"].astype(str).str.upper()
        df = df[~name_upper.str.contains("ST", na=False)]
        parts.append(df[["code", "name", "market"]])
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates("code").reset_index(drop=True)
    return out


def fetch_sector(name_or_code: str) -> pd.DataFrame:
    """拉通达信行业板块日线。

    输入可以是行业名(如 '化工')或 6 位 TDX 代码(如 '880305')。
    底层走 ``client.index()``,因为 TDX 把行业板块也建模成指数。
    """
    code = _resolve_sector_code(name_or_code)
    raw = _call_with_retry("index", symbol=code, frequency=_FREQ_DAILY, offset=_MAX_BARS_PER_CALL)
    return _normalize(raw)
