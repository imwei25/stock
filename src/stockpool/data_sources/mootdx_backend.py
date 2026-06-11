"""mootdx 后端 (通达信行情服务器, TCP 协议)。

特点:
- 不依赖 HTTP 爬虫,稳定性高;
- 支持拉到当日盘中/收盘数据(几分钟级延迟);
- 单次请求最多 800 根 K 线,fetch_stock 内部按 start=0/800/1600/... 分页
  拼接,直到覆盖目标起始日期 / 凑够 min_bars / 数据到头(硬上限 4800 根);
- 个股 volume 原始单位为"手",统一放大为"股"(与 baostock 口径一致);
  指数/板块 volume 保持 TDX 原始单位(只做比值类用途);
- 复权:bars 原始数据为不复权价,本模块用同源 xdxr 事件(TCP)做
  **段内锚定后复权**——段首因子=1,事件因子只依赖段内 prev_close,
  增量段由 fetcher 用缓存重叠 bar 锚定到既有尺度,天然无接缝。
  (不用 mootdx 自带 to_adjust:它走新浪 HTTP 拉因子,且对部分窗口
  有 fillna(1.0) 边界 bug。)
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_RETRY_DELAYS = [2, 4, 8]
_MAX_BARS_PER_CALL = 800   # mootdx 单次上限
_MAX_TOTAL_BARS = 4800     # 分页拼接硬上限 (6 页),防止异常参数下失控翻页
_FREQ_DAILY = 9            # mootdx frequency 编码: 9 = 日线

# Per-thread Quotes client. mootdx 的 TCP 连接非线程安全,共享会互相打架;
# 每线程一个 client 才能稳定并发 bulk-fetch 全市场。
_local = threading.local()


def _get_client(force_new: bool = False):
    if force_new or getattr(_local, "client", None) is None:
        from mootdx.quotes import Quotes
        _local.client = Quotes.factory(market="std")
    return _local.client


def _normalize(df: pd.DataFrame, scale_volume: bool = True) -> pd.DataFrame:
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
    # 丢弃 TDX "未开盘"占位行(末根 bar 成交量近零 & OHLC 同价)。
    # 注意:volume < 1 的判断以原始单位"手"为准,必须在下方放大为"股"之前执行。
    if len(out) > 1:
        last = out.iloc[-1]
        if last["volume"] < 1 and last["open"] == last["high"] == last["low"] == last["close"]:
            out = out.iloc[:-1].reset_index(drop=True)
    if scale_volume:
        # TDX 个股 volume 原始单位为"手"(1 手 = 100 股);×100 统一为"股",
        # 与 baostock 口径一致(P1-6)。指数/板块路径传 scale_volume=False。
        out["volume"] = out["volume"] * 100.0
    return out


def _offset_for_start(start: str | None) -> int:
    if start is None:
        return _MAX_BARS_PER_CALL
    days_back = (pd.Timestamp.today().normalize() - pd.to_datetime(start)).days
    # 日历日 → 交易日的粗估 (~0.7 倍) + 缓冲
    est_bars = int(days_back * 0.75) + 10
    return max(5, min(est_bars, _MAX_BARS_PER_CALL))


def _fetch_bars_paged(
    code: str,
    start: str | None = None,
    min_bars: int | None = None,
) -> pd.DataFrame:
    """分页拉日线 K 并拼接(mootdx 单次最多 800 根)。

    ``client.bars`` 的 ``start`` 是"从最近一根往回数的偏移位置":
    start=0 取最新一页,start=800 取再往前一页,以此类推。

    终止条件(任一满足即停):
      ① 已覆盖到 ``start`` 目标起始日期;
      ② 累计根数达到 ``min_bars``;
      ③ 服务器返回空页/短页(历史数据到头);
      ④ 累计达到 ``_MAX_TOTAL_BARS`` 硬上限,防失控。
    两者都未给时保持旧行为:只拉最近一页(≤800 根)。
    """
    target = pd.to_datetime(start) if start is not None else None
    pages: list[pd.DataFrame] = []
    total = 0
    pos = 0
    while pos < _MAX_TOTAL_BARS:
        # 首页按 start 粗估根数(增量拉取通常只差几根,不必整页 800)
        if pos == 0:
            req = _offset_for_start(start)
        else:
            req = min(_MAX_BARS_PER_CALL, _MAX_TOTAL_BARS - pos)
        try:
            raw = _call_with_retry("bars", symbol=code, frequency=_FREQ_DAILY,
                                   start=pos, offset=req)
        except Exception:
            if not pages:
                raise  # 首页都拉不到 → 真网络错误,向上抛
            raw = None  # 翻页越过最早历史时服务器返回空(重试耗尽)→ 视为到头
        if raw is None or len(raw) == 0:
            break  # ③ 空页:数据到头
        pages.append(raw)
        total += len(raw)
        if len(raw) < req:
            break  # ③ 短页:数据到头
        date_col = "date" if "date" in raw.columns else "datetime"
        covered = target is not None and pd.to_datetime(raw[date_col]).min() <= target
        enough = min_bars is not None and total >= min_bars
        if covered or enough:
            break  # ① / ②
        if target is None and min_bars is None:
            break  # 兼容旧行为:无明确需求只拉最近一页
        pos += req

    if not pages:
        raise RuntimeError(f"mootdx bars 返回空: {code}")
    out = _normalize(pd.concat(pages, ignore_index=True))
    if min_bars is not None and len(out) < min_bars:
        log.warning("mootdx %s: 历史数据到头,仅拉到 %d 根 (请求 min_bars=%d)",
                    code, len(out), min_bars)
    return out


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


_XDXR_EVENT_COLS = ["date", "fenhong", "peigu", "peigujia", "songzhuangu"]


def _fetch_xdxr(code: str) -> pd.DataFrame:
    """拉除权除息事件 (category==1: 分红/送转/配股),可能为空。

    返回列: date, fenhong, peigu, peigujia, songzhuangu(均为每 10 股口径)。
    网络失败时 raise(宁可整次拉取失败,也不能把未复权数据当 hfq 写入缓存)。
    """
    last_err: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            client = _get_client(force_new=(attempt > 1))
            raw = client.xdxr(symbol=code)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("mootdx xdxr attempt %d/%d for %s failed: %s",
                        attempt, len(_RETRY_DELAYS), code, e)
            if attempt < len(_RETRY_DELAYS):
                time.sleep(delay)
    else:
        assert last_err is not None
        raise last_err

    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_XDXR_EVENT_COLS)
    df = raw.reset_index() if "date" in getattr(raw.index, "names", []) else raw.copy()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df[df.get("category", pd.Series(dtype=float)) == 1].copy()
    if df.empty:
        return pd.DataFrame(columns=_XDXR_EVENT_COLS)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ("fenhong", "peigu", "peigujia", "songzhuangu"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)
    return df[_XDXR_EVENT_COLS].sort_values("date").reset_index(drop=True)


def _apply_hfq(bars: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """段内锚定后复权:段首因子=1,只用段内可见的除权事件。

    每个事件 e(每 10 股分红 fenhong、送转 songzhuangu、配股 peigu @ peigujia):
        理论除权价 P_ex = (P_prev*10 - fenhong + peigu*peigujia)
                          / (10 + peigu + songzhuangu)
        因子 ratio_e = P_prev / P_ex,自事件日起乘到所有后续 bar。
    P_prev 取段内事件日前最后一根 bar 的原始 close;事件落在段首 bar 或
    段外时只贡献常数尺度,在段内锚定语义下直接忽略(增量段由 fetcher
    用缓存重叠 bar 锚定,全量段以窗口起点为基准)。

    OHLC 同步缩放;volume 保持原始值(与 baostock/akshare 口径一致)。
    """
    if bars.empty or events is None or events.empty:
        return bars
    out = bars.reset_index(drop=True).copy()
    dates = out["date"].to_numpy()
    raw_close = out["close"].to_numpy(dtype=float)
    factor = np.ones(len(out))

    for ev in events.sort_values("date").itertuples(index=False):
        ev_date = np.datetime64(pd.Timestamp(ev.date))
        idx = int(np.searchsorted(dates, ev_date, side="left"))
        if idx <= 0 or idx >= len(out):
            continue  # 段首/段前事件 → 常数尺度,忽略;段后事件无 bar 可调
        p_prev = raw_close[idx - 1]
        denom = 10.0 + ev.peigu + ev.songzhuangu
        p_ex = (p_prev * 10.0 - ev.fenhong + ev.peigu * ev.peigujia) / denom
        if not np.isfinite(p_ex) or p_ex <= 0 or p_prev <= 0:
            log.warning("xdxr event at %s yields invalid ex-price (%s); skipped",
                        ev.date, p_ex)
            continue
        factor[idx:] *= p_prev / p_ex

    for col in ("open", "high", "low", "close"):
        out[col] = out[col].to_numpy(dtype=float) * factor
    return out


def fetch_stock(
    code: str,
    start: str | None = None,
    min_bars: int | None = None,
) -> pd.DataFrame:
    """拉 A 股日线,段内锚定后复权 (hfq)。volume 单位为"股"。

    mootdx 单次最多 800 根,内部经 ``_fetch_bars_paged`` 分页拼接:
    给 ``start`` 时翻页直到覆盖目标起始日期;给 ``min_bars`` 时翻页直到
    凑够根数(或数据到头);都不给时保持旧行为只拉最近一页。

    mootdx bars 返回不复权价;本函数在**完整拼接后的窗口**上叠加同源
    xdxr 事件做段内 hfq:返回段的首根 bar 即原始价(因子=1),段内除权
    事件之后的 bar 已被放大到连续口径。增量调用方(fetcher.fetch_daily)
    用缓存重叠 bar 把本段锚定到既有缓存尺度。
    """
    out = _fetch_bars_paged(code, start=start, min_bars=min_bars)
    if start is not None:
        out = out[out["date"] >= pd.to_datetime(start)]
    out = out.reset_index(drop=True)
    return _apply_hfq(out, _fetch_xdxr(code))


def fetch_index(symbol: str) -> pd.DataFrame:
    """拉指数日线。symbol 形如 'sh000001' / 'sz399001'。

    指数 volume 跨源单位不保证一致(下游只做量比等比值类用途),
    保持 TDX 原始单位,不做手→股放大。
    """
    sym = symbol[2:] if symbol[:2].lower() in ("sh", "sz") else symbol
    raw = _call_with_retry("index", symbol=sym, frequency=_FREQ_DAILY, offset=_MAX_BARS_PER_CALL)
    return _normalize(raw, scale_volume=False)


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
    """List all A-share stocks (excluding 科创板 688*, 北交所 8*/4*).

    Returns DataFrame with columns: code, name, market ('SH'/'SZ').

    Note(P0-4 ②):**不再按名称剔除 ST**。按"当前是否 ST"整段剔除历史样本
    是用未来信息筛历史(2023 年健康、2026 年戴帽的票,其 2023 年截面也被剔),
    且 mootdx 的 name 是双重编码乱码,ASCII "ST" 匹配本就不可靠(P3-4)。
    ST 状态由 ``stockpool.ipo_dates.load_or_build_stock_basics`` 的干净
    baostock 名单标记,应用层(Pool B/推荐池)在**当下决策时**自行剔除。
    """
    parts: list[pd.DataFrame] = []
    for market, label in [(_MARKET_SZ, "SZ"), (_MARKET_SH, "SH")]:
        raw = _call_with_retry("stocks", market=market)
        df = raw.copy()
        df["market"] = label
        df = df[df["code"].apply(lambda c: _is_a_share(c, market))]
        parts.append(df[["code", "name", "market"]])
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates("code").reset_index(drop=True)
    return out


def fetch_sector(name_or_code: str) -> pd.DataFrame:
    """拉通达信行业板块日线。

    输入可以是行业名(如 '化工')或 6 位 TDX 代码(如 '880305')。
    底层走 ``client.index()``,因为 TDX 把行业板块也建模成指数。
    板块 volume 跨源单位不保证一致(只做比值类用途),保持原始单位。
    """
    code = _resolve_sector_code(name_or_code)
    raw = _call_with_retry("index", symbol=code, frequency=_FREQ_DAILY, offset=_MAX_BARS_PER_CALL)
    return _normalize(raw, scale_volume=False)
