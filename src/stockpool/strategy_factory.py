"""Strategy factory + ML-strategy backtest helper.

Decouples the CLI from concrete strategy classes:

  * ``build_strategy(cfg, pool_data=None, current_stock_code=None)`` —
    return the strategy implementation selected by ``cfg.strategy.name``.
  * ``simulate_ml_equity_curve(...)`` — mirror of ``backtest_composite.
    simulate_equity_curve``'s output shape but driven by ``MLFactorStrategy``;
    lets ``cmd_backtest`` route ML runs without touching the report renderer.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Mapping

import pandas as pd

log = logging.getLogger(__name__)

from stockpool.backtest_composite import EquityResult
from stockpool.backtesting import (
    BacktestEngine,
    CompositeVerdictStrategy,
    MLFactorStrategy,
    MultiLotBacktestEngine,
    Strategy,
    TradeCosts,
    buy_and_hold_baseline,
)
from stockpool.backtesting.sizing import FixedLotSizer, LotSizer
from stockpool.config import AppConfig


def build_strategy(
    cfg: AppConfig,
    pool_data: Mapping[str, pd.DataFrame] | None = None,
    current_stock_code: str | None = None,
    factor_panel: Mapping[str, pd.DataFrame] | None = None,
    close_panel: pd.DataFrame | None = None,
    shared_cache: dict | None = None,
) -> Strategy:
    """Construct the strategy referenced by ``cfg.strategy.name``.

    Args:
        cfg: loaded ``AppConfig``.
        pool_data: required for ``ml_factor`` strategies in pooled mode. Pass
            the full daily-history dict; the strategy applies per-bar
            look-ahead-safe truncation internally.
        current_stock_code: which stock in ``pool_data`` is the one being
            backtested — excluded from pool truncation to avoid double-counting.
        factor_panel: precomputed ``{factor_name: T×N DataFrame}``. Pass this
            when iterating many stocks against the same pool, so the (potentially
            expensive) panel-wide factor computation runs once. If omitted and
            ``pool_data`` is provided in pooled mode, the panel is built here.
    """
    name = cfg.strategy.name
    if name == "composite_verdict":
        return CompositeVerdictStrategy(
            weights=cfg.weights,
            scoring=cfg.scoring,
            verdicts_cfg=cfg.verdicts,
            indicators_cfg=cfg.indicators,
        )
    if name == "ml_factor":
        # 在 pooled 模式 + 有 pool_data 时,预算因子面板,让 WQ101 cross-sec
        # 因子在 predict 阶段拿到真实横截面值(否则会退化为 1-stock 常数)。
        if (
            factor_panel is None
            and cfg.strategy.ml_factor.panel_mode == "pooled"
            and pool_data
        ):
            factor_panel = build_factor_panel(cfg.strategy.ml_factor.factors, pool_data)
        if (
            close_panel is None
            and cfg.strategy.ml_factor.panel_mode == "pooled"
            and pool_data
        ):
            close_panel = build_close_panel(pool_data)
        return MLFactorStrategy(
            cfg=cfg.strategy.ml_factor,
            pool_data=pool_data,
            current_stock_code=current_stock_code,
            factor_panel=factor_panel,
            close_panel=close_panel,
            cache_dir=cfg.data.cache_dir,
            shared_cache=shared_cache,
        )
    raise ValueError(f"unknown strategy: {name!r}")


def build_close_panel(
    pool_data: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """从 ``{code: daily_df}`` 装 T×N close 宽表(行 = union dates,列 = code)。

    用于跨股训练时一次性算 forward-return labels,避免每个 refit_bar 重算。
    与 ``build_factor_panel`` 内部使用同一份 OHLCV panel 构造逻辑。
    """
    if not pool_data:
        return pd.DataFrame()
    per_stock: dict[str, pd.Series] = {}
    for code, df in pool_data.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        per_stock[code] = d.set_index("date").sort_index()["close"]
    all_dates = sorted(set().union(*(s.index for s in per_stock.values())))
    idx = pd.DatetimeIndex(all_dates, name="date")
    return pd.DataFrame(
        {code: s.reindex(idx) for code, s in per_stock.items()},
        index=idx,
    )


def build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    preprocess_cfg: "PreprocessConfig | None" = None,
) -> dict[str, pd.DataFrame]:
    """从 ``{code: daily_df}`` 装一个 OHLCV Panel,在 Panel 上算所有因子,
    返回 ``{factor_name: T×N DataFrame}``。

    Look-ahead 安全:因子在第 i 行只用 ``[:i+1]`` 数据(由 Factor 契约保证),
    所以一次性预算整段历史不会泄露未来。

    **不应用 tradability mask** — 时间序列因子需要看真实价格(包括涨停日)。
    Mask 仅在标签 (``forward_return_panel``) 与训练样本筛选上生效,详见
    ``compute_factor_panel`` docstring。

    Args:
        factor_names: 因子名列表。
        pool_data: ``{code: daily_df}``.
        preprocess_cfg: 可选的 ``PreprocessConfig``。非 None 且非全关时,
            对原始因子 panel 运行 winsorize / cs_zscore / industry_neutralize
            流水线(见 ``ml/preprocess.py``)。sector_map 从
            ``factors.context.get_sector_map()`` 读取(caller 责任注入)。
    """
    from stockpool.ml.dataset import compute_factor_panel
    from stockpool.ml import preprocess as preproc_mod

    # 1) 把每股 daily_df → date-indexed,按列拼成宽表
    per_stock: dict[str, pd.DataFrame] = {}
    for code, df in pool_data.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        per_stock[code] = d.set_index("date").sort_index()
    if not per_stock:
        return {}
    all_dates = sorted(set().union(*(d.index for d in per_stock.values())))
    idx = pd.DatetimeIndex(all_dates, name="date")
    panel: dict[str, pd.DataFrame] = {}
    for field in ("open", "high", "low", "close", "volume"):
        panel[field] = pd.DataFrame(
            {code: d[field].reindex(idx) for code, d in per_stock.items()},
            index=idx,
        )

    raw = compute_factor_panel(panel, factor_names)
    if preprocess_cfg is None or preproc_mod._is_all_off(preprocess_cfg):
        return raw

    from stockpool.factors.context import get_sector_map, get_mcap_panel
    from stockpool.factors.registry import list_specs
    sector_map = get_sector_map() or None
    log_mcap = get_mcap_panel() if preprocess_cfg.market_cap_neutralize else None
    types_map = {
        s.base_name: s.types for s in list_specs() if s.base_name in factor_names
    }
    return preproc_mod.apply_preprocess_pipeline(
        raw, preprocess_cfg, sector_map=sector_map, factor_types=types_map,
        n_codes=len(pool_data), log_mcap_panel=log_mcap,
    )


def _load_pit_shares_panel(
    close_panel: pd.DataFrame,
    cache_dir: str | Path,
) -> pd.DataFrame | None:
    """profit 表逐季 PIT totalShare → 与 close_panel 对齐的 T×N 股本面板。

    复用 ``factors.fundamentals._pit_align``(按 pubDate ffill,PIT 保证 +
    同日披露取最新 statDate + ffill 上限)。profit parquet 缺失/无字段时
    返回 None,调用方回退静态快照。**只读缓存,不触发网络**(基本面缓存
    由 fetch 流程/scripts 维护)。
    """
    profit_path = Path(cache_dir) / "fundamentals_profit.parquet"
    if not profit_path.exists():
        return None
    try:
        raw = pd.read_parquet(profit_path)
        if raw.empty or "totalShare" not in raw.columns:
            return None
        from stockpool.factors.fundamentals import _pit_align
        raw = raw.copy()
        raw["totalShare"] = pd.to_numeric(raw["totalShare"], errors="coerce")
        return _pit_align(raw, "totalShare", close_panel, table="profit")
    except Exception as e:  # noqa: BLE001
        log.warning("PIT shares panel build failed (%s); falling back to snapshot", e)
        return None


def build_log_mcap_panel(
    pool_data: Mapping[str, pd.DataFrame],
    cache_dir: str | Path | None,
) -> pd.DataFrame | None:
    """Build a T×N ``log(total_market_cap)`` panel for market-cap neutralize.

    ``market_cap_t = close_t × totalShare_t``(P2-22):股本优先用 profit 表
    **逐季 PIT** 序列(按公告日 ffill,增发/回购在公告日生效,无前视);
    PIT 覆盖不到的格子(早于基本面覆盖窗口 / 缺数据的票)回退
    ``data/mcap_shares.parquet`` 的最新快照静态广播(此时是文档化近似)。
    ``log`` is applied so the neutralize OLS regresses on a roughly-normal
    size variable.

    Remaining documented approximation: cached ``close`` 为 hfq 复权价,
    绝对 mcap 被各股复权因子缩放,截面 size 排序是近似(修复需 raw close
    或复权因子列,见 review P2-22 长期项)。

    Returns ``None`` when both PIT profit data and the snapshot are missing
    (caller then skips the market_cap_neutralize step with a warning).
    """
    if not pool_data or cache_dir is None:
        return None

    import numpy as np

    close_panel = build_close_panel(pool_data)
    if close_panel.empty:
        return None

    # 1) PIT 逐季股本(优先)
    shares_panel = _load_pit_shares_panel(close_panel, cache_dir)

    # 2) 静态快照(回退 / 填洞)
    shares: pd.Series | None = None
    shares_path = Path(cache_dir) / "mcap_shares.parquet"
    if shares_path.exists():
        snap = pd.read_parquet(shares_path)
        if not snap.empty and "totalShare" in snap.columns:
            shares = (
                snap.assign(code=snap["code"].astype(str).str.zfill(6))
                .dropna(subset=["totalShare"])
                .drop_duplicates("code", keep="last")
                .set_index("code")["totalShare"]
            )

    if shares_panel is None and shares is None:
        log.warning(
            "build_log_mcap_panel: 无 PIT profit 数据也无 %s 快照;"
            "market_cap_neutralize will skip", shares_path,
        )
        return None

    if shares_panel is not None and shares is not None:
        static_panel = pd.DataFrame(
            np.broadcast_to(
                shares.reindex(close_panel.columns).to_numpy(dtype=float),
                close_panel.shape,
            ),
            index=close_panel.index, columns=close_panel.columns,
        )
        shares_panel = shares_panel.fillna(static_panel)
    elif shares_panel is None:
        shares_panel = pd.DataFrame(
            np.broadcast_to(
                shares.reindex(close_panel.columns).to_numpy(dtype=float),
                close_panel.shape,
            ),
            index=close_panel.index, columns=close_panel.columns,
        )

    mcap = close_panel * shares_panel
    mcap = mcap.where(mcap > 0)  # guard non-positive / NaN before log
    log_mcap = np.log(mcap)
    # Report MEDIAN coverage, not last-bar: the trailing edge is sparse when
    # the universe cache (built on some past date) is older than the freshly
    # fetched application stocks, so only those few stocks have a close on the
    # latest bars. Median reflects the bulk of the window that actually drives
    # neutralization.
    if len(log_mcap):
        cov = log_mcap.notna().sum(axis=1)
        med, last = int(cov.median()), int(cov.iloc[-1])
    else:
        med = last = 0
    log.info(
        "log_mcap panel built: %d×%d, median coverage %d/%d codes (last-bar %d)",
        log_mcap.shape[0], log_mcap.shape[1], med, log_mcap.shape[1], last,
    )
    return log_mcap


def maybe_inject_mcap_panel(
    preprocess_cfg, pool_data: Mapping[str, pd.DataFrame], cache_dir: str | Path | None,
) -> None:
    """Build + inject the log-mcap panel into factor context iff needed.

    No-op unless ``preprocess_cfg.market_cap_neutralize`` is True. Mirrors the
    ``set_sector_map`` injection done by the pool-prep entry points so
    ``build_factor_panel`` can pick the panel up from context.
    """
    from stockpool.factors.context import set_mcap_panel
    if preprocess_cfg is None or not getattr(preprocess_cfg, "market_cap_neutralize", False):
        return
    set_mcap_panel(build_log_mcap_panel(pool_data, cache_dir))


def _fundamentals_latest_mtime(cache_dir: str | Path | None) -> str | None:
    """Return the newest mtime among ``data/fundamentals_*.parquet`` as ISO string.

    Used by :func:`load_or_build_factor_panel` to invalidate the factor panel cache
    when fundamentals (which factor sigs don't track) are refreshed independently.

    Returns ``None`` when ``cache_dir`` is ``None`` or no ``fundamentals_*.parquet``
    files exist (treated as "no fundamentals to worry about" — cache remains valid).
    """
    if cache_dir is None:
        return None
    p = Path(cache_dir)
    if not p.exists():
        return None
    parquets = list(p.glob("fundamentals_*.parquet"))
    if not parquets:
        return None
    latest = max(f.stat().st_mtime for f in parquets)
    import datetime
    return datetime.datetime.fromtimestamp(latest).isoformat()


def _factor_panel_sig(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    preprocess_cfg: "PreprocessConfig | None" = None,
) -> tuple[str, str]:
    """Return (12-char sig, last_date_iso) identifying a (factor list, universe,
    history range, preprocess config) tuple.

    Universe = sorted code list. last_date = max of any stock's max date.

    ``preprocess_cfg`` is included only when non-None **and** not all-off — an
    all-off cfg omits the ``"preprocess"`` key from the sig dict entirely so the
    hash is byte-identical to the pre-PR baseline (existing
    ``factor_panels/<sig>/`` caches remain valid).

    Mask config is **not** part of the key — factor panels are mask-
    independent (mask only affects labels downstream of factor computation).
    """
    from stockpool.ml.preprocess import _is_all_off

    codes = sorted(pool_data.keys())
    last_date = pd.Timestamp.min
    for df in pool_data.values():
        if len(df) > 0:
            d = pd.to_datetime(df["date"]).max()
            if d > last_date:
                last_date = d
    last_iso = "" if last_date is pd.Timestamp.min else last_date.date().isoformat()
    blob_dict: dict = {
        "factors": sorted(factor_names),
        "codes": codes,
        "last_date": last_iso,
    }
    if preprocess_cfg is not None and not _is_all_off(preprocess_cfg):
        blob_dict["preprocess"] = preprocess_cfg.model_dump()
    blob = json.dumps(blob_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12], last_iso


def load_or_build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    cache_dir: str | Path,
    refresh: bool = False,
    preprocess_cfg: "PreprocessConfig | None" = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Disk-cached wrapper around ``build_factor_panel`` + ``build_close_panel``.

    Cache layout:
      ``<cache_dir>/factor_panels/<sig>/manifest.json``
      ``<cache_dir>/factor_panels/<sig>/close.parquet``
      ``<cache_dir>/factor_panels/<sig>/<factor_name>.parquet`` × N

    Cache key (``sig``) hashes (sorted factor names, sorted universe codes,
    last_date). Any change → fresh sig → recompute. There is no incremental
    update: pushing last_date by one bar triggers a full rebuild.

    Pass ``refresh=True`` to bypass the cache and overwrite.

    Returns ``(factor_panel, close_panel)``.
    """
    if not pool_data:
        return {}, pd.DataFrame()

    sig, last_iso = _factor_panel_sig(factor_names, pool_data, preprocess_cfg=preprocess_cfg)
    root = Path(cache_dir) / "factor_panels" / sig
    manifest_path = root / "manifest.json"

    if not refresh and manifest_path.exists():
        try:
            meta = json.loads(manifest_path.read_text(encoding="utf-8"))
            close_path = root / "close.parquet"
            paths = {n: root / f"{n}.parquet" for n in meta.get("factors", [])}
            if close_path.exists() and all(p.exists() for p in paths.values()):
                # Fundamentals (baostock quarterly) live outside the factor sig
                # — if they were refreshed since the cache was built, the
                # cached factor values are stale → force a rebuild.
                # A None→non-None transition (fundamentals appeared since
                # build) also counts as stale.
                cached_fund_date = meta.get("fundamentals_snapshot_date")
                current_fund_date = _fundamentals_latest_mtime(cache_dir)
                stale = False
                if current_fund_date is not None:
                    if cached_fund_date is None or current_fund_date > cached_fund_date:
                        stale = True
                if stale:
                    log.info(
                        "Factor panel cache stale: fundamentals refreshed since build "
                        "(cached=%s, current=%s); rebuilding",
                        cached_fund_date, current_fund_date,
                    )
                    # fall through to rebuild
                else:
                    log.info("Factor panel cache hit: %s (sig=%s)", root, sig)
                    close_panel = pd.read_parquet(close_path)
                    factor_panel = {n: pd.read_parquet(p) for n, p in paths.items()}
                    return factor_panel, close_panel
            else:
                log.warning("Factor panel manifest exists but parquets incomplete; rebuilding")
        except Exception as e:
            log.warning("Factor panel cache read failed (%s); rebuilding", e)

    log.info("Building factor panel: %d factors × %d stocks (sig=%s)",
             len(factor_names), len(pool_data), sig)
    factor_panel = build_factor_panel(factor_names, pool_data, preprocess_cfg=preprocess_cfg)
    close_panel = build_close_panel(pool_data)

    root.mkdir(parents=True, exist_ok=True)
    try:
        close_panel.to_parquet(root / "close.parquet")
        for name, wide in factor_panel.items():
            wide.to_parquet(root / f"{name}.parquet")
        manifest_dict: dict = {
            "sig": sig,
            "factors": list(factor_panel.keys()),
            "n_codes": len(pool_data),
            "last_date": last_iso,
            "built_at": pd.Timestamp.now("UTC").isoformat(),
            "fundamentals_snapshot_date": _fundamentals_latest_mtime(cache_dir),
        }
        manifest_path.write_text(
            json.dumps(manifest_dict, indent=2),
            encoding="utf-8",
        )
        log.info("Factor panel cached: %s", root)
    except Exception as e:
        log.warning("Failed to write factor panel cache (%s); proceeding in-memory", e)

    return factor_panel, close_panel


def simulate_strategy_equity_curve(
    daily_df: pd.DataFrame,
    strategy: Strategy,
    holding_days_list: list[int],
    with_buy_and_hold: bool = True,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    risk_free_rate: float = 0.02,
    engine: str = "single",
    position_size: float | None = None,
    lot_sizer: LotSizer | None = None,
    max_concurrent_lots: int | None = None,
) -> EquityResult:
    """Generic equity-curve simulator: runs ``strategy`` for each holding-day cap.

    Output shape matches ``backtest_composite.simulate_equity_curve`` so the
    HTML renderer accepts both.

    Args:
        engine: ``"single"`` or ``"multi_lot"``.
        lot_sizer: only used when ``engine="multi_lot"`` — a ``LotSizer``
                   callable (e.g. ``FixedLotSizer(0.1)`` or ``VolTargetLotSizer(...)``)
                   that determines lot size per buy. Preferred over the
                   deprecated ``position_size``.
        position_size: deprecated alias of ``lot_sizer=FixedLotSizer(position_size)``;
                       kept for backwards compat (existing tests pass it as a
                       kwarg). Mutually exclusive with ``lot_sizer`` — passing
                       both raises ValueError. If both are None, defaults to
                       ``FixedLotSizer(0.1)``.
        max_concurrent_lots: only used when ``engine="multi_lot"`` — cap on
                             simultaneous open lots; None = uncapped by count
                             (cash still self-caps).
    """
    costs = TradeCosts(buy_cost=buy_cost, sell_cost=sell_cost)
    if engine == "single":
        bt = BacktestEngine(strategy, costs=costs, risk_free_rate=risk_free_rate)
    elif engine == "multi_lot":
        if lot_sizer is None:
            size = position_size if position_size is not None else 0.1
            lot_sizer = FixedLotSizer(size)
        elif position_size is not None:
            raise ValueError(
                "Pass either lot_sizer or position_size, not both"
            )
        bt = MultiLotBacktestEngine(
            strategy, lot_sizer=lot_sizer, costs=costs,
            risk_free_rate=risk_free_rate, max_concurrent_lots=max_concurrent_lots,
        )
    else:
        raise ValueError(f"engine must be 'single' or 'multi_lot', got {engine!r}")

    # Generate signals once; reuse across holding-day sweeps.
    signals = strategy.generate_signals(daily_df)
    curves: dict[int, pd.DataFrame] = {}
    metrics: dict[int, dict] = {}
    metrics_active: dict[int, dict | None] = {}
    for N in holding_days_list:
        result = bt.run_on_signals(signals, max_holding_days=N)
        curves[N] = result.curve
        metrics[N] = result.metrics
        metrics_active[N] = result.metrics_active

    bh_curve = None
    bh_metrics = None
    if with_buy_and_hold and len(daily_df) > 0:
        bh = buy_and_hold_baseline(daily_df, risk_free_rate=risk_free_rate)
        bh_curve = bh.curve[["date", "equity"]].reset_index(drop=True)
        bh_metrics = bh.metrics

    return EquityResult(
        curves=curves,
        metrics=metrics,
        buy_and_hold=bh_curve,
        buy_and_hold_metrics=bh_metrics,
        metrics_active=metrics_active,
    )
