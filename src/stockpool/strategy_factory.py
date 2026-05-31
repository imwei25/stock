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
    """
    from stockpool.ml.dataset import compute_factor_panel

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

    return compute_factor_panel(panel, factor_names)


def _factor_panel_sig(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
) -> tuple[str, str]:
    """Return (12-char sig, last_date_iso) identifying a (factor list, universe,
    history range) tuple.

    Universe = sorted code list. last_date = max of any stock's max date.

    Mask config is **not** part of the key — factor panels are now mask-
    independent (mask only affects labels downstream of factor computation).
    """
    codes = sorted(pool_data.keys())
    last_date = pd.Timestamp.min
    for df in pool_data.values():
        if len(df) > 0:
            d = pd.to_datetime(df["date"]).max()
            if d > last_date:
                last_date = d
    last_iso = "" if last_date is pd.Timestamp.min else last_date.date().isoformat()
    blob = json.dumps(
        {
            "factors": sorted(factor_names),
            "codes": codes,
            "last_date": last_iso,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12], last_iso


def load_or_build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    cache_dir: str | Path,
    refresh: bool = False,
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

    sig, last_iso = _factor_panel_sig(factor_names, pool_data)
    root = Path(cache_dir) / "factor_panels" / sig
    manifest_path = root / "manifest.json"

    if not refresh and manifest_path.exists():
        try:
            meta = json.loads(manifest_path.read_text(encoding="utf-8"))
            close_path = root / "close.parquet"
            paths = {n: root / f"{n}.parquet" for n in meta.get("factors", [])}
            if close_path.exists() and all(p.exists() for p in paths.values()):
                log.info("Factor panel cache hit: %s (sig=%s)", root, sig)
                close_panel = pd.read_parquet(close_path)
                factor_panel = {n: pd.read_parquet(p) for n, p in paths.items()}
                return factor_panel, close_panel
            log.warning("Factor panel manifest exists but parquets incomplete; rebuilding")
        except Exception as e:
            log.warning("Factor panel cache read failed (%s); rebuilding", e)

    log.info("Building factor panel: %d factors × %d stocks (sig=%s)",
             len(factor_names), len(pool_data), sig)
    factor_panel = build_factor_panel(factor_names, pool_data)
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
    for N in holding_days_list:
        result = bt.run_on_signals(signals, max_holding_days=N)
        curves[N] = result.curve
        metrics[N] = result.metrics

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
    )
