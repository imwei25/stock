"""Pool B — 全市场量化推荐池.

设计见 ``docs`` 与 ``CLAUDE.md`` 中的 Recommend Pool 段。核心契约:

* 应用池 (``cfg.stocks``) 保持原状, Pool B 独立, 允许重叠;
* 对全市场调用当前 ``cfg.strategy`` 的 ``predict_latest`` 打分;
* 漏斗: 流动性 (近 20 日均成交额) → ST 名称剔除 → 行业上限 → top-N;
* 周缓存: 跨过 ISO 周边界才重算 (``refresh="weekly"``, 默认), 缓存键含
  ``cfg.content_hash`` — 换策略 (改 yaml 任意字段) 自动失效;
* Pool B 不做回测 (MVP) — 周频再平衡的组合曲线留作 follow-up。
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd

from stockpool.config import AppConfig
from stockpool.fetcher import list_universe, load_universe_cache
from stockpool.industry_map import _UNKNOWN, industry_of, load_or_build_industry_map
from stockpool.strategy_factory import build_strategy

log = logging.getLogger(__name__)


@dataclass
class PoolBEntry:
    rank: int
    code: str
    name: str
    industry: str
    final_score: float
    verdict: str
    is_in_pool_a: bool


def compute_or_load_pool_b(
    cfg: AppConfig,
    run_date: date,
    pool_data: Mapping[str, pd.DataFrame] | None = None,
    factor_panel: Mapping[str, pd.DataFrame] | None = None,
    close_panel: pd.DataFrame | None = None,
) -> list[PoolBEntry]:
    """Return current week's Pool B top-N, computing + caching if needed.

    ``pool_data`` / ``factor_panel`` / ``close_panel`` are *strategy training
    inputs* (built by ``cli._prepare_ml_pool`` for ml_factor +
    training_universe=all). They are passed through to ``build_strategy`` so
    the strategy uses real cross-sec factor values at predict time AND avoids
    rebuilding the close panel (4000+ stock pandas copy/reindex) on every
    per-stock ``build_strategy`` call. They do **not** define the iteration
    universe — Pool B always iterates the full ``load_universe_cache`` output.
    """
    cfg_pool = cfg.recommend_pool
    cache_path = _cache_path_for(cfg, run_date)

    if cfg_pool.refresh != "always" and cache_path.exists():
        log.info("Pool B cache hit: %s", cache_path)
        return _read_cache(cache_path)

    if cfg_pool.refresh == "never":
        log.warning("Pool B refresh=never and no cache at %s — returning empty",
                    cache_path)
        return []

    entries = _compute_pool_b(cfg, pool_data, factor_panel, close_panel,
                              run_date=run_date)
    _write_cache(cache_path, entries)
    log.info("Pool B written: %s (%d entries)", cache_path, len(entries))
    return entries


def _cache_path_for(cfg: AppConfig, run_date: date) -> Path:
    iso = run_date.isocalendar()
    fname = f"poolb_{cfg.content_hash}_{iso.year}w{iso.week:02d}.parquet"
    return Path(cfg.recommend_pool.cache_dir) / fname


def _prev_pool_codes(cfg: AppConfig, run_date: date | None) -> set[str]:
    """上一 ISO 周的池内代码(P2-28 粘性用)。缓存缺失/读失败返回空集。"""
    if run_date is None:
        return set()
    try:
        prev = _cache_path_for(cfg, run_date - pd.Timedelta(days=7).to_pytimedelta())
        if prev.exists():
            return {str(r.code).zfill(6)
                    for r in pd.read_parquet(prev).itertuples(index=False)}
    except Exception as e:  # noqa: BLE001
        log.debug("Pool B: prev-week cache unreadable (%s)", e)
    return set()


def _compute_pool_b(
    cfg: AppConfig,
    pool_data: Mapping[str, pd.DataFrame] | None,
    factor_panel: Mapping[str, pd.DataFrame] | None,
    close_panel: pd.DataFrame | None = None,
    run_date: date | None = None,
) -> list[PoolBEntry]:
    cfg_pool = cfg.recommend_pool

    universe_data = load_universe_cache(cfg.data.cache_dir, cfg.data.history_days)
    if not universe_data:
        log.warning(
            "Pool B: universe cache empty at %s — run "
            "`python -m stockpool fetch-universe` first",
            cfg.data.cache_dir,
        )
        return []

    name_map = _build_name_map(cfg.data.cache_dir)
    industry_map = load_or_build_industry_map(
        cfg.data.cache_dir,
        max_age_days=cfg_pool.industry_map_max_age_days,
        source=cfg_pool.industry_source,
    )
    # Make the same map available to factors that consume sector context
    # (industry_relative_strength_N + WQ101 indneutralize).
    from stockpool.factors.context import set_sector_map
    set_sector_map(industry_map)

    try:
        from stockpool.ipo_dates import load_st_codes
        st_codes = load_st_codes(cfg.data.cache_dir)
    except Exception as e:  # noqa: BLE001
        log.warning("Pool B: load_st_codes failed (%s); ST 过滤退化为名称匹配", e)
        st_codes = set()
    survivors = _apply_funnel(
        universe_data, name_map,
        min_avg_amount_20d=cfg_pool.min_avg_amount_20d,
        st_codes=st_codes,
    )
    log.info("Pool B funnel: %d → %d after liquidity + ST filter",
             len(universe_data), len(survivors))
    if not survivors:
        return []

    scored = _score_universe(
        cfg, survivors, name_map, industry_map,
        pool_data=pool_data, factor_panel=factor_panel,
        close_panel=close_panel,
    )

    pool_a_codes = {s.code for s in cfg.stocks}
    picked = _industry_cap_top_n(
        scored,
        top_n=cfg_pool.top_n,
        max_per_industry=cfg_pool.max_per_industry,
        sticky_codes=_prev_pool_codes(cfg, run_date),
    )
    return [
        PoolBEntry(
            rank=i + 1, code=row["code"], name=row["name"],
            industry=row["industry"], final_score=row["final_score"],
            verdict=row["verdict"], is_in_pool_a=row["code"] in pool_a_codes,
        )
        for i, row in enumerate(picked)
    ]


def _build_name_map(cache_dir: str | Path) -> dict[str, str]:
    """``{code: 干净中文名}``,优先 stock_basics(baostock),回退 mootdx 乱码名。"""
    try:
        from stockpool.ipo_dates import load_stock_basics_cached_only
        basics = load_stock_basics_cached_only(cache_dir)
        if not basics.empty:
            return {str(r.code).zfill(6): str(r.name)
                    for r in basics.itertuples(index=False)}
    except Exception as e:  # noqa: BLE001
        log.warning("Pool B: stock_basics unavailable (%s), falling back to mootdx", e)
    try:
        df = list_universe()
        return {str(r.code).zfill(6): str(r.name)
                for r in df.itertuples(index=False)}
    except Exception as e:
        log.warning("Pool B: failed to load universe name map (%s), "
                    "names will fall back to code", e)
        return {}


def _apply_funnel(
    universe_data: Mapping[str, pd.DataFrame],
    name_map: Mapping[str, str],
    min_avg_amount_20d: float,
    st_codes: "set[str] | None" = None,
) -> dict[str, pd.DataFrame]:
    """Liquidity gate + ST 剔除(当下决策,无前视问题)。

    训练池现在**保留** ST(P0-4 ②);推荐池是"今天买什么"的当下决策,
    在这里按干净名单(stock_basics)剔除当前 ST,名称匹配只作兜底。
    """
    st_codes = st_codes or set()
    out: dict[str, pd.DataFrame] = {}
    for code, daily in universe_data.items():
        if len(daily) < 20:
            continue
        name = name_map.get(code, "")
        if code in st_codes or "ST" in name.upper():
            continue
        tail = daily.tail(20)
        # volume 单位已在数据层统一为"股"(P1-6, 全部数据源一致);amount 单位为元
        avg_amount = float((tail["volume"] * tail["close"]).mean())
        if avg_amount < min_avg_amount_20d:
            continue
        out[code] = daily
    return out


def _score_universe(
    cfg: AppConfig,
    survivors: Mapping[str, pd.DataFrame],
    name_map: Mapping[str, str],
    industry_map: Mapping[str, str],
    pool_data: Mapping[str, pd.DataFrame] | None,
    factor_panel: Mapping[str, pd.DataFrame] | None,
    close_panel: pd.DataFrame | None = None,
) -> list[dict]:
    """For each survivor, call current strategy's predict_latest. Return rows
    sorted by ``final_score`` descending. Per-stock failures are logged and
    skipped — Pool B should never blow up the daily report run."""
    shared_cache: dict = {}
    rows: list[dict] = []
    fail_count = 0
    total = len(survivors)
    t_build = 0.0
    t_predict = 0.0
    t_loop_start = time.perf_counter()
    print(f"[TIME] Pool B scoring start: {total} stocks", flush=True)

    # P2-30: 整个循环只 build 一次 strategy(避免每股重复构造/深拷贝面板),
    # 池化策略用 with_stock(code) 做轻量绑定;不支持 with_stock 的策略
    # (composite)无状态,直接复用同一实例。
    _t = time.perf_counter()
    base_strategy = build_strategy(
        cfg, pool_data=pool_data,
        factor_panel=factor_panel, close_panel=close_panel,
        shared_cache=shared_cache,
    )
    base_build = time.perf_counter() - _t
    can_bind = hasattr(base_strategy, "with_stock")

    for i, (code, daily) in enumerate(survivors.items(), 1):
        try:
            _t = time.perf_counter()
            strategy = base_strategy.with_stock(code) if can_bind else base_strategy
            t_build += time.perf_counter() - _t
            _t = time.perf_counter()
            latest = strategy.predict_latest(daily)
            t_predict += time.perf_counter() - _t
            score = latest.get("final_score", latest.get("score"))
            verdict = latest.get("signal", "neutral")
            if score is None:
                fail_count += 1
                continue
            score_f = float(score)
            if score_f != score_f:  # NaN
                fail_count += 1
                continue
            # P2-28 tiebreak:composite 的 final_score 是粗粒度离散值,
            # 大量并列时旧排序由 glob 文件序决定(top-N 边界实质随机);
            # 用 20 日动量做次级排序。
            try:
                closes = daily["close"].astype(float)
                mom20 = float(closes.iloc[-1] / closes.iloc[-21] - 1) \
                    if len(closes) >= 21 else 0.0
            except Exception:  # noqa: BLE001
                mom20 = 0.0
            rows.append({
                "code": code,
                "name": name_map.get(code, code),
                "industry": industry_of(code, industry_map),
                "final_score": score_f,
                "verdict": str(verdict),
                "mom20": mom20,
            })
        except Exception as e:  # noqa: BLE001
            fail_count += 1
            log.debug("Pool B: predict failed for %s (%s)", code, e)
        if i % 200 == 0:
            elapsed = time.perf_counter() - t_loop_start
            eta = elapsed / i * (total - i)
            print(f"[TIME] Pool B {i}/{total} ok={len(rows)} fail={fail_count} "
                  f"elapsed={elapsed:.1f}s build_avg={t_build/i*1000:.1f}ms "
                  f"predict_avg={t_predict/i*1000:.1f}ms ETA={eta:.0f}s",
                  flush=True)

    total_loop = time.perf_counter() - t_loop_start
    print(f"[TIME] Pool B scoring done: {total_loop:.1f}s total "
          f"(base_build={base_build:.1f}s bind_total={t_build:.1f}s "
          f"predict_total={t_predict:.1f}s "
          f"ok={len(rows)} fail={fail_count})", flush=True)
    log.info("Pool B scoring done: ok=%d fail=%d", len(rows), fail_count)
    rows.sort(key=lambda r: (r["final_score"], r.get("mom20", 0.0)), reverse=True)
    return rows


def _industry_cap_top_n(
    scored: list[dict],
    top_n: int,
    max_per_industry: int,
    sticky_codes: "set[str] | None" = None,
) -> list[dict]:
    """Greedy: walk score-desc list, skip any code whose industry bucket is
    already full, stop once we have ``top_n``. ``"未知"`` counts as a normal
    bucket (so a flood of unmapped stocks can't drown out the top-N).

    P2-28 粘性(hysteresis):``sticky_codes``(上周池内的票)只要本周排名
    仍在 **top 1.5N** 内就优先保留 —— 边界票不再每周进进出出,按池操作的
    换手大幅下降。实现:两遍贪心,先收"粘性且排名 ≤1.5N"的,再按排名补足。

    Degrade gracefully when the industry map is empty (e.g. akshare network
    failure): every stock ends up in the "未知" bucket and the cap would
    truncate the pool to just ``max_per_industry`` items — useless. Detect
    that and skip the cap entirely.
    """
    industries = {r["industry"] for r in scored}
    if industries <= {_UNKNOWN}:
        log.warning(
            "Pool B: industry map empty or all unmapped — skipping industry cap. "
            "Fix network/cache and rerun with recommend_pool.refresh=always."
        )
        return scored[:top_n]

    sticky_codes = sticky_codes or set()
    sticky_window = int(top_n * 1.5)
    bucket: dict[str, int] = {}
    out: list[dict] = []
    taken: set[str] = set()

    def _try_take(row) -> bool:
        ind = row["industry"]
        if bucket.get(ind, 0) >= max_per_industry:
            return False
        out.append(row)
        taken.add(row["code"])
        bucket[ind] = bucket.get(ind, 0) + 1
        return True

    # 第一遍:粘性票(上周在池 + 本周排名 ≤ 1.5N)
    if sticky_codes:
        for rank, row in enumerate(scored[:sticky_window]):
            if len(out) >= top_n:
                break
            if row["code"] in sticky_codes:
                _try_take(row)
    # 第二遍:按排名补足
    for row in scored:
        if len(out) >= top_n:
            break
        if row["code"] in taken:
            continue
        _try_take(row)
    # 输出按分数序展示
    out.sort(key=lambda r: -r["final_score"])
    return out[:top_n]


def _read_cache(path: Path) -> list[PoolBEntry]:
    df = pd.read_parquet(path)
    return [
        PoolBEntry(
            rank=int(r.rank), code=str(r.code), name=str(r.name),
            industry=str(r.industry), final_score=float(r.final_score),
            verdict=str(r.verdict), is_in_pool_a=bool(r.is_in_pool_a),
        )
        for r in df.itertuples(index=False)
    ]


def _write_cache(path: Path, entries: list[PoolBEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(e) for e in entries])
    if df.empty:
        # 仍写一个空 parquet,标记 "本周已计算过,确实为空" — 否则
        # 下次启动会再跑一次代价高昂的全市场打分。
        df = pd.DataFrame(columns=[
            "rank", "code", "name", "industry",
            "final_score", "verdict", "is_in_pool_a",
        ])
    df.to_parquet(path, index=False)
