"""Adapter: per-stock ``Strategy`` → portfolio (T × N) score panel.

For each code, calls ``legacy.generate_signals(daily)`` and extracts the
``score_field`` column (default ``"final_score"`` — emitted by both
``CompositeVerdictStrategy`` and ``MLFactorStrategy``). Walk-forward training
happens inside the legacy strategy, so the resulting panel is look-ahead-safe
by construction.

Failure isolation: any per-stock exception is logged at WARNING and the code
is skipped — the panel still builds for the survivors.

Parallel execution: ``n_workers > 1`` runs stocks across a
``multiprocessing.Pool`` with the legacy strategy pickled once per worker via
``Pool(initializer=...)``. Each worker thus has its OWN ``_shared_cache``
(the legacy strategy's cross-stock Lasso-fit cache) — i.e. monthly Lasso
fits are duplicated ``n_workers`` times instead of computed once. For ``237``
stocks × ``share_pool_fit=True`` × ``~12`` monthly refits, the trade-off is
``12`` fits serial vs ``~n_workers × 12`` parallel — typically still net-faster
because the per-bar Python loop (which dominates serial time) parallelises
linearly. On Windows the spawn-time pickle of the strategy (containing
factor_panel + pool_data) can be hundreds of MB per worker; budget memory
accordingly.
"""
from __future__ import annotations

import logging
import os
from typing import Mapping

import pandas as pd

log = logging.getLogger("stockpool")


# Module-global populated by ``_worker_init`` inside each Pool worker so the
# legacy strategy is pickled once at worker startup, not per-task. Worker
# tasks reach the strategy via this global — module state is the canonical
# way to share heavy objects across imap tasks in ``multiprocessing.Pool``.
_WORKER_STRATEGY = None


def _worker_init(strategy, sector_map):
    """Pool initializer: stash the legacy strategy in worker module-global
    AND restore the parent's sector_map (worker processes don't inherit
    Python module-level globals across spawn — without this, sector-aware
    factors like IndustryRelativeStrength raise ``RuntimeError: sector_map
    is empty``)."""
    global _WORKER_STRATEGY
    _WORKER_STRATEGY = strategy
    from stockpool.factors.context import set_sector_map
    set_sector_map(sector_map or {})
    from stockpool._instrumentation import checkpoint, reset_clock
    reset_clock()
    checkpoint("worker_init: pickle inflated + sector_map restored")


# Module-global, per-worker counter so we can sample a checkpoint on the
# first task each worker handles (worker memory after first inflate is the
# OOM-likely peak).
_WORKER_STOCKS_SEEN: int = 0


def _score_one_stock(args):
    """Worker task: returns ``(code, series_or_None, err_msg_or_None)``."""
    global _WORKER_STOCKS_SEEN
    code, daily, score_field = args
    _WORKER_STOCKS_SEEN += 1
    if _WORKER_STOCKS_SEEN == 1:
        from stockpool._instrumentation import checkpoint
        checkpoint(f"worker first stock start ({code})")
    try:
        sig = _WORKER_STRATEGY.generate_signals(daily)  # type: ignore[union-attr]
    except Exception as e:  # noqa: BLE001 — failure-isolation contract
        return (code, None, f"generate_signals failed: {e}")
    if _WORKER_STOCKS_SEEN == 1:
        from stockpool._instrumentation import checkpoint
        checkpoint(
            f"worker first stock done ({code}) "
            f"— peak after _ensure_pooled_xy_long + first generate_signals"
        )
    if score_field not in sig.columns:
        return (code, None, f"missing {score_field!r} in generate_signals output")
    if "date" not in sig.columns:
        return (code, None, "missing 'date' column")
    s = sig.set_index("date")[score_field]
    s = s[~s.index.duplicated(keep="last")]
    return (code, s, None)


def _serial_loop(legacy_strategy, tasks, score_field, code_iter):
    """Original in-process implementation; preserved for n_workers <= 1."""
    series_by_code: dict[str, pd.Series] = {}
    for code, daily in code_iter:
        try:
            sig = legacy_strategy.generate_signals(daily)
        except Exception as e:  # noqa: BLE001 — failure-isolation contract
            log.warning("score panel: %s generate_signals failed (%s); skip", code, e)
            continue
        if score_field not in sig.columns:
            log.warning(
                "score panel: %s missing %r in generate_signals output; skip",
                code, score_field,
            )
            continue
        if "date" not in sig.columns:
            log.warning("score panel: %s missing 'date' column; skip", code)
            continue
        s = sig.set_index("date")[score_field]
        s = s[~s.index.duplicated(keep="last")]
        series_by_code[code] = s
    return series_by_code


def _prewarm_monthly_fits(legacy_strategy, panel_data) -> int:
    """Pre-compute monthly ML fits in the main process so subprocess workers
    inherit them via the pickled ``_shared_cache``.

    Only applies when: (a) strategy is ``MLFactorStrategy``, (b) it's in
    pooled + share_pool_fit mode, (c) the shared_cache is a real dict.
    Otherwise no-op.

    Calls ``generate_signals`` on the first stock in iteration order (the same
    host that serial mode iterates first via ``panel_data.items()``).  This
    ensures the cache is populated with **exactly** the same ``(sig, year,
    month)`` → ``(pipeline, quantiles)`` entries as the serial loop would
    build — including month-boundary triggered refits — so prewarm + parallel
    produces a bit-exact score panel compared to no-prewarm + serial mode.

    The previous implementation walked at ``refit_every`` cadence via
    ``_try_fit`` directly, which missed month-boundary refits and produced
    different pipeline fits than ``generate_signals``.

    After the walk, drops the ``__pooled_xy_long__`` key from shared_cache
    so it doesn't bloat the pickle blob sent to workers (workers will
    rebuild that panel locally — same as today's baseline).

    Returns the number of fits placed into the cache (for log/diagnostic).
    """
    # Lazy imports to avoid circular-import surface for non-ML callers.
    from stockpool.backtesting.strategies import MLFactorStrategy

    if not isinstance(legacy_strategy, MLFactorStrategy):
        return 0
    if not legacy_strategy._is_sharing():
        return 0
    if legacy_strategy._shared_cache is None:
        return 0
    if not panel_data:
        return 0

    # Use dict iteration order (first key) — this matches what the serial loop
    # does when iterating panel_data.items(), so prewarm + parallel produces
    # bit-exact the same score panel as no-prewarm + serial mode.
    warmup_code = next(iter(panel_data))
    warmup_strat = legacy_strategy.with_stock(warmup_code)
    daily = panel_data[warmup_code]

    pre_count = len(legacy_strategy._shared_cache)

    # Call generate_signals rather than walking _try_fit directly.  This
    # replicates the EXACT same trigger logic (refit_every cadence + month-
    # boundary `need_month_fit` checks) that the serial scoring loop runs for
    # the first stock, so the ``(sig, year, month)`` cache entries are
    # populated with the same pipelines.  Using _try_fit at range(0,n,refit_every)
    # misses month-boundary refits and produces different training cutoffs.
    warmup_strat.generate_signals(daily)

    # Drop the heavy long-form pooled panel from the cache before pickle.
    # Key shape is ``("__pooled_xy_long__", sig)``. Workers will rebuild
    # locally (same as the pre-prewarm baseline).
    heavy_keys = [
        k for k in list(legacy_strategy._shared_cache.keys())
        if isinstance(k, tuple) and len(k) == 2
        and k[0] == "__pooled_xy_long__"
    ]
    for k in heavy_keys:
        del legacy_strategy._shared_cache[k]

    n_new = len(legacy_strategy._shared_cache) - pre_count + len(heavy_keys)
    log.info(
        "precompute_scores: pre-warmed %d monthly fits from host=%s "
        "(via generate_signals); dropped %d heavy keys before worker spawn",
        n_new, warmup_code, len(heavy_keys),
    )
    return n_new


def precompute_scores_from_legacy(
    legacy_strategy,
    panel_data: Mapping[str, pd.DataFrame],
    score_field: str = "final_score",
    n_workers: int | None = None,
    prewarm: bool = True,
) -> pd.DataFrame:
    """Build a (T × N) score panel by calling ``legacy.generate_signals`` per stock.

    Args:
        legacy_strategy: a per-stock ``Strategy`` whose ``generate_signals``
            output frame contains ``date`` and ``score_field`` columns.
        panel_data: ``{code: daily_df}`` — typically loaded from cache.
        score_field: column to extract (default ``"final_score"``).
        n_workers: number of parallel workers. ``None`` (default) = auto =
            ``min(8, max(1, cpu_count() - 1))``; ``1`` = serial (preserves
            original behavior); higher = ``multiprocessing.Pool`` parallelism.
        prewarm: When True (default) and ``n_workers > 1``, pre-compute the
            monthly ML fits in the main process so workers inherit them via
            the pickled ``_shared_cache``. Set False to disable (verifies
            equivalence in tests).

    Returns:
        ``pd.DataFrame`` indexed by date, columns = codes, values = score.
        Codes whose ``generate_signals`` raises or omits ``score_field`` are
        skipped. If *all* codes fail, returns an empty frame.
    """
    from stockpool._instrumentation import checkpoint, pool_data_size_mb
    checkpoint("precompute_scores_from_legacy: entry")
    if n_workers is None:
        # Conservative default: each Pool worker on Windows (spawn) gets a
        # fresh deep-copy of legacy_strategy via pickle — that includes the
        # full pool_data + factor_panel + close_panel dicts (~hundreds of
        # MB per worker for a 4358-stock training pool). Each worker also
        # independently rebuilds `_ensure_pooled_xy_long` (~5 GB long-form
        # DataFrame in pooled mode), so total memory ≈ n_workers × ~6 GB.
        # Default to 3 keeps us under ~20 GB worker memory on a 32 GB box;
        # users can raise via the --workers CLI flag if they have headroom.
        n_workers = max(1, min(3, (os.cpu_count() or 1) - 1))

    # Short-circuit: tiny workloads where Pool spawn overhead would dwarf
    # any parallelism gain. Threshold 20 covers CLI smoke tests
    # (typically 3-8 synthetic stocks) and avoids spawning 3 worker
    # processes each pickling the full strategy just to score 4 stocks.
    if 0 < len(panel_data) < 20 and n_workers > 1:
        log.info(
            "precompute_scores: tiny workload (%d stocks < 20) — forcing serial",
            len(panel_data),
        )
        n_workers = 1

    # NEW: prewarm before tasks/Pool setup so the populated cache is what
    # gets pickled into each worker.
    if n_workers > 1 and prewarm:
        checkpoint("before _prewarm_monthly_fits")
        _prewarm_monthly_fits(legacy_strategy, panel_data)
        checkpoint("after _prewarm_monthly_fits", extra={
            "shared_cache_size": len(legacy_strategy._shared_cache or {}),
        })

    tasks = [(code, daily, score_field) for code, daily in panel_data.items()]
    checkpoint("tasks built", extra={
        "n_tasks": len(tasks),
        "pool_data_mb": pool_data_size_mb(panel_data),
    })

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    if n_workers <= 1:
        log.info(
            "precompute_scores: serial mode (n_workers=1) over %d stocks",
            len(tasks),
        )
        code_iter = tqdm(
            panel_data.items(), total=len(panel_data),
            desc="precompute_scores", unit="stock", mininterval=2.0,
        ) if tqdm is not None else panel_data.items()
        series_by_code = _serial_loop(legacy_strategy, tasks, score_field, code_iter)
    else:
        from multiprocessing import Pool
        log.info(
            "precompute_scores: parallel mode (n_workers=%d) over %d stocks",
            n_workers, len(tasks),
        )
        series_by_code = {}
        progress = (
            tqdm(total=len(tasks), desc="precompute_scores", unit="stock", mininterval=2.0)
            if tqdm is not None else None
        )
        # Snapshot the parent's sector_map so workers can restore it on
        # init (see _worker_init for why this is required on spawn).
        from stockpool.factors.context import get_sector_map
        parent_sector_map = get_sector_map()
        checkpoint("before Pool() spawn — parent peak before pickle")
        with Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(legacy_strategy, parent_sector_map),
        ) as pool:
            checkpoint("Pool() spawned — workers initialised")
            for code, series, err in pool.imap_unordered(_score_one_stock, tasks):
                if err is not None:
                    log.warning("score panel: %s %s; skip", code, err)
                else:
                    series_by_code[code] = series
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()
        checkpoint("Pool() exited", extra={"n_done": len(series_by_code)})

    if not series_by_code:
        return pd.DataFrame()
    panel = pd.DataFrame(series_by_code)
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()
