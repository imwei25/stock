"""Lightweight memory + timing instrumentation for OOM debugging.

Designed for portfolio-backtest / portfolio-ab paths where the
training pool can swell into 10-30 GB and Windows spawn-pickle
to 3 workers can push past 32 GB.

Usage:
    from stockpool._instrumentation import checkpoint, panel_size_mb
    checkpoint("after load_universe", extra={"size": len(pool_data)})
    checkpoint("factor_panel built", extra={"factors": len(fp), "size_mb": panel_size_mb(fp)})

Each checkpoint emits one INFO log line:
    [MEM] <label>  RSS=NNNN MB  vmem=NNNN MB  t=+NNNNs  Δ=+NNNNs  worker_id=X  <extra>

psutil is a soft dependency:
    pip install psutil
otherwise RSS/vmem print as nan.

The clock is module-global (per process). In multiprocessing.Pool
workers each child has its own clock starting at worker init.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Mapping

log = logging.getLogger("stockpool")

_T0: float = time.perf_counter()
_LAST_T: float = _T0
_PSUTIL = None
try:
    import psutil as _psutil
    _PSUTIL = _psutil
except ImportError:
    _PSUTIL = None


def reset_clock() -> None:
    """Reset the baseline timestamp (call at the start of a worker)."""
    global _T0, _LAST_T
    _T0 = time.perf_counter()
    _LAST_T = _T0


def _rss_vmem_mb() -> tuple[float, float]:
    """Return (rss_mb, vmem_mb). nan if psutil missing."""
    if _PSUTIL is None:
        return float("nan"), float("nan")
    try:
        info = _PSUTIL.Process(os.getpid()).memory_info()
        return info.rss / (1024 ** 2), info.vms / (1024 ** 2)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def checkpoint(label: str, extra: Mapping[str, Any] | None = None) -> None:
    """Log RSS / vmem + elapsed time since start + delta since last checkpoint."""
    global _LAST_T
    now = time.perf_counter()
    rss, vmem = _rss_vmem_mb()
    extra_str = ""
    if extra:
        extra_str = " " + " ".join(
            f"{k}={_fmt_value(v)}" for k, v in extra.items()
        )
    log.info(
        "[MEM] %-42s  RSS=%6.0fMB  vmem=%6.0fMB  pid=%-6d  t=+%6.1fs  Δt=+%6.1fs%s",
        label, rss, vmem, os.getpid(), now - _T0, now - _LAST_T, extra_str,
    )
    _LAST_T = now


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def panel_size_mb(panel: Mapping[str, Any]) -> float:
    """Estimate {name -> DataFrame} dict memory in MB (sum of df.memory_usage).

    Doesn't account for index sharing or numpy overhead — good enough for
    OOM-scale diagnostics. Returns 0 on empty / non-dict input.
    """
    if not panel:
        return 0.0
    total_bytes = 0
    for df in panel.values():
        try:
            if hasattr(df, "memory_usage"):
                # deep=False: ignores object-dtype string heap; fine for floats
                total_bytes += int(df.memory_usage(deep=False).sum())
            elif hasattr(df, "values"):
                total_bytes += df.values.nbytes
        except Exception:  # noqa: BLE001 — diagnostic helper, never raise
            pass
    return total_bytes / (1024 ** 2)


def pool_data_size_mb(pool_data: Mapping[str, Any]) -> float:
    """Estimate {code -> daily_df} dict memory in MB. Same as panel_size_mb."""
    return panel_size_mb(pool_data)
