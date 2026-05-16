"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_daily() -> pd.DataFrame:
    """30 trading days of synthetic OHLCV — deterministic, computable by hand."""
    dates = pd.date_range("2026-01-02", periods=30, freq="B")
    close = np.array(
        [10.0, 10.2, 10.5, 10.3, 10.6, 10.8, 11.0, 10.9, 11.2, 11.5,
         11.4, 11.6, 11.8, 12.0, 11.9, 12.1, 12.3, 12.5, 12.4, 12.6,
         12.8, 13.0, 12.9, 13.1, 13.3, 13.5, 13.4, 13.6, 13.8, 14.0]
    )
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": np.full(30, 1_000_000, dtype=float),
    })
