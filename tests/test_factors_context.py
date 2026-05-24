"""Tests for shared factor context (sector_map injection)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _reset_sector_map():
    """Isolate ClassVar between tests to avoid pollution."""
    from stockpool.factors.context import set_sector_map
    set_sector_map({})
    yield
    set_sector_map({})


def test_set_get_sector_map_roundtrip():
    from stockpool.factors.context import set_sector_map, get_sector_map
    set_sector_map({"600000": "银行", "000001": "银行"})
    assert get_sector_map() == {"600000": "银行", "000001": "银行"}


def test_get_sector_map_returns_copy():
    """Mutation of returned dict must not affect internal state."""
    from stockpool.factors.context import set_sector_map, get_sector_map
    set_sector_map({"600000": "银行"})
    snapshot = get_sector_map()
    snapshot["FAKE"] = "X"
    assert get_sector_map() == {"600000": "银行"}


def test_empty_sector_map_default():
    from stockpool.factors.context import get_sector_map
    assert get_sector_map() == {}


def test_indneutralize_with_context_empty_map():
    """Empty sector_map → cross-sec demean (subtract daily row mean)."""
    from stockpool.factors.context import indneutralize_with_context
    x = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=pd.date_range("2024-01-01", periods=2),
        columns=["A", "B", "C"],
    )
    out = indneutralize_with_context(x)
    expected = pd.DataFrame(
        [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
        index=x.index, columns=x.columns,
    )
    pd.testing.assert_frame_equal(out, expected)


def test_indneutralize_with_context_nonempty():
    """Non-empty sector_map → group demean within each sector."""
    from stockpool.factors.context import set_sector_map, indneutralize_with_context
    set_sector_map({"A": "X", "B": "X", "C": "Y"})
    x = pd.DataFrame(
        [[1.0, 3.0, 10.0]],
        index=pd.date_range("2024-01-01", periods=1),
        columns=["A", "B", "C"],
    )
    out = indneutralize_with_context(x)
    # X-sector mean = (1+3)/2 = 2; Y-sector solo → mean = 10 → demean to 0
    expected = pd.DataFrame(
        [[-1.0, 1.0, 0.0]],
        index=x.index, columns=x.columns,
    )
    pd.testing.assert_frame_equal(out, expected)


def test_wq101_set_sector_map_reexport():
    """Old import path 'from stockpool.factors.wq101 import set_sector_map' still works."""
    from stockpool.factors.wq101 import set_sector_map as wq_set
    from stockpool.factors.context import get_sector_map
    wq_set({"600519": "白酒"})
    assert get_sector_map() == {"600519": "白酒"}
