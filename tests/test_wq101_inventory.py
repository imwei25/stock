"""Tests for the WQ101 window inventory script.

The script does a static AST scan of `src/stockpool/factors/wq101.py` and
emits a CSV of (alpha_id, op, window, count_in_alpha, category, transformable)
rows for every integer-literal window arg passed to a window-bearing op.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def test_inventory_extracts_alpha002_windows(tmp_path):
    """alpha_002 has correlation(., ., 6) and delta(., 2) → inventory windows [2, 6]."""
    out_csv = tmp_path / "inv.csv"
    rc = subprocess.run(
        [sys.executable, "scripts/wq101_window_inventory.py",
         "--output", str(out_csv)],
        cwd=Path.cwd(),
    ).returncode
    assert rc == 0
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    a002 = [r for r in rows if r["alpha_id"] == "alpha_002"]
    # alpha_002: ops.delta(log_v, 2) + ops.correlation(a, b, 6)
    windows = sorted(int(r["window"]) for r in a002)
    assert windows == [2, 6], f"alpha_002 windows should be [2,6], got {windows}"
    ops_seen = {r["op"] for r in a002}
    assert "correlation" in ops_seen
    assert "delta" in ops_seen


def test_inventory_classifies_categories(tmp_path):
    out_csv = tmp_path / "inv.csv"
    subprocess.run(
        [sys.executable, "scripts/wq101_window_inventory.py",
         "--output", str(out_csv)],
        cwd=Path.cwd(), check=True,
    )
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    for r in rows:
        if not r["window"]:  # skip transformable=False rows with empty window
            continue
        w = int(r["window"])
        cat = r["category"]
        if w <= 10:
            assert cat == "short", f"w={w} should be short"
        elif w <= 30:
            assert cat == "medium", f"w={w} should be medium"
        elif w >= 60:
            assert cat == "long", f"w={w} should be long"
        else:
            assert cat == "other", f"w={w} should be other"


def test_inventory_handles_alphas_without_window_literals(tmp_path):
    """Alphas with no whitelisted window-bearing op (e.g. alpha_056 placeholder
    that returns _nan_like(panel) directly) should still appear as a single
    'no literals' row with empty op/window and transformable=False.
    """
    out_csv = tmp_path / "inv.csv"
    subprocess.run(
        [sys.executable, "scripts/wq101_window_inventory.py",
         "--output", str(out_csv)],
        cwd=Path.cwd(), check=True,
    )
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    a056 = [r for r in rows if r["alpha_id"] == "alpha_056"]
    assert len(a056) == 1, f"alpha_056 should have exactly 1 placeholder row, got {len(a056)}"
    r = a056[0]
    assert r["op"] == ""
    assert r["window"] == ""
    assert r["category"] == ""
    assert r["transformable"] == "False"
