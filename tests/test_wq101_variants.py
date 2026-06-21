"""Tests for the WQ101 variant generator (Task 5 of A-share localization).

The generator script reads a baseline factor_analysis JSON, picks the top-N
WQ101 alphas, AST-rewrites each one's compute method body with three window
transformation rules (_compress / _rev_short / _expand_long), and emits a
Python module with the generated variant classes.
"""
from __future__ import annotations


def test_generator_rewrites_correlation_window(tmp_path):
    """alpha_002's correlation(., ., 6) -> alpha_002_compress with window 3."""
    import subprocess
    import sys
    import json

    baseline = tmp_path / "baseline.json"
    # synthetic baseline ranking alpha_002 / alpha_003 / alpha_004 as top-3
    baseline.write_text(json.dumps({
        "factor_names": ["alpha_002", "alpha_003", "alpha_004"],
        "abs_ic_mean": {"alpha_002": 0.10, "alpha_003": 0.09, "alpha_004": 0.08},
    }))
    out_py = tmp_path / "wq101_variants_test.py"
    rc = subprocess.run(
        [sys.executable, "scripts/generate_wq101_variants.py",
         "--baseline", str(baseline),
         "--top-n", "3",
         "--output", str(out_py)],
        check=True,
    ).returncode
    assert rc == 0
    src = out_py.read_text(encoding="utf-8")
    # alpha_002_compress should rewrite correlation(., ., 6) -> correlation(., ., 3)
    assert "class Alpha002_compress" in src or "Alpha002Compress" in src
    # Direct literal check: the substring "correlation(a, b, 3)" must appear
    # (whitespace agnostic via normalization)
    norm = " ".join(src.split())
    assert "correlation(a, b, 3)" in norm or "correlation( a, b, 3 )" in norm


def test_generated_module_registers_and_computes(tmp_path):
    """After running generator and importing, alpha_002_compress is registered."""
    import subprocess
    import sys
    import json
    import importlib
    from pathlib import Path

    # Use the real output path so import via canonical module name works.
    out_py = Path("src/stockpool/factors/wq101_variants.py")
    backup = None
    if out_py.exists():
        backup = out_py.read_text(encoding="utf-8")
    try:
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({
            "factor_names": ["alpha_002"],
            "abs_ic_mean": {"alpha_002": 0.10},
        }))
        subprocess.run(
            [sys.executable, "scripts/generate_wq101_variants.py",
             "--baseline", str(baseline), "--top-n", "1",
             "--output", str(out_py)],
            check=True,
        )
        # Drop any cached module + clear the variant entries from registry.
        from stockpool.factors.registry import _REGISTRY
        for k in list(_REGISTRY):
            if k.endswith("_compress") or k.endswith("_rev_short") or k.endswith("_expand_long"):
                del _REGISTRY[k]
        # Re-import
        if "stockpool.factors.wq101_variants" in sys.modules:
            del sys.modules["stockpool.factors.wq101_variants"]
        importlib.import_module("stockpool.factors.wq101_variants")
        assert "alpha_002_compress" in _REGISTRY
    finally:
        if backup is not None:
            out_py.write_text(backup, encoding="utf-8")
        elif out_py.exists():
            out_py.unlink()
