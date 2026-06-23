def test_picker_passes_dual_half_threshold(tmp_path):
    """Variant passing both halves is selected over baseline."""
    import json, subprocess, sys
    from pathlib import Path

    # Synthetic baseline + variants. alpha_002_compress passes both halves.
    def _mkjson(d): return json.dumps(d, ensure_ascii=False)

    h1 = tmp_path / "h1.json"; h2 = tmp_path / "h2.json"
    base_template = {
        "factor_names": ["alpha_002", "alpha_002_compress",
                          "alpha_002_rev_short", "alpha_002_expand_long"],
        "abs_ic_mean": {"alpha_002": 0.08, "alpha_002_compress": 0.12,
                        "alpha_002_rev_short": 0.09,
                        "alpha_002_expand_long": 0.06},
        "ic_ir": {"alpha_002": 0.10, "alpha_002_compress": 0.30,
                  "alpha_002_rev_short": 0.15,
                  "alpha_002_expand_long": 0.05},
        "degenerate_day_ratio": {"alpha_002": 0.0, "alpha_002_compress": 0.0,
                                  "alpha_002_rev_short": 0.05,
                                  "alpha_002_expand_long": 0.0},
    }
    h1.write_text(_mkjson(base_template))
    h2.write_text(_mkjson(base_template))

    cur = tmp_path / "selection.json"
    cur.write_text(_mkjson({"factors": ["alpha_002", "momentum_20"]}))

    winners_csv = tmp_path / "winners.csv"
    new_sel = tmp_path / "new_sel.json"

    subprocess.run(
        [sys.executable, "scripts/pick_wq101_winners.py",
         "--h1", str(h1), "--h2", str(h2),
         "--current-selection", str(cur),
         "--winners-csv", str(winners_csv),
         "--output-selection", str(new_sel),
         "--baseline-top-n", "1"],
        check=True,
    )
    import csv
    rows = list(csv.DictReader(open(winners_csv, encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["chosen_variant"] == "alpha_002_compress"
    out = json.loads(new_sel.read_text(encoding="utf-8"))
    # baseline alpha_002 replaced by alpha_002_compress
    assert "alpha_002_compress" in out["factors"]
    assert "alpha_002" not in out["factors"]
    # unrelated factor kept
    assert "momentum_20" in out["factors"]


def test_picker_keeps_baseline_when_no_variant_passes(tmp_path):
    import json, subprocess, sys
    base = {
        "factor_names": ["alpha_002", "alpha_002_compress"],
        "abs_ic_mean": {"alpha_002": 0.08, "alpha_002_compress": 0.085},
        "ic_ir": {"alpha_002": 0.10, "alpha_002_compress": 0.11},
        "degenerate_day_ratio": {"alpha_002": 0.0, "alpha_002_compress": 0.0},
    }
    h1 = tmp_path / "h1.json"; h2 = tmp_path / "h2.json"
    h1.write_text(json.dumps(base)); h2.write_text(json.dumps(base))
    cur = tmp_path / "cur.json"
    cur.write_text(json.dumps({"factors": ["alpha_002"]}))
    winners = tmp_path / "win.csv"
    new = tmp_path / "new.json"
    subprocess.run(
        [sys.executable, "scripts/pick_wq101_winners.py",
         "--h1", str(h1), "--h2", str(h2),
         "--current-selection", str(cur),
         "--winners-csv", str(winners),
         "--output-selection", str(new),
         "--baseline-top-n", "1"],
        check=True,
    )
    import csv
    rows = list(csv.DictReader(open(winners, encoding="utf-8")))
    assert rows == []  # no winners
    out = json.loads(new.read_text(encoding="utf-8"))
    assert out["factors"] == ["alpha_002"]  # baseline preserved
