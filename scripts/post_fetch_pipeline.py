"""Wait for resume_fundamentals to finish, then:

  1. Verify fundamental factors are no longer mostly-NaN.
  2. Re-run `factors analyze` with the now-populated fundamentals.
  3. Re-pick top factors via `factors pick-by-ic` → reports/selection_post_fundamentals.json.
  4. Diff old vs new selection and report fundamental factors entering.
  5. Write ab_fundamentals.yaml comparing old vs new selection.
  6. Run `python -m stockpool ab` and surface the final HTML path.

Designed to be run in the background. Logs to logs/post_fetch_pipeline.log.
Polls logs/resume_fundamentals.log mtime: once quiet for QUIET_SECONDS,
assume fetch is done (or aborted) and move on.

Idempotent: if any phase output already exists, the phase is skipped.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")

# How long the resume_fundamentals log must be quiet before we treat fetch as done.
QUIET_SECONDS = 600  # 10 min

RESUME_LOG = ROOT / "logs" / "resume_fundamentals.log"
PIPELINE_LOG = ROOT / "logs" / "post_fetch_pipeline.log"

SEL_OLD = ROOT / "reports" / "selection.json"
SEL_NEW = ROOT / "reports" / "selection_post_fundamentals.json"

ANALYSIS_DIR = ROOT / "reports" / "factor_analysis"
AB_YAML = ROOT / "ab_fundamentals.yaml"

PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(PIPELINE_LOG, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("post_fetch_pipeline")


def run(cmd: list[str], desc: str) -> int:
    log.info("RUN: %s", desc)
    log.info("CMD: %s", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=ROOT)
    log.info("rc=%d  elapsed=%.1fs", rc, time.time() - t0)
    return rc


# ----- Phase 0: wait for fetch -----
def wait_for_fetch() -> None:
    if not RESUME_LOG.exists():
        log.warning("No resume log at %s; assuming fetch already done", RESUME_LOG)
        return
    log.info("Polling %s until quiet for %ds", RESUME_LOG, QUIET_SECONDS)
    while True:
        age = time.time() - RESUME_LOG.stat().st_mtime
        if age > QUIET_SECONDS:
            log.info("Resume log quiet for %.0fs > %ds → fetch done", age, QUIET_SECONDS)
            return
        time.sleep(120)


# ----- Phase 1: verify fundamentals coverage -----
FUND_FACTOR_TO_TABLE = {
    "roe": "profit", "roa": "profit",
    "gross_margin": "profit", "net_margin": "profit",
    "pe": "profit", "market_cap": "profit", "log_market_cap": "profit",
    "revenue_yoy": "growth",
    "pb": "balance",
}


def verify_coverage() -> dict[str, int]:
    log.info("=" * 60)
    log.info("Phase 1: verify fundamentals coverage")
    univ = set(
        pd.read_parquet(ROOT / "data" / "universe.parquet")["code"]
        .astype(str).str.zfill(6)
    )
    coverage: dict[str, int] = {}
    for tbl in ("profit", "growth", "balance"):
        p = ROOT / "data" / f"fundamentals_{tbl}.parquet"
        if not p.exists():
            log.warning("[%s] no parquet — skipping", tbl)
            coverage[tbl] = 0
            continue
        df = pd.read_parquet(p)
        have = set(df["code"].astype(str).str.zfill(6).unique())
        coverage[tbl] = len(have)
        log.info(
            "[%s] %d/%d codes covered (%.1f%%)  missing=%d",
            tbl, len(have), len(univ),
            100.0 * len(have) / len(univ),
            len(univ - have),
        )
    return coverage


# ----- Phase 2: factor analyze -----
def run_analyze() -> Path:
    log.info("=" * 60)
    log.info("Phase 2: factors analyze")
    run_date = date.today().isoformat()
    out_json = ANALYSIS_DIR / f"{run_date}.json"
    if out_json.exists():
        log.info("Analysis JSON already exists: %s — skipping", out_json)
        return out_json
    rc = run(
        [PYTHON, str(ROOT / "scripts" / "factors_analyze_chunked.py"),
         "--universe", "all", "--output", str(ANALYSIS_DIR)],
        "factors analyze (universe=all, chunked)",
    )
    if rc != 0 or not out_json.exists():
        raise RuntimeError(f"factors analyze failed; rc={rc}")
    return out_json


# ----- Phase 3: pick-by-ic + back-up old selection -----
def run_pick(analysis_json: Path) -> Path:
    log.info("=" * 60)
    log.info("Phase 3: factors pick-by-ic → %s", SEL_NEW)
    if SEL_NEW.exists():
        log.info("New selection already exists: %s — skipping pick", SEL_NEW)
        return SEL_NEW
    rc = run(
        [PYTHON, "-m", "stockpool", "factors", "pick-by-ic",
         "--input", str(analysis_json),
         "--output", str(SEL_NEW),
         "--top-n", "20", "--max-corr", "0.6", "--min-ir", "0.05"],
        f"factors pick-by-ic ← {analysis_json.name}",
    )
    if rc != 0 or not SEL_NEW.exists():
        raise RuntimeError(f"pick-by-ic failed; rc={rc}")
    return SEL_NEW


# ----- Phase 4: diff selections -----
def diff_selections() -> None:
    log.info("=" * 60)
    log.info("Phase 4: diff selections")
    old = set(json.loads(SEL_OLD.read_text(encoding="utf-8"))["factors"])
    new = set(json.loads(SEL_NEW.read_text(encoding="utf-8"))["factors"])
    fund = {"roe", "roa", "gross_margin", "net_margin", "revenue_yoy",
            "pe", "pb", "market_cap", "log_market_cap"}
    added = sorted(new - old)
    dropped = sorted(old - new)
    kept = sorted(old & new)
    new_fund = sorted(new & fund)
    log.info("kept    (%d): %s", len(kept), kept)
    log.info("added   (%d): %s", len(added), added)
    log.info("dropped (%d): %s", len(dropped), dropped)
    log.info("→ fundamental factors entering new selection (%d): %s",
             len(new_fund), new_fund)


# ----- Phase 5: write AB yaml -----
AB_TEMPLATE = """\
# Auto-generated: compare pre-fundamentals selection vs post-fundamentals selection.
# Only `factors_file` differs; selector / weighter / horizon held constant.
base_config: config.yaml

arms:
  pre_fundamentals:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        selector:
          type: lasso
          lasso: {alpha: 0.001, max_iter: 1000, tol: 1.0e-6}
        weighter:
          type: ic
          ic: {use_rank: true, min_abs_ic: 0.0}
        thresholds: {strong_buy: 0.90, buy: 0.70, sell: 0.30, strong_sell: 0.10}
        buy_verdicts:     ["buy", "strong_buy"]
        sell_verdicts:    ["sell", "strong_sell"]
        refresh_verdicts: ["strong_buy"]
    backtest:
      equity_curve_holding_days: [5]

  post_fundamentals:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection_post_fundamentals.json
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        selector:
          type: lasso
          lasso: {alpha: 0.001, max_iter: 1000, tol: 1.0e-6}
        weighter:
          type: ic
          ic: {use_rank: true, min_abs_ic: 0.0}
        thresholds: {strong_buy: 0.90, buy: 0.70, sell: 0.30, strong_sell: 0.10}
        buy_verdicts:     ["buy", "strong_buy"]
        sell_verdicts:    ["sell", "strong_sell"]
        refresh_verdicts: ["strong_buy"]
    backtest:
      equity_curve_holding_days: [5]
"""


def write_ab_yaml() -> Path:
    log.info("=" * 60)
    log.info("Phase 5: write %s", AB_YAML)
    AB_YAML.write_text(AB_TEMPLATE, encoding="utf-8")
    return AB_YAML


# ----- Phase 6: run AB -----
def run_ab() -> int:
    log.info("=" * 60)
    log.info("Phase 6: run AB")
    rc = run(
        [PYTHON, "-m", "stockpool", "ab", "--config", str(AB_YAML)],
        "stockpool ab",
    )
    return rc


def main() -> int:
    log.info("post_fetch_pipeline starting (cwd=%s)", ROOT)

    wait_for_fetch()
    coverage = verify_coverage()
    if min(coverage.values()) < 0.5 * 4358:
        log.warning(
            "Coverage looks low (%s); continuing anyway but factors may still NaN",
            coverage,
        )

    analysis_json = run_analyze()
    run_pick(analysis_json)
    diff_selections()
    write_ab_yaml()
    rc = run_ab()

    ab_html = ROOT / "reports" / "ab" / "latest.html"
    if ab_html.exists():
        log.info("=" * 60)
        log.info("AB report: %s", ab_html)
    log.info("post_fetch_pipeline DONE (ab rc=%d)", rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
