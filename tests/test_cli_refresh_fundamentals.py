# tests/test_cli_refresh_fundamentals.py
"""Smoke test for --refresh-fundamentals CLI flag wiring."""
from __future__ import annotations

import sys

import pytest


def test_run_accepts_refresh_fundamentals_flag(monkeypatch, capsys):
    """`python -m stockpool run --refresh-fundamentals` should not raise unknown arg."""
    from stockpool import cli

    parser = cli._build_parser()
    args = parser.parse_args(["run", "--config", "config.yaml", "--refresh-fundamentals"])
    assert args.refresh_fundamentals is True


def test_backtest_accepts_refresh_fundamentals_flag():
    from stockpool import cli
    parser = cli._build_parser()
    args = parser.parse_args(["backtest", "--config", "config.yaml", "--refresh-fundamentals"])
    assert args.refresh_fundamentals is True


def test_portfolio_backtest_accepts_refresh_fundamentals_flag():
    from stockpool import cli
    parser = cli._build_parser()
    args = parser.parse_args(
        ["portfolio-backtest", "--config", "config.yaml", "--refresh-fundamentals"]
    )
    assert args.refresh_fundamentals is True
