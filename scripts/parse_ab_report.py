"""Quick parser for reports/ab/latest.html to extract aggregate diff table.

Usage: python scripts/parse_ab_report.py [path-to-html]
       (defaults to reports/ab/latest.html)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

METRICS = [
    "Total return",
    "Annualized return",
    "Sharpe",
    "Max drawdown",
    "Win rate",
    "Avg trade ret %",
    "Trade count",
]


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def parse(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")

    # arm names from banner: look for "Arm A: <name>" / "Arm B: <name>"
    arm_a = re.search(r"Arm A[:\s]*</?[^>]*>?\s*<[^>]*>\s*([\w\-_]+)", html)
    arm_b = re.search(r"Arm B[:\s]*</?[^>]*>?\s*<[^>]*>\s*([\w\-_]+)", html)
    if not arm_a:
        # fallback simpler: find first two distinct arm name labels in header
        names = re.findall(r"<th>([\w\-_]+) (?:mean|wins)</th>", html)
        arm_a_name = names[0] if names else "A"
        arm_b_name = next((n for n in names if n != arm_a_name), "B")
    else:
        arm_a_name, arm_b_name = arm_a.group(1), arm_b.group(1)

    # common stocks count from "N common stocks" or similar - look for digits in banner
    common = re.search(r"common[^<]*?(\d+)", html, re.IGNORECASE)
    common_n = common.group(1) if common else "?"

    print(f"Arm A: {arm_a_name}")
    print(f"Arm B: {arm_b_name}")
    print(f"Common stocks: {common_n}")
    print()
    print(f"{'Metric':<22} {'A mean':>12} {'B mean':>12} {'Δ (B-A)':>12} {'A wins':>8} {'B wins':>8}")
    print("-" * 80)

    for label in METRICS:
        # find row: <tr><td>Label ...</td><td>a_mean</td><td>a_med</td><td>b_mean</td><td>b_med</td><td><strong>delta</strong></td><td>a_wins</td><td>b_wins</td></tr>
        # label may have a lower-better span; match prefix only
        pattern = (
            r"<tr><td>"
            + re.escape(label)
            + r"[^<]*(?:<span[^>]*>[^<]*</span>)?</td>"
            + r"<td>([^<]+)</td>"
            + r"<td>([^<]+)</td>"
            + r"<td>([^<]+)</td>"
            + r"<td>([^<]+)</td>"
            + r"<td><strong>([^<]+)</strong></td>"
            + r"<td>(\d+)</td>"
            + r"<td>(\d+)</td>"
        )
        m = re.search(pattern, html)
        if not m:
            print(f"{label:<22}  <not found>")
            continue
        a_mean, _a_med, b_mean, _b_med, diff, a_wins, b_wins = m.groups()
        print(f"{label:<22} {a_mean:>12} {b_mean:>12} {diff:>12} {a_wins:>8} {b_wins:>8}")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/ab/latest.html")
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1
    parse(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
