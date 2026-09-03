"""
tune.py
Coarse joint grid search over the four SMC detector-level tuning parameters
(min_confluence, min_ob_body_ratio, min_fvg_gap_ratio, min_break_distance) against
real Kraken archive history, plus a strictly separate --evaluate-holdout path for
a one-time final check against data the search itself never touches.

    python -m backtest.tune
        Runs the full grid search over the tuning period and writes the complete
        result distribution to docs/tuning-log.md.

    python -m backtest.tune --evaluate-holdout --min-confluence 3 \
        --min-ob-body-ratio 0.5 --min-fvg-gap-ratio 0.5 --min-break-distance 50.0
        Runs exactly ONE already-chosen combination against the held-out period and
        appends the result to docs/tuning-log.md. Never invoked by the search path -
        see evaluate_holdout()'s docstring for why.

Period split
------------
Tuning period:  2025-04-01 -> 2025-12-31 (9 months). The grid search only ever
touches this window.
Held-out period: 2026-01-01 -> 2026-03-31 (the most recent quarter in the merged
archive). Reserved for a single deliberate --evaluate-holdout run after reviewing
the tuning period's full result distribution - never read by run_search().

This replaces an originally-planned Sep 2025 - Sep 2026 window: the merged
archive's real, confirmed boundary is 2026-03-31 (see docs/detector-logic.md and
the archive-import work), not "today" - there is no data past that date to tune or
validate against yet.

Why this split and not walk-forward/k-fold/etc.: the simplest structurally-sound
split for a first real validation pass on this project - tune on one contiguous
stretch, confirm on a disjoint later stretch the search never saw. Anything fancier
is future work once this baseline separation is trusted and once more archive
history accumulates.

Why a coarse joint grid, not a fine sweep: with 4 parameters, even 4 values each is
already 256 backtest runs (see the parallelism note below) - a fine sweep would
multiply that combinatorially for a first pass whose only job is to find whether
any *region* of this space looks structurally different from all-off, not to
pinpoint an exact optimum. Fine-tuning around a promising region is legitimate
future work; doing it now, on one 9-month window, with SMC signals already known to
be sparse on spot (see the kickoff notes' SMC-on-spot caveat), would mostly just be
fitting this particular window's noise more precisely.

Parallelism: a single backtest run over the ~79,200-candle tuning window takes on
the order of a minute (data/smc.py's detectors are correctness-first, not
vectorized - see docs/detector-logic.md; that's a legitimate future optimization
target, out of scope for building this harness). 256 runs sequentially would take
hours. run_search() uses ProcessPoolExecutor across all available CPU cores instead
- each grid point is a fully independent backtest, so this doesn't change any
result, only wall-clock time. Each worker process loads+slices the snapshot once
(via an initializer) and reuses it for every point that lands on that worker,
rather than reloading a ~100MB CSV per grid point.
"""
import argparse
import itertools
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.risk import RiskManager
from backtest.runner import FillConfig, run_backtest
from data.snapshot import load_snapshot
from signals.smc_aggregator import SMCSignalAggregator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT = PROJECT_ROOT / "data_snapshots" / "BTC-USD_5m_2013-10-06_2026-03-31.csv"
TUNING_LOG_PATH = PROJECT_ROOT / "docs" / "tuning-log.md"

TUNING_START = "2025-04-01"
TUNING_END = "2025-12-31"
HOLDOUT_START = "2026-01-01"
HOLDOUT_END = "2026-03-31"

INITIAL_BALANCE = 20_000.0

# Below this trade count, a combination's stats are noted as low-sample in the log
# rather than silently ranked alongside combinations with enough trades to mean
# something - matches this project's own "distinguish stints from bar-level
# observations, don't let a large-looking number create false confidence" habit.
LOW_SAMPLE_TRADE_THRESHOLD = 20

# ---- Coarse joint grid - 4 values per parameter, each with a stated reason -------

MIN_CONFLUENCE_GRID = [1, 2, 3, 4]
# Spans the aggregator's full valid range (4 detectors -> net confluence -4..+4,
# and only the positive side is meaningful for a long-only, no-shorting strategy).
# 1 = any single detector agreeing is enough (loosest, most trades). 3 is
# tradingagents-kr's stock-tuned value - included as a reference point already
# documented as not transferring (see signals/smc_aggregator.py), not a favored
# guess. 4 = all 4 detectors must agree (strictest, fewest trades).

MIN_OB_BODY_RATIO_GRID = [0.0, 0.5, 1.0, 2.0]
MIN_FVG_GAP_RATIO_GRID = [0.0, 0.5, 1.0, 2.0]
# ATR-relative ratios (data/smc.py's own unit for these two filters - see
# docs/detector-logic.md). 0.0 = off (the current default/baseline, so this grid
# always includes "no change from today"). 0.5 = body/gap at least half the recent
# average bar range (mild). 1.0 = at least a full ATR (a genuinely average-or-larger
# move). 2.0 = double ATR (only unusually large displacement counts). Even steps
# bracketing "no filter" through "strict".

MIN_BREAK_DISTANCE_GRID = [0.0, 25.0, 50.0, 100.0]
# Raw price distance in dollars, not an ATR ratio (that's what was specified for
# this parameter - see data/smc.py). 0.0 = off. 50.0 is not a new arbitrary number:
# it matches this project's own min_sl_distance default (backtest/risk.py) - the
# same noise-floor assumption already made elsewhere in this codebase, reused here
# rather than inventing a second one. 25.0/100.0 bracket that reference point at
# half and double.


@dataclass(frozen=True)
class GridPoint:
    min_confluence: int
    min_ob_body_ratio: float
    min_fvg_gap_ratio: float
    min_break_distance: float


def _run_point(point: GridPoint, df: pd.DataFrame) -> dict[str, Any]:
    """Runs one parameter combination's backtest against df and returns its
    params + full trade-level metrics as a single flat dict - the row shape both
    run_search() and evaluate_holdout() produce."""
    aggregator = SMCSignalAggregator(min_confluence=point.min_confluence)
    risk = RiskManager(initial_balance=INITIAL_BALANCE)
    result = run_backtest(
        df, aggregator, risk, FillConfig(),
        min_ob_body_ratio=point.min_ob_body_ratio,
        min_fvg_gap_ratio=point.min_fvg_gap_ratio,
        min_break_distance=point.min_break_distance,
    )
    return {
        "min_confluence": point.min_confluence,
        "min_ob_body_ratio": point.min_ob_body_ratio,
        "min_fvg_gap_ratio": point.min_fvg_gap_ratio,
        "min_break_distance": point.min_break_distance,
        **result.metrics(),
    }


# Populated once per worker process by _init_worker() - avoids each of the 256
# grid-point tasks reloading/re-slicing the ~100MB snapshot CSV independently.
_worker_df: pd.DataFrame | None = None


def _init_worker(snapshot_path: Path, start: str, end: str) -> None:
    global _worker_df
    full = load_snapshot(snapshot_path)
    _worker_df = full.loc[start:end]


def _run_point_in_worker(point: GridPoint) -> dict[str, Any]:
    return _run_point(point, _worker_df)


def build_grid() -> list[GridPoint]:
    return [
        GridPoint(c, ob, fvg, brk)
        for c, ob, fvg, brk in itertools.product(
            MIN_CONFLUENCE_GRID, MIN_OB_BODY_RATIO_GRID, MIN_FVG_GAP_RATIO_GRID, MIN_BREAK_DISTANCE_GRID,
        )
    ]


def run_search(
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    start: str = TUNING_START,
    end: str = TUNING_END,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Runs the full coarse grid against [start, end) of snapshot_path, in parallel
    across worker processes. Returns every combination's result row - the full
    distribution, not a filtered top-N (that's the caller's job, e.g.
    write_tuning_log() below)."""
    grid = build_grid()
    print(f"Running {len(grid)} grid points over {start} -> {end} ...")

    results: list[dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker, initargs=(snapshot_path, start, end),
    ) as pool:
        futures = [pool.submit(_run_point_in_worker, point) for point in grid]
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results.append(row)
            elapsed = time.time() - t0
            print(
                f"[{i:>3}/{len(grid)}] {elapsed:6.1f}s  "
                f"conf={row['min_confluence']} ob={row['min_ob_body_ratio']} "
                f"fvg={row['min_fvg_gap_ratio']} brk={row['min_break_distance']:<5} "
                f"-> trades={row['total_trades']:>4} roi={row['roi_pct']:>7.2f}% "
                f"pf={row['profit_factor']:.2f}"
            )

    print(f"Done in {time.time() - t0:.1f}s.")
    return results


def evaluate_holdout(
    min_confluence: int,
    min_ob_body_ratio: float,
    min_fvg_gap_ratio: float,
    min_break_distance: float,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    start: str = HOLDOUT_START,
    end: str = HOLDOUT_END,
) -> dict[str, Any]:
    """Runs exactly ONE already-chosen parameter combination against the held-out
    period. Meant to be invoked once, deliberately, by a human who has reviewed the
    full tuning-log.md distribution and picked a combination to confirm - not
    looped, not swept, and never called from run_search() or the default CLI
    action. Running this repeatedly against different combinations would quietly
    turn the "held-out" period into a second tuning set, defeating the entire point
    of keeping it separate.
    """
    df = load_snapshot(snapshot_path).loc[start:end]
    point = GridPoint(min_confluence, min_ob_body_ratio, min_fvg_gap_ratio, min_break_distance)
    return _run_point(point, df)


# ---- Reporting --------------------------------------------------------------------

_METRIC_COLUMNS = [
    ("total_trades", "Trades", "{:d}"),
    ("win_rate_pct", "Win%", "{:.1f}"),
    ("profit_factor", "PF", "{:.2f}"),
    ("total_pnl", "PnL", "{:.0f}"),
    ("roi_pct", "ROI%", "{:.2f}"),
    ("max_drawdown_pct", "MaxDD%", "{:.2f}"),
    ("avg_r_multiple", "AvgR", "{:.2f}"),
    ("median_r_multiple", "MedR", "{:.2f}"),
    ("std_r_multiple", "StdR", "{:.2f}"),
]


def _metric_cells(row: dict[str, Any]) -> str:
    return " | ".join(fmt.format(row[key]) for key, _, fmt in _METRIC_COLUMNS)


def _format_row(row: dict[str, Any]) -> str:
    flag = "†" if row["total_trades"] < LOW_SAMPLE_TRADE_THRESHOLD else " "
    return (
        f"| {row['min_confluence']} | {row['min_ob_body_ratio']:.2f} | {row['min_fvg_gap_ratio']:.2f} | "
        f"{row['min_break_distance']:.1f} | {_metric_cells(row)} | {flag} |"
    )


def _grid_search_section(results: list[dict[str, Any]], start: str, end: str) -> str:
    ranked = sorted(results, key=lambda r: r["roi_pct"], reverse=True)
    low_sample = [r for r in results if r["total_trades"] < LOW_SAMPLE_TRADE_THRESHOLD]
    zero_trade = [r for r in results if r["total_trades"] == 0]
    trade_counts = [r["total_trades"] for r in results]

    header_cells = " | ".join(label for _, label, _ in _METRIC_COLUMNS)
    sep_cells = " | ".join("---" for _ in _METRIC_COLUMNS)

    lines = [
        "## Grid search results",
        "",
        f"Tuning period: **{start} -> {end}** ({TUNING_START} - {TUNING_END} by default). "
        f"{len(results)} combinations, sorted by ROI (highest first) - full distribution, not a top-N cut.",
        "",
        f"- Combinations with fewer than {LOW_SAMPLE_TRADE_THRESHOLD} trades are flagged `†` and should be read "
        "as low-sample, not as a genuine result - a handful of trades over 9 months of a sparse-signal "
        "strategy is not enough to distinguish structure from chance.",
        f"- {len(low_sample)}/{len(results)} combinations are low-sample (`†`); {len(zero_trade)} produced zero trades.",
        f"- Trade count across the grid: min {min(trade_counts)}, median {statistics.median(trade_counts):.0f}, "
        f"max {max(trade_counts)}.",
        "",
        f"| Conf | OB ratio | FVG ratio | Break $ | {header_cells} | † |",
        f"|---|---|---|---|{sep_cells}|---|",
    ]
    lines.extend(_format_row(row) for row in ranked)
    lines.append("")
    lines.append(
        "**Reading this table**: this is an in-sample result on a single 9-month window - a high ROI/PF "
        "here is a candidate worth checking, not a validated edge. See the kickoff notes' caveats on "
        "SMC-on-spot signal sparsity, bars-vs-stints sample-size inflation, and \"results that look good "
        "by chance vs. results with a structural explanation\" before treating any row here as a decision. "
        "The held-out period ({}..{}) is reserved for a single deliberate `--evaluate-holdout` check on "
        "whichever combination gets chosen from this table - not for re-running the search.".format(
            HOLDOUT_START, HOLDOUT_END
        )
    )
    return "\n".join(lines)


def write_tuning_log(results: list[dict[str, Any]], start: str, end: str, path: Path = TUNING_LOG_PATH) -> None:
    """Writes the grid search section fresh, but preserves any existing '## Holdout
    evaluations' section already in the file (from prior --evaluate-holdout runs) -
    re-running the search must not erase a past holdout check's record."""
    preserved_tail = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        marker = "\n## Holdout evaluations"
        idx = existing.find(marker)
        if idx != -1:
            preserved_tail = existing[idx:]

    header = (
        "# Parameter tuning log\n\n"
        "Generated by `python -m backtest.tune`. See that module's docstring for the full methodology "
        "(period split, grid rationale, why parallel, why not a finer sweep).\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + _grid_search_section(results, start, end) + "\n" + preserved_tail, encoding="utf-8")
    print(f"Wrote {path}")


def append_holdout_result(row: dict[str, Any], start: str, end: str, path: Path = TUNING_LOG_PATH) -> None:
    """Appends a dated '### Holdout evaluation' entry under '## Holdout
    evaluations', creating either as needed. Never called automatically - only from
    the --evaluate-holdout CLI path."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Parameter tuning log\n", encoding="utf-8")

    existing = path.read_text(encoding="utf-8")
    if "## Holdout evaluations" not in existing:
        existing = existing.rstrip("\n") + "\n\n## Holdout evaluations\n\nOne-time deliberate checks via " \
            "`python -m backtest.tune --evaluate-holdout ...`, never run automatically. " \
            f"Held-out period: {HOLDOUT_START} -> {HOLDOUT_END}.\n"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header_cells = " | ".join(label for _, label, _ in _METRIC_COLUMNS)
    sep_cells = " | ".join("---" for _ in _METRIC_COLUMNS)
    entry = (
        f"\n### {timestamp}\n\n"
        f"Params: `min_confluence={row['min_confluence']}`, `min_ob_body_ratio={row['min_ob_body_ratio']}`, "
        f"`min_fvg_gap_ratio={row['min_fvg_gap_ratio']}`, `min_break_distance={row['min_break_distance']}`. "
        f"Evaluated on {start} -> {end}.\n\n"
        f"| {header_cells} |\n|{sep_cells}|\n"
        f"| {_metric_cells(row)} |\n"  # metric cells only - params already stated in the prose above
    )
    path.write_text(existing.rstrip("\n") + "\n" + entry, encoding="utf-8")
    print(f"Appended holdout evaluation to {path}")


# ---- CLI ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--max-workers", type=int, default=None, help="default: os.cpu_count()")
    parser.add_argument(
        "--evaluate-holdout", action="store_true",
        help="Run ONE combination against the held-out period instead of the grid search. "
             "Requires --min-confluence/--min-ob-body-ratio/--min-fvg-gap-ratio/--min-break-distance.",
    )
    parser.add_argument("--min-confluence", type=int)
    parser.add_argument("--min-ob-body-ratio", type=float)
    parser.add_argument("--min-fvg-gap-ratio", type=float)
    parser.add_argument("--min-break-distance", type=float)
    args = parser.parse_args()

    if args.evaluate_holdout:
        missing = [
            name for name, val in [
                ("--min-confluence", args.min_confluence), ("--min-ob-body-ratio", args.min_ob_body_ratio),
                ("--min-fvg-gap-ratio", args.min_fvg_gap_ratio), ("--min-break-distance", args.min_break_distance),
            ] if val is None
        ]
        if missing:
            parser.error(f"--evaluate-holdout requires all of: {', '.join(missing)}")

        print(
            f"Evaluating ONE combination against the held-out period ({HOLDOUT_START} -> {HOLDOUT_END}) - "
            "this is a deliberate one-time check, not part of the search."
        )
        row = evaluate_holdout(
            args.min_confluence, args.min_ob_body_ratio, args.min_fvg_gap_ratio, args.min_break_distance,
            snapshot_path=args.snapshot,
        )
        for key, label, fmt in _METRIC_COLUMNS:
            print(f"  {label:>7}: {fmt.format(row[key])}")
        append_holdout_result(row, HOLDOUT_START, HOLDOUT_END)
        return

    results = run_search(snapshot_path=args.snapshot, max_workers=args.max_workers)
    write_tuning_log(results, TUNING_START, TUNING_END)


if __name__ == "__main__":
    main()
