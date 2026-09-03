"""
tune.py
Coarse joint grid search over three SMC detector-level tuning parameters
(min_confluence, min_ob_body_ratio, min_fvg_gap_ratio) against real Kraken archive
history, plus a strictly separate --evaluate-holdout path for a one-time final check
against data the search itself never touches.

min_break_distance (detect_bos_choch's close-crossing threshold) is deliberately
NOT a grid parameter - removed outright, not deprioritized. Confirmed via a direct
code trace: signals/smc_aggregator.py does not read structure_bias/structure_events
(BOS/CHoCH's output) anywhere - the 2026-09 rebuild's 5-vote aggregator reads order
block/FVG/trendline/channel/fakeout_trap only - and backtest/runner.py's
_structural_stop_loss reads only order_blocks, never structure_bias/structure_events
either. min_break_distance's entire causal chain terminates at detect_bos_choch's own
output, which nothing downstream reads. Sweeping it could not have changed a single
backtest result in this codebase - see docs/tuning-log.md for the same note kept
alongside the actual results.

    python -m backtest.tune
        Runs the full grid search over the tuning period and writes the complete
        result distribution to docs/tuning-log.md.

    python -m backtest.tune --evaluate-holdout --min-confluence 3 \
        --min-ob-body-ratio 0.5 --min-fvg-gap-ratio 0.5
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

Why a coarse joint grid, not a fine sweep: with 3 parameters, even 4 values each is
already 64 backtest runs (see the parallelism note below) - a fine sweep would
multiply that combinatorially for a first pass whose only job is to find whether
any *region* of this space looks structurally different from all-off, not to
pinpoint an exact optimum. Fine-tuning around a promising region is legitimate
future work; doing it now, on one 9-month window, with SMC signals already known to
be sparse on spot (see the kickoff notes' SMC-on-spot caveat), would mostly just be
fitting this particular window's noise more precisely.

Parallelism: a single backtest run over the ~79,200-candle tuning window takes on
the order of a minute. 64 runs sequentially would still take over an hour.
run_search() uses ProcessPoolExecutor across all available CPU cores instead - each
grid point is a fully independent backtest, so this doesn't change any result, only
wall-clock time. Each worker process loads+slices the snapshot once (via an
initializer) and reuses it for every point that lands on that worker, rather than
reloading a ~100MB CSV per grid point.
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

# min_break_distance (detect_bos_choch's close-crossing threshold) is deliberately
# NOT swept here - removed outright, not deprioritized. See this module's own
# docstring for the code-trace confirming it has zero causal path to any backtest
# output: signals/smc_aggregator.py's 5-vote model never reads structure_bias/
# structure_events, and backtest/runner.py's stop-loss placement reads only
# order_blocks. Every backtest run below uses run_backtest()'s own default
# (DEFAULT_MIN_BREAK_DISTANCE = 0.0) implicitly, since it's never passed through.


@dataclass(frozen=True)
class GridPoint:
    min_confluence: int
    min_ob_body_ratio: float
    min_fvg_gap_ratio: float


def _run_point(point: GridPoint, df: pd.DataFrame, allow_short: bool = False) -> dict[str, Any]:
    """Runs one parameter combination's backtest against df and returns its
    params + full trade-level metrics as a single flat dict - the row shape both
    run_search() and evaluate_holdout() produce.

    allow_short defaults to False (the only executable-on-Kraken-Spot mode) and
    must only ever be set True from run_diagnostic_long_short() below - see that
    function's docstring and backtest/runner.py's FillEngine.allow_short docstring
    for why this is diagnostic-only, never a real trading mode."""
    aggregator = SMCSignalAggregator(min_confluence=point.min_confluence)
    # skip_if_capital_capped=True is fixed here, not a swept grid parameter -
    # it's an established fix (backtest/risk.py, commit 2a396ab; re-verified
    # compatible with the post-rebuild detector/aggregator code via a live
    # smoke test), not an open question this grid is meant to explore. Sweeping
    # it would mean half the grid re-measures a known-worse (forced-sizing)
    # baseline instead of spending that compute on genuinely unknown parameters.
    risk = RiskManager(initial_balance=INITIAL_BALANCE, skip_if_capital_capped=True)
    result = run_backtest(
        df, aggregator, risk, FillConfig(),
        min_ob_body_ratio=point.min_ob_body_ratio,
        min_fvg_gap_ratio=point.min_fvg_gap_ratio,
        allow_short=allow_short,
    )
    return {
        "min_confluence": point.min_confluence,
        "min_ob_body_ratio": point.min_ob_body_ratio,
        "min_fvg_gap_ratio": point.min_fvg_gap_ratio,
        # Always computed (cheap, harmless when allow_short=False - short_trades
        # is just always 0 then) so the long+short diagnostic can report the split
        # without a separate code path.
        "long_trades": sum(1 for t in result.trades if t.side == "BUY"),
        "short_trades": sum(1 for t in result.trades if t.side == "SELL"),
        **result.metrics(),
    }


# Populated once per worker process by _init_worker() - avoids each of the 256
# grid-point tasks reloading/re-slicing the ~100MB snapshot CSV independently.
_worker_df: pd.DataFrame | None = None
_worker_allow_short: bool = False


def _init_worker(snapshot_path: Path, start: str, end: str, allow_short: bool = False) -> None:
    global _worker_df, _worker_allow_short
    full = load_snapshot(snapshot_path)
    _worker_df = full.loc[start:end]
    _worker_allow_short = allow_short


def _run_point_in_worker(point: GridPoint) -> dict[str, Any]:
    return _run_point(point, _worker_df, allow_short=_worker_allow_short)


def build_grid() -> list[GridPoint]:
    return [
        GridPoint(c, ob, fvg)
        for c, ob, fvg in itertools.product(
            MIN_CONFLUENCE_GRID, MIN_OB_BODY_RATIO_GRID, MIN_FVG_GAP_RATIO_GRID,
        )
    ]


def run_search(
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    start: str = TUNING_START,
    end: str = TUNING_END,
    max_workers: int | None = None,
    allow_short: bool = False,
) -> list[dict[str, Any]]:
    """Runs the full coarse grid against [start, end) of snapshot_path, in parallel
    across worker processes. Returns every combination's result row - the full
    distribution, not a filtered top-N (that's the caller's job, e.g.
    write_tuning_log() below).

    allow_short defaults to False (the real, executable-on-Kraken-Spot mode this
    project runs in). It exists here only so run_diagnostic_long_short() can reuse
    this same parallel search machinery for its long+short comparison - the default
    CLI action (`python -m backtest.tune`) never passes True."""
    grid = build_grid()
    label = "long+short (DIAGNOSTIC)" if allow_short else "long-only"
    print(f"Running {len(grid)} grid points ({label}) over {start} -> {end} ...")

    results: list[dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker, initargs=(snapshot_path, start, end, allow_short),
    ) as pool:
        futures = [pool.submit(_run_point_in_worker, point) for point in grid]
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results.append(row)
            elapsed = time.time() - t0
            print(
                f"[{i:>3}/{len(grid)}] {elapsed:6.1f}s  "
                f"conf={row['min_confluence']} ob={row['min_ob_body_ratio']} "
                f"fvg={row['min_fvg_gap_ratio']} "
                f"-> trades={row['total_trades']:>4} roi={row['roi_pct']:>7.2f}% "
                f"pf={row['profit_factor']:.2f}"
            )

    print(f"Done in {time.time() - t0:.1f}s.")
    return results


def run_diagnostic_long_short(
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    start: str = TUNING_START,
    end: str = TUNING_END,
    max_workers: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Runs the full grid twice over the identical period - once long-only
    (allow_short=False, the only mode this project ever actually trades in) and
    once with shorts additionally enabled (allow_short=True) - to measure how much
    of the long-only result is attributable to missing the short side of this
    window's whipsaw, rather than to the strategy's entries/exits being wrong.

    Re-runs the long-only leg fresh here rather than reusing the numbers already
    committed in docs/tuning-log.md, so the comparison is guaranteed apples-to-
    apples against the exact same code path in the same process, not parsed back
    out of already-rounded markdown.

    **DIAGNOSTIC ONLY.** Kraken Spot has no margin and cannot open a short position
    at all. This function - and the allow_short=True leg specifically - is never
    called from live/trader.py or viz/control.py's paper trading, and never will
    be; see FillEngine.allow_short's docstring in backtest/runner.py. If this
    comparison shows shorts meaningfully changing the picture, that's a signal to
    revisit the Spot-vs-Margin/Futures venue decision later - not something to
    wire into execution off the back of this diagnostic alone.
    """
    print("=== Diagnostic leg 1/2: long-only (real, executable mode) ===")
    long_only = run_search(snapshot_path, start, end, max_workers, allow_short=False)
    print("=== Diagnostic leg 2/2: long+short (DIAGNOSTIC ONLY - not executable on Kraken Spot) ===")
    long_short = run_search(snapshot_path, start, end, max_workers, allow_short=True)
    return long_only, long_short


def evaluate_holdout(
    min_confluence: int,
    min_ob_body_ratio: float,
    min_fvg_gap_ratio: float,
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
    point = GridPoint(min_confluence, min_ob_body_ratio, min_fvg_gap_ratio)
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
        f"{_metric_cells(row)} | {flag} |"
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
        f"{len(results)} combinations, sorted by ROI (highest first) - full distribution, not a top-N cut. "
        "Swept: `min_confluence`, `min_ob_body_ratio`, `min_fvg_gap_ratio`.",
        "",
        "- **`min_break_distance` is deliberately excluded from this grid - not deprioritized, removed.** "
        "It gates `detect_bos_choch`'s close-crossing threshold, and a direct code trace confirms that "
        "detector's output (`structure_bias`/`structure_events`) has no causal path to any backtest "
        "result: `signals/smc_aggregator.py`'s 5-vote model (order block/FVG/trendline/channel/"
        "fakeout_trap) never reads it, and `backtest/runner.py`'s stop-loss placement reads only "
        "`order_blocks`. Sweeping it could not have changed a single result in this table - every "
        "combination below implicitly ran with `min_break_distance` at its unswept default (`0.0`).",
        f"- Combinations with fewer than {LOW_SAMPLE_TRADE_THRESHOLD} trades are flagged `†` and should be read "
        "as low-sample, not as a genuine result - a handful of trades over 9 months of a sparse-signal "
        "strategy is not enough to distinguish structure from chance.",
        f"- {len(low_sample)}/{len(results)} combinations are low-sample (`†`); {len(zero_trade)} produced zero trades.",
        f"- Trade count across the grid: min {min(trade_counts)}, median {statistics.median(trade_counts):.0f}, "
        f"max {max(trade_counts)}.",
        "",
        f"| Conf | OB ratio | FVG ratio | {header_cells} | † |",
        f"|---|---|---|{sep_cells}|---|",
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


# The log file is always written in this section order: grid search results,
# then (if it's ever been run) the long+short diagnostic, then (if it's ever been
# run) holdout evaluations. Every writer below preserves whichever of these
# sections it isn't itself responsible for, found by these markers - never by
# slicing a formatted row string (see the "Caught and fixed a real bug" note in
# this project's git history for why that's specifically disallowed here).
_DIAGNOSTIC_MARKER = "\n## Diagnostic: long+short"
_HOLDOUT_MARKER = "\n## Holdout evaluations"


def _find_earliest_marker(text: str, markers: list[str]) -> int:
    """Position of whichever marker appears first in text, or len(text) if none do."""
    positions = [p for p in (text.find(m) for m in markers) if p != -1]
    return min(positions) if positions else len(text)


def write_tuning_log(results: list[dict[str, Any]], start: str, end: str, path: Path = TUNING_LOG_PATH) -> None:
    """Writes the grid search section fresh, but preserves any existing diagnostic
    and/or holdout sections already in the file (from prior --diagnostic-allow-short
    or --evaluate-holdout runs, in either order) - re-running the search must not
    erase either record."""
    preserved_tail = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        cut = _find_earliest_marker(existing, [_DIAGNOSTIC_MARKER, _HOLDOUT_MARKER])
        preserved_tail = existing[cut:]

    header = (
        "# Parameter tuning log\n\n"
        "Generated by `python -m backtest.tune`. See that module's docstring for the full methodology "
        "(period split, grid rationale, why parallel, why not a finer sweep).\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + _grid_search_section(results, start, end) + "\n" + preserved_tail, encoding="utf-8")
    print(f"Wrote {path}")


def _diagnostic_short_section_text(
    long_only: list[dict[str, Any]], long_short: list[dict[str, Any]], start: str, end: str,
) -> str:
    key = lambda r: (r["min_confluence"], r["min_ob_body_ratio"], r["min_fvg_gap_ratio"])
    by_long_only = {key(r): r for r in long_only}
    by_long_short = {key(r): r for r in long_short}
    paired_keys = sorted(by_long_only.keys() & by_long_short.keys())

    long_only_rois = [by_long_only[k]["roi_pct"] for k in paired_keys]
    long_short_rois = [by_long_short[k]["roi_pct"] for k in paired_keys]
    roi_deltas = [by_long_short[k]["roi_pct"] - by_long_only[k]["roi_pct"] for k in paired_keys]

    lines = [
        "## Diagnostic: long+short (BACKTEST ONLY - not executable on Kraken Spot)",
        "",
        "**This is not a trading mode.** Kraken Spot has no margin and cannot open a short "
        "position at all - `live/trader.py` and paper trading (`viz/control.py`) never enable "
        "this and never will (see `FillEngine.allow_short`'s docstring in `backtest/runner.py`). "
        "This section exists only to measure how much of the long-only result above is "
        "attributable to missing the short side of this window's whipsaw, not to propose actually "
        "shorting on this venue. If shorts meaningfully change the picture, that's a signal to "
        "revisit the Spot-vs-Margin/Futures venue decision later - not something to wire into "
        "execution off the back of this diagnostic alone.",
        "",
        f"Same grid, same period (**{start} -> {end}**), same {len(paired_keys)} combinations - the only "
        "difference is BEARISH signals are additionally allowed to open a simulated short instead of "
        "being discarded.",
        "",
        f"- Best ROI: long-only {max(long_only_rois):.2f}% vs. long+short {max(long_short_rois):.2f}%",
        f"- Median ROI: long-only {statistics.median(long_only_rois):.2f}% vs. long+short "
        f"{statistics.median(long_short_rois):.2f}%",
        f"- Median ROI delta (long+short minus long-only), across all {len(paired_keys)} paired "
        f"combinations: {statistics.median(roi_deltas):+.2f} points",
        f"- Combinations where long+short beat long-only: {sum(1 for d in roi_deltas if d > 0)}/{len(paired_keys)}",
        "",
        "| Conf | OB ratio | FVG ratio | Long-only ROI% | Long+short ROI% | Δ ROI (pts) | "
        "Long+short trades (long/short) |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in sorted(paired_keys, key=lambda k: by_long_short[k]["roi_pct"] - by_long_only[k]["roi_pct"], reverse=True):
        lo, ls = by_long_only[k], by_long_short[k]
        delta = ls["roi_pct"] - lo["roi_pct"]
        lines.append(
            f"| {k[0]} | {k[1]:.2f} | {k[2]:.2f} | {lo['roi_pct']:.2f} | {ls['roi_pct']:.2f} | "
            f"{delta:+.2f} | {ls['total_trades']} ({ls['long_trades']}/{ls['short_trades']}) |"
        )
    return "\n".join(lines)


def write_diagnostic_short_section(
    long_only: list[dict[str, Any]], long_short: list[dict[str, Any]], start: str, end: str,
    path: Path = TUNING_LOG_PATH,
) -> None:
    """Writes/replaces the '## Diagnostic: long+short' section, preserving the
    grid-search section before it and the holdout section after it (in either
    presence/absence combination) - never called automatically, only from the
    --diagnostic-allow-short CLI path."""
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Parameter tuning log\n"

    diag_start = existing.find(_DIAGNOSTIC_MARKER)
    if diag_start != -1:
        # Replace a prior diagnostic run rather than duplicating it.
        holdout_after_diag = existing.find(_HOLDOUT_MARKER, diag_start + 1)
        head = existing[:diag_start]
        tail = existing[holdout_after_diag:] if holdout_after_diag != -1 else ""
    else:
        holdout_start = existing.find(_HOLDOUT_MARKER)
        head = existing[:holdout_start] if holdout_start != -1 else existing
        tail = existing[holdout_start:] if holdout_start != -1 else ""

    section = _diagnostic_short_section_text(long_only, long_short, start, end)
    path.write_text(head.rstrip("\n") + "\n\n" + section + "\n\n" + tail.lstrip("\n"), encoding="utf-8")
    print(f"Wrote diagnostic long+short section to {path}")


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
        f"`min_fvg_gap_ratio={row['min_fvg_gap_ratio']}`. "
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
             "Requires --min-confluence/--min-ob-body-ratio/--min-fvg-gap-ratio.",
    )
    parser.add_argument(
        "--diagnostic-allow-short", action="store_true",
        help="Re-run the full grid twice (long-only and long+short) over the tuning period and write a "
             "comparison to docs/tuning-log.md. DIAGNOSTIC ONLY - Kraken Spot cannot execute short "
             "positions; this never affects live/trader.py or paper trading. See "
             "run_diagnostic_long_short()'s docstring.",
    )
    parser.add_argument("--min-confluence", type=int)
    parser.add_argument("--min-ob-body-ratio", type=float)
    parser.add_argument("--min-fvg-gap-ratio", type=float)
    args = parser.parse_args()

    if args.evaluate_holdout:
        missing = [
            name for name, val in [
                ("--min-confluence", args.min_confluence), ("--min-ob-body-ratio", args.min_ob_body_ratio),
                ("--min-fvg-gap-ratio", args.min_fvg_gap_ratio),
            ] if val is None
        ]
        if missing:
            parser.error(f"--evaluate-holdout requires all of: {', '.join(missing)}")

        print(
            f"Evaluating ONE combination against the held-out period ({HOLDOUT_START} -> {HOLDOUT_END}) - "
            "this is a deliberate one-time check, not part of the search."
        )
        row = evaluate_holdout(
            args.min_confluence, args.min_ob_body_ratio, args.min_fvg_gap_ratio,
            snapshot_path=args.snapshot,
        )
        for key, label, fmt in _METRIC_COLUMNS:
            print(f"  {label:>7}: {fmt.format(row[key])}")
        append_holdout_result(row, HOLDOUT_START, HOLDOUT_END)
        return

    if args.diagnostic_allow_short:
        print(
            "Running the DIAGNOSTIC long+short comparison - this re-runs the full grid twice "
            "(long-only, then long+short) and is not a trading mode. See "
            "run_diagnostic_long_short()'s docstring."
        )
        long_only, long_short = run_diagnostic_long_short(snapshot_path=args.snapshot, max_workers=args.max_workers)
        write_diagnostic_short_section(long_only, long_short, TUNING_START, TUNING_END)
        return

    results = run_search(snapshot_path=args.snapshot, max_workers=args.max_workers)
    write_tuning_log(results, TUNING_START, TUNING_END)


if __name__ == "__main__":
    main()
