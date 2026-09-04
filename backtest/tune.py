"""
tune.py
Coarse joint grid search over six SMC detector-level tuning parameters
(min_confluence, min_ob_body_ratio, min_fvg_gap_ratio, min_trendline_break_distance,
min_channel_break_distance, max_trap_retest_distance) against real Kraken archive
history, plus a strictly separate --evaluate-holdout path for a one-time final check
against data the search itself never touches.

This is the 2026-09 rebuild's grid - built on the 6-detector/5-vote system, not the
pre-rebuild one. Two things deliberately NOT swept, both documented here rather than
silently omitted:

- min_break_distance (detect_bos_choch's close-crossing threshold) - removed
  outright, not deprioritized. Confirmed via a direct code trace:
  signals/smc_aggregator.py does not read structure_bias/structure_events
  (BOS/CHoCH's output) anywhere - the 5-vote aggregator reads order block/FVG/
  trendline/channel/fakeout_trap only - and backtest/runner.py's
  _structural_stop_loss reads only order_blocks, never structure_bias/
  structure_events either. Its entire causal chain terminates at detect_bos_choch's
  own output, which nothing downstream reads. Sweeping it could not have changed a
  single backtest result in this codebase.
- trendline_points - fixed at its default (3), not swept. It's structural (changes
  what points the trendline/channel fits are computed *from* - the shape of the
  line, not a threshold that filters an already-computed value), the same category
  as swing_left_bars/swing_right_bars, which are also fixed rather than swept.
  Sweeping structural parameters alongside filter thresholds would conflate two
  different kinds of question in one grid; that's a legitimate future round, not
  this one.

    python -m backtest.tune
        Runs the full grid search over the tuning period and writes the complete
        result distribution to docs/tuning-log.md.

    python -m backtest.tune --evaluate-holdout --min-confluence 3 \
        --min-ob-body-ratio 1.0 --min-fvg-gap-ratio 1.0 \
        --min-trendline-break-distance 100.0 --min-channel-break-distance 100.0 \
        --max-trap-retest-distance 100.0
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

Why a coarse joint grid, not a fine sweep: with 6 parameters, even 3-5 values each
is already 1,215 backtest runs (see the parallelism note below) - a fine sweep
would multiply that combinatorially for a first pass whose only job is to find
whether any *region* of this space looks structurally different from all-off, not
to pinpoint an exact optimum. Fine-tuning around a promising region is legitimate
future work; doing it now, on one 9-month window, with SMC signals already known to
be sparse on spot (see the kickoff notes' SMC-on-spot caveat), would mostly just be
fitting this particular window's noise more precisely. min_confluence gets its
full 5-value range (1-5, not a further-reduced subset) since it's the aggregator's
own discrete vote-count scale, not a continuous threshold being coarsely sampled -
reducing it further would mean skipping over achievable, meaningfully different
strictness levels, not just sampling a continuous range more coarsely.

Parallelism: a single backtest run over the ~79,200-candle tuning window takes
roughly 37s (measured directly against this snapshot with the post-rebuild
detectors - trendline/channel each build their own state tracker per call, so this
is slower per-run than the pre-rebuild detectors were). 1,215 runs sequentially
would take over 12 hours. run_search() uses ProcessPoolExecutor across all
available CPU cores instead (16 on the machine this was measured on, giving an
estimated ~47 minutes wall-clock for the full grid) - each grid point is a fully
independent backtest, so this doesn't change any result, only wall-clock time.
Each worker process loads+slices the snapshot once (via an initializer) and reuses
it for every point that lands on that worker, rather than reloading a ~100MB CSV
per grid point.
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

# ---- Coarse joint grid - each parameter's own value count, each with a stated reason --

MIN_CONFLUENCE_GRID = [1, 2, 3, 4, 5]
# Spans the aggregator's full valid range (5 votes as of the 2026-09 rebuild's
# aggregator rewire -> net confluence -5..+5, and only the positive side is
# meaningful for a long-only, no-shorting strategy). All 5 achievable values are
# included, not a coarser subset - see this module's own docstring for why
# min_confluence is treated differently from the continuous-threshold parameters
# below. 1 = any single detector agreeing is enough (loosest, most trades). 3 is
# tradingagents-kr's stock-tuned value against the *old* 4-vote model - included as
# a loose reference point already documented as not transferring even before the
# vote count changed (see signals/smc_aggregator.py), not a favored guess. 5 = all
# 5 detectors must agree (strictest, fewest trades) - unreachable in the pre-rebuild
# grid, since the old aggregator only ever had 4 votes.

# Real 14-bar ATR over the tuning period (2025-04-01..2025-12-31), computed directly
# against the snapshot this grid runs against: mean $107.42, median $84.96 (the
# mean is pulled up by fat-tail volatility spikes - max $4047 - so median is the
# more representative "typical bar-to-bar range" figure). Referenced below for
# every ATR-anchored grid value, so the anchor is stated once, not re-derived per
# parameter.
_TUNING_PERIOD_MEDIAN_ATR = 84.96

MIN_OB_BODY_RATIO_GRID = [0.0, 1.0, 2.0]
MIN_FVG_GAP_RATIO_GRID = [0.0, 1.0, 2.0]
# ATR-relative ratios (data/smc.py's own unit for these two filters - see
# docs/detector-logic.md) - unchanged unit from before, reduced from 4 to 3 values
# (dropping the intermediate 0.5 "mild" step) to keep the 6-parameter grid's total
# size tractable. 0.0 = off (the current default/baseline, so this grid always
# includes "no change from today"). 1.0 = at least a full ATR (a genuinely
# average-or-larger move). 2.0 = double ATR (only unusually large displacement
# counts).

MIN_TRENDLINE_BREAK_DISTANCE_GRID = [0.0, 100.0, 200.0]
MIN_CHANNEL_BREAK_DISTANCE_GRID = [0.0, 100.0, 200.0]
# Raw price distance in dollars at the code level (detect_trendline/detect_channel
# take a dollar threshold, not an internally-ATR-scaled ratio like min_ob_body_ratio/
# min_fvg_gap_ratio - see data/smc.py), but the *values* are chosen using the same
# ATR-ratio reasoning as those two filters, anchored on _TUNING_PERIOD_MEDIAN_ATR
# ($84.96): 0.0 = off (unchanged baseline). 100.0 is approximately 1x the tuning
# period's median ATR - a break has to clear roughly a "typical bar's worth" of
# price movement past the line to count. 200.0 is approximately 2x median ATR -
# only unusually large displacement counts, same bracketing logic as the OB/FVG
# ratio filters, just expressed in dollars because the underlying parameter is a
# dollar threshold rather than an internally-scaled ratio. Swept independently (not
# tied together) even though the reasoning and candidate values are shared - the
# trendline's projected line and the channel's constructed boundary are different
# lines with potentially different noise characteristics (see detect_channel's own
# docstring), so a tight trendline threshold paired with a loose channel one (or
# vice versa) is a real, distinct grid point worth covering.

MAX_TRAP_RETEST_DISTANCE_GRID = [100.0, 200.0, float("inf")]
# Unit/rationale proposed here since the video doesn't specify one (same TODO
# treatment as every other unvalidated default in this codebase - see
# DEFAULT_MAX_TRAP_RETEST_DISTANCE's own comment in data/smc.py). max_trap_retest_
# distance answers "how far can the second extreme differ from the first and still
# count as a shallow retest, not a fresh continuation" - structurally, that is the
# same *kind* of question the OB/FVG ATR-ratio filters already answer ("is this
# price movement significant relative to prevailing volatility, or is it noise"),
# just applied to the gap between two swing extremes instead of a candle body or a
# gap size. The same justification used to pick an ATR-ratio unit for those filters
# in the first place - "BTC's price scale drifts too much for a fixed dollar
# minimum to stay meaningful" - applies identically here, so ATR-ratio (expressed
# in dollars, same as the two break-distance filters above, since the code
# parameter is a raw dollar threshold) is the natural choice, not an arbitrary
# match to the others. Values, anchored on the same _TUNING_PERIOD_MEDIAN_ATR
# ($84.96): 100.0 (~1x median ATR) - a tight tolerance, the second extreme must
# stay close to the first. 200.0 (~2x median ATR) - a looser tolerance. inf - the
# code default (DEFAULT_MAX_TRAP_RETEST_DISTANCE), reproducing "off" (any second
# extreme counts as a valid retest, however far) as the baseline reference point,
# same role 0.0 plays for every minimum-distance filter above.

# min_break_distance (detect_bos_choch's close-crossing threshold) is deliberately
# NOT swept here - removed outright, not deprioritized. See this module's own
# docstring for the code-trace confirming it has zero causal path to any backtest
# output: signals/smc_aggregator.py's 5-vote model never reads structure_bias/
# structure_events, and backtest/runner.py's stop-loss placement reads only
# order_blocks. Every backtest run below uses run_backtest()'s own default
# (DEFAULT_MIN_BREAK_DISTANCE = 0.0) implicitly, since it's never passed through.

# trendline_points is deliberately NOT swept here either - fixed at its default (3)
# for every grid point. See this module's own docstring for why (structural, same
# category as swing_left_bars/swing_right_bars, not a filter threshold).


@dataclass(frozen=True)
class GridPoint:
    min_confluence: int
    min_ob_body_ratio: float
    min_fvg_gap_ratio: float
    min_trendline_break_distance: float
    min_channel_break_distance: float
    max_trap_retest_distance: float


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
        min_trendline_break_distance=point.min_trendline_break_distance,
        min_channel_break_distance=point.min_channel_break_distance,
        max_trap_retest_distance=point.max_trap_retest_distance,
        allow_short=allow_short,
    )
    return {
        "min_confluence": point.min_confluence,
        "min_ob_body_ratio": point.min_ob_body_ratio,
        "min_fvg_gap_ratio": point.min_fvg_gap_ratio,
        "min_trendline_break_distance": point.min_trendline_break_distance,
        "min_channel_break_distance": point.min_channel_break_distance,
        "max_trap_retest_distance": point.max_trap_retest_distance,
        # Always computed (cheap, harmless when allow_short=False - short_trades
        # is just always 0 then) so the long+short diagnostic can report the split
        # without a separate code path.
        "long_trades": sum(1 for t in result.trades if t.side == "BUY"),
        "short_trades": sum(1 for t in result.trades if t.side == "SELL"),
        **result.metrics(),
    }


# Populated once per worker process by _init_worker() - avoids each of the 1,215
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
        GridPoint(c, ob, fvg, tl_brk, ch_brk, trap)
        for c, ob, fvg, tl_brk, ch_brk, trap in itertools.product(
            MIN_CONFLUENCE_GRID, MIN_OB_BODY_RATIO_GRID, MIN_FVG_GAP_RATIO_GRID,
            MIN_TRENDLINE_BREAK_DISTANCE_GRID, MIN_CHANNEL_BREAK_DISTANCE_GRID,
            MAX_TRAP_RETEST_DISTANCE_GRID,
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
                f"[{i:>4}/{len(grid)}] {elapsed:6.1f}s  "
                f"conf={row['min_confluence']} ob={row['min_ob_body_ratio']} "
                f"fvg={row['min_fvg_gap_ratio']} tl_brk={row['min_trendline_break_distance']:<5} "
                f"ch_brk={row['min_channel_break_distance']:<5} trap={row['max_trap_retest_distance']:<5} "
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
    min_trendline_break_distance: float,
    min_channel_break_distance: float,
    max_trap_retest_distance: float,
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
    point = GridPoint(
        min_confluence, min_ob_body_ratio, min_fvg_gap_ratio,
        min_trendline_break_distance, min_channel_break_distance, max_trap_retest_distance,
    )
    return _run_point(point, df)


# ---- Targeted follow-up: risk-parameter grid (rr_ratio x min_sl_distance) --------
# Deliberately separate from the 6-parameter detector grid above, not folded into a
# combined sweep: this checks the one major *risk-management* lever that has never
# been touched (backtest/risk.py's own module docstring flags both rr_ratio and
# min_sl_distance as unvalidated ports - "don't assume it's automatically good" -
# and puts them on the same re-validation checklist). Detector/confluence
# parameters are held fixed at the best statistically-meaningful (non-low-sample)
# combination the 1215-point grid found, rather than swept jointly with risk
# parameters - that would multiply an already-expensive grid by a further 16x to
# answer a question that's orthogonal to detector tuning.

# ROI -3.54%, 26 trades - the best non-low-sample (>=20 trades) result out of 1215
# detector/confluence combinations tuned on 2025-04-01 -> 2025-12-31. Not a
# genuinely profitable combination - see docs/tuning-log.md's "Grid search results"
# section, where zero of the 1215 combinations were profitable - just the best
# available baseline to hold constant while checking whether risk-parameter tuning
# is the missing lever.
BEST_KNOWN_DETECTOR_POINT = GridPoint(
    min_confluence=5,
    min_ob_body_ratio=0.0,
    min_fvg_gap_ratio=0.0,
    min_trendline_break_distance=100.0,
    min_channel_break_distance=200.0,
    max_trap_retest_distance=float("inf"),
)

RR_RATIO_GRID = [1.0, 1.5, 2.0, 3.0]
# take_profit_price() (backtest/risk.py) sets the TP at rr_ratio times the realized
# SL distance from entry. 1.0 = symmetric risk/reward, breakeven at a 50% win rate.
# 1.5 is the current shipped default - a straight port from the original crypto
# bot's risk.py, explicitly flagged in this codebase's risk.py docstring as
# unvalidated, kept here as the as-shipped reference point being audited, not a
# favored guess. 2.0 and 3.0 are the classic asymmetric R:R targets common in
# ICT/SMC-style trading (breakeven win rates 33%/25% respectively) - worth testing
# given BEST_KNOWN_DETECTOR_POINT's underlying combination runs a 61.5% win rate at
# rr_ratio's current default, comfortably above even the 3.0 breakeven bar if
# realized R holds anywhere close to intended.

MIN_SL_DISTANCE_GRID = [0.0, 50.0, 85.0, 170.0]
# calculate_size() (backtest/risk.py) rejects a trade outright (size 0) if the
# stop distance is tighter than this floor. risk.py's own docstring flags the
# shipped default ($50) as tuned for a stock-scale port, never re-validated for
# BTC's current price range, on the same checklist as rr_ratio above. Values:
# 0.0 = off (no floor - baseline "no change from the filter's absence", same role
# 0.0 plays for every other minimum-distance filter in this module). 50.0 = the
# current shipped default, kept as the literal value being audited, not dropped.
# 85.0 and 170.0 are ~1x/2x the same _TUNING_PERIOD_MEDIAN_ATR ($84.96) anchor used
# for every other distance-based filter in this module - same reasoning: a stop
# distance narrower than roughly a typical bar's worth of movement is more likely
# noise than a real structural level.


@dataclass(frozen=True)
class RiskGridPoint:
    rr_ratio: float
    min_sl_distance: float


def _run_risk_point(
    point: RiskGridPoint, df: pd.DataFrame, detector_point: GridPoint = BEST_KNOWN_DETECTOR_POINT,
) -> dict[str, Any]:
    """Runs one (rr_ratio, min_sl_distance) combination against df, with every
    detector/confluence parameter held fixed at detector_point. Row shape mirrors
    _run_point()'s (plus the two risk fields) so the same reporting helpers apply."""
    aggregator = SMCSignalAggregator(min_confluence=detector_point.min_confluence)
    risk = RiskManager(
        initial_balance=INITIAL_BALANCE, skip_if_capital_capped=True,
        rr_ratio=point.rr_ratio, min_sl_distance=point.min_sl_distance,
    )
    result = run_backtest(
        df, aggregator, risk, FillConfig(),
        min_ob_body_ratio=detector_point.min_ob_body_ratio,
        min_fvg_gap_ratio=detector_point.min_fvg_gap_ratio,
        min_trendline_break_distance=detector_point.min_trendline_break_distance,
        min_channel_break_distance=detector_point.min_channel_break_distance,
        max_trap_retest_distance=detector_point.max_trap_retest_distance,
    )
    return {
        "rr_ratio": point.rr_ratio,
        "min_sl_distance": point.min_sl_distance,
        **result.metrics(),
    }


def _init_risk_worker(snapshot_path: Path, start: str, end: str) -> None:
    global _worker_df
    full = load_snapshot(snapshot_path)
    _worker_df = full.loc[start:end]


def _run_risk_point_in_worker(point: RiskGridPoint) -> dict[str, Any]:
    return _run_risk_point(point, _worker_df)


def build_risk_grid() -> list[RiskGridPoint]:
    return [RiskGridPoint(rr, sl) for rr, sl in itertools.product(RR_RATIO_GRID, MIN_SL_DISTANCE_GRID)]


def run_risk_search(
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    start: str = TUNING_START,
    end: str = TUNING_END,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Runs the small (rr_ratio x min_sl_distance) grid against [start, end) of
    snapshot_path, detector/confluence parameters fixed at BEST_KNOWN_DETECTOR_POINT.
    Returns every combination's result row - the full plateau, not a top-N cut."""
    grid = build_risk_grid()
    print(
        f"Running {len(grid)} risk grid points over {start} -> {end} "
        f"(detector params fixed at {BEST_KNOWN_DETECTOR_POINT}) ..."
    )
    results: list[dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_risk_worker, initargs=(snapshot_path, start, end),
    ) as pool:
        futures = [pool.submit(_run_risk_point_in_worker, point) for point in grid]
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results.append(row)
            elapsed = time.time() - t0
            print(
                f"[{i:>2}/{len(grid)}] {elapsed:6.1f}s  rr={row['rr_ratio']} sl={row['min_sl_distance']:<6} "
                f"-> trades={row['total_trades']:>4} roi={row['roi_pct']:>7.2f}% pf={row['profit_factor']:.2f}"
            )
    print(f"Done in {time.time() - t0:.1f}s.")
    return results


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
    trap = row["max_trap_retest_distance"]
    trap_cell = "inf" if trap == float("inf") else f"{trap:.1f}"
    return (
        f"| {row['min_confluence']} | {row['min_ob_body_ratio']:.2f} | {row['min_fvg_gap_ratio']:.2f} | "
        f"{row['min_trendline_break_distance']:.1f} | {row['min_channel_break_distance']:.1f} | {trap_cell} | "
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
        "Swept: `min_confluence` (1-5, the aggregator's full 5-vote range), `min_ob_body_ratio`, "
        "`min_fvg_gap_ratio`, `min_trendline_break_distance`, `min_channel_break_distance`, "
        "`max_trap_retest_distance` - 6 parameters, this is the 2026-09 rebuild's grid (order block/FVG/"
        "trendline/channel/fakeout-trap detectors, 5-vote aggregator), not the pre-rebuild one.",
        "",
        "- **`min_break_distance` is deliberately excluded from this grid - not deprioritized, removed.** "
        "It gates `detect_bos_choch`'s close-crossing threshold, and a direct code trace confirms that "
        "detector's output (`structure_bias`/`structure_events`) has no causal path to any backtest "
        "result: `signals/smc_aggregator.py`'s 5-vote model (order block/FVG/trendline/channel/"
        "fakeout_trap) never reads it, and `backtest/runner.py`'s stop-loss placement reads only "
        "`order_blocks`. Sweeping it could not have changed a single result in this table - every "
        "combination below implicitly ran with `min_break_distance` at its unswept default (`0.0`).",
        "- **`trendline_points` is fixed at its default (`3`), not swept.** It's structural (changes what "
        "points the trendline/channel fits are computed from) rather than a filter threshold - same "
        "category as `swing_left_bars`/`swing_right_bars`, also fixed. See `backtest/tune.py`'s module "
        "docstring.",
        f"- Combinations with fewer than {LOW_SAMPLE_TRADE_THRESHOLD} trades are flagged `†` and should be read "
        "as low-sample, not as a genuine result - a handful of trades over 9 months of a sparse-signal "
        "strategy is not enough to distinguish structure from chance.",
        f"- {len(low_sample)}/{len(results)} combinations are low-sample (`†`); {len(zero_trade)} produced zero trades.",
        f"- Trade count across the grid: min {min(trade_counts)}, median {statistics.median(trade_counts):.0f}, "
        f"max {max(trade_counts)}.",
        "",
        f"| Conf | OB ratio | FVG ratio | TL brk$ | Ch brk$ | Trap$ | {header_cells} | † |",
        f"|---|---|---|---|---|---|{sep_cells}|---|",
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
# run) the risk-parameter grid, then (if it's ever been run) holdout evaluations.
# Every writer below preserves whichever of these sections it isn't itself
# responsible for, found by these markers - never by slicing a formatted row
# string (see the "Caught and fixed a real bug" note in this project's git
# history for why that's specifically disallowed here).
_DIAGNOSTIC_MARKER = "\n## Diagnostic: long+short"
_RISK_GRID_MARKER = "\n## Risk-parameter grid"
_HOLDOUT_MARKER = "\n## Holdout evaluations"


def _find_earliest_marker(text: str, markers: list[str], start: int = 0) -> int:
    """Position of whichever marker appears first in text at or after start, or
    len(text) if none do."""
    positions = [p for p in (text.find(m, start) for m in markers) if p != -1]
    return min(positions) if positions else len(text)


def write_tuning_log(results: list[dict[str, Any]], start: str, end: str, path: Path = TUNING_LOG_PATH) -> None:
    """Writes the grid search section fresh, but preserves any existing diagnostic,
    risk-grid, and/or holdout sections already in the file (from prior
    --diagnostic-allow-short, run_risk_search, or --evaluate-holdout runs, in any
    combination) - re-running the search must not erase any of them."""
    preserved_tail = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        cut = _find_earliest_marker(existing, [_DIAGNOSTIC_MARKER, _RISK_GRID_MARKER, _HOLDOUT_MARKER])
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
    key = lambda r: (
        r["min_confluence"], r["min_ob_body_ratio"], r["min_fvg_gap_ratio"],
        r["min_trendline_break_distance"], r["min_channel_break_distance"], r["max_trap_retest_distance"],
    )
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
    grid-search section before it and the risk-grid/holdout sections after it (in
    any presence/absence combination) - never called automatically, only from the
    --diagnostic-allow-short CLI path."""
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Parameter tuning log\n"

    diag_start = existing.find(_DIAGNOSTIC_MARKER)
    if diag_start != -1:
        # Replace a prior diagnostic run rather than duplicating it.
        after_diag = _find_earliest_marker(existing, [_RISK_GRID_MARKER, _HOLDOUT_MARKER], start=diag_start + 1)
        head = existing[:diag_start]
        tail = existing[after_diag:]
    else:
        insert_before = _find_earliest_marker(existing, [_RISK_GRID_MARKER, _HOLDOUT_MARKER])
        head = existing[:insert_before]
        tail = existing[insert_before:]

    section = _diagnostic_short_section_text(long_only, long_short, start, end)
    path.write_text(head.rstrip("\n") + "\n\n" + section + "\n\n" + tail.lstrip("\n"), encoding="utf-8")
    print(f"Wrote diagnostic long+short section to {path}")


def _risk_grid_section_text(results: list[dict[str, Any]], start: str, end: str) -> str:
    ranked = sorted(results, key=lambda r: r["roi_pct"], reverse=True)
    low_sample = [r for r in results if r["total_trades"] < LOW_SAMPLE_TRADE_THRESHOLD]
    p = BEST_KNOWN_DETECTOR_POINT
    header_cells = " | ".join(label for _, label, _ in _METRIC_COLUMNS)
    sep_cells = " | ".join("---" for _ in _METRIC_COLUMNS)

    lines = [
        "## Risk-parameter grid: rr_ratio x min_sl_distance (targeted follow-up)",
        "",
        f"Tuning period: **{start} -> {end}**. {len(results)} combinations "
        f"({len(RR_RATIO_GRID)} rr_ratio x {len(MIN_SL_DISTANCE_GRID)} min_sl_distance values), sorted by "
        "ROI (highest first) - full plateau, not a top-N cut. Kept deliberately separate from the "
        "6-parameter detector grid above rather than swept jointly with it - this checks a different, "
        "orthogonal lever (risk management, not signal quality).",
        "",
        "Every combination below holds detector/confluence parameters fixed at the best non-low-sample "
        f"result from the detector grid: `min_confluence={p.min_confluence}`, "
        f"`min_ob_body_ratio={p.min_ob_body_ratio}`, `min_fvg_gap_ratio={p.min_fvg_gap_ratio}`, "
        f"`min_trendline_break_distance={p.min_trendline_break_distance}`, "
        f"`min_channel_break_distance={p.min_channel_break_distance}`, "
        f"`max_trap_retest_distance={p.max_trap_retest_distance}` (`skip_if_capital_capped=True`, same "
        "fixed sizing behavior as the detector grid).",
        "",
        f"- Combinations with fewer than {LOW_SAMPLE_TRADE_THRESHOLD} trades are flagged `†` (low-sample).",
        f"- {len(low_sample)}/{len(results)} combinations are low-sample.",
        "",
        f"| RR ratio | Min SL$ | {header_cells} | † |",
        f"|---|---|{sep_cells}|---|",
    ]
    for row in ranked:
        flag = "†" if row["total_trades"] < LOW_SAMPLE_TRADE_THRESHOLD else " "
        lines.append(f"| {row['rr_ratio']:.1f} | {row['min_sl_distance']:.1f} | {_metric_cells(row)} | {flag} |")

    # Computed, not hardcoded, so this stays accurate if the grid is ever re-run
    # with different values/results - which lever actually moved ROI is a
    # data-dependent finding, not a fixed claim about this codebase.
    by_rr: dict[float, list[float]] = {}
    for row in results:
        by_rr.setdefault(row["rr_ratio"], []).append(row["roi_pct"])
    rr_spread = max(statistics.mean(v) for v in by_rr.values()) - min(statistics.mean(v) for v in by_rr.values())
    max_sl_spread_within_rr = max((max(v) - min(v)) for v in by_rr.values())
    best_rr = max(by_rr, key=lambda k: statistics.mean(by_rr[k]))
    worst_rr = min(by_rr, key=lambda k: statistics.mean(by_rr[k]))

    lines.append("")
    lines.append(
        f"**Reading this table**: holding min_sl_distance fixed and varying rr_ratio moves ROI by "
        f"{rr_spread:.2f} points (worst rr_ratio={worst_rr} -> best rr_ratio={best_rr}); holding rr_ratio "
        f"fixed and varying min_sl_distance moves ROI by at most {max_sl_spread_within_rr:.2f} points "
        f"within any single rr_ratio. "
        + (
            "min_sl_distance had zero measured effect here - every value tested (0 through the largest in "
            "MIN_SL_DISTANCE_GRID) produced identical trades/ROI for a given rr_ratio, meaning every "
            "realized stop distance in this combination already exceeds the largest floor tested. At "
            "BTC's current price scale, `_structural_stop_loss` (backtest/runner.py) places stops via "
            "order-block levels or a 1% price buffer - roughly $750-$1,260 across this period's $75k-$126k "
            "range - so a floor in the tens-to-low-hundreds of dollars can't bind. A future round anchored "
            "on that actual scale (percent-of-price or a much larger dollar floor) would be a fairer test "
            "of this parameter than this grid's values; rr_ratio is unambiguously the lever that matters "
            "at the values tested here."
            if max_sl_spread_within_rr < 0.01
            else "Both levers show a measurable effect here - see the table above for the actual shape."
        )
    )
    return "\n".join(lines)


def write_risk_grid_section(
    results: list[dict[str, Any]], start: str, end: str, path: Path = TUNING_LOG_PATH,
) -> None:
    """Writes/replaces the '## Risk-parameter grid' section, preserving the grid-
    search/diagnostic sections before it and the holdout section after it (in any
    presence/absence combination) - never called automatically, only from
    run_risk_search()'s CLI path."""
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Parameter tuning log\n"

    risk_start = existing.find(_RISK_GRID_MARKER)
    if risk_start != -1:
        # Replace a prior risk-grid run rather than duplicating it.
        after_risk = _find_earliest_marker(existing, [_HOLDOUT_MARKER], start=risk_start + 1)
        head = existing[:risk_start]
        tail = existing[after_risk:]
    else:
        insert_before = _find_earliest_marker(existing, [_HOLDOUT_MARKER])
        head = existing[:insert_before]
        tail = existing[insert_before:]

    section = _risk_grid_section_text(results, start, end)
    path.write_text(head.rstrip("\n") + "\n\n" + section + "\n\n" + tail.lstrip("\n"), encoding="utf-8")
    print(f"Wrote risk-parameter grid section to {path}")


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
        f"`min_fvg_gap_ratio={row['min_fvg_gap_ratio']}`, "
        f"`min_trendline_break_distance={row['min_trendline_break_distance']}`, "
        f"`min_channel_break_distance={row['min_channel_break_distance']}`, "
        f"`max_trap_retest_distance={row['max_trap_retest_distance']}`. "
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
             "Requires --min-confluence/--min-ob-body-ratio/--min-fvg-gap-ratio/"
             "--min-trendline-break-distance/--min-channel-break-distance/--max-trap-retest-distance.",
    )
    parser.add_argument(
        "--diagnostic-allow-short", action="store_true",
        help="Re-run the full grid twice (long-only and long+short) over the tuning period and write a "
             "comparison to docs/tuning-log.md. DIAGNOSTIC ONLY - Kraken Spot cannot execute short "
             "positions; this never affects live/trader.py or paper trading. See "
             "run_diagnostic_long_short()'s docstring.",
    )
    parser.add_argument(
        "--risk-grid", action="store_true",
        help="Run the small, separate rr_ratio x min_sl_distance grid (detector/confluence params held "
             "fixed at BEST_KNOWN_DETECTOR_POINT) instead of the main detector grid, and write its own "
             "section to docs/tuning-log.md. See run_risk_search()'s docstring.",
    )
    parser.add_argument("--min-confluence", type=int)
    parser.add_argument("--min-ob-body-ratio", type=float)
    parser.add_argument("--min-fvg-gap-ratio", type=float)
    parser.add_argument("--min-trendline-break-distance", type=float)
    parser.add_argument("--min-channel-break-distance", type=float)
    parser.add_argument("--max-trap-retest-distance", type=float, help="Accepts 'inf' for no filter.")
    args = parser.parse_args()

    if args.evaluate_holdout:
        missing = [
            name for name, val in [
                ("--min-confluence", args.min_confluence), ("--min-ob-body-ratio", args.min_ob_body_ratio),
                ("--min-fvg-gap-ratio", args.min_fvg_gap_ratio),
                ("--min-trendline-break-distance", args.min_trendline_break_distance),
                ("--min-channel-break-distance", args.min_channel_break_distance),
                ("--max-trap-retest-distance", args.max_trap_retest_distance),
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
            args.min_trendline_break_distance, args.min_channel_break_distance, args.max_trap_retest_distance,
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

    if args.risk_grid:
        print(
            "Running the risk-parameter grid (rr_ratio x min_sl_distance) - detector/confluence params "
            "held fixed. See run_risk_search()'s docstring."
        )
        results = run_risk_search(snapshot_path=args.snapshot, max_workers=args.max_workers)
        write_risk_grid_section(results, TUNING_START, TUNING_END)
        return

    results = run_search(snapshot_path=args.snapshot, max_workers=args.max_workers)
    write_tuning_log(results, TUNING_START, TUNING_END)


if __name__ == "__main__":
    main()
