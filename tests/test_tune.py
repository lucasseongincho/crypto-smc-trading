"""
test_tune.py
Tests backtest/tune.py's grid construction, single-point execution, and markdown
report generation against small synthetic data - not the real multi-year snapshot
(too slow for a unit test; see the module's own docstring for real timing).
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from backtest.tune import (
    MAX_TRAP_RETEST_DISTANCE_GRID,
    MIN_CHANNEL_BREAK_DISTANCE_GRID,
    MIN_CONFLUENCE_GRID,
    MIN_FVG_GAP_RATIO_GRID,
    MIN_OB_BODY_RATIO_GRID,
    MIN_SL_DISTANCE_GRID,
    MIN_TRENDLINE_BREAK_DISTANCE_GRID,
    RR_RATIO_GRID,
    BEST_KNOWN_DETECTOR_POINT,
    GridPoint,
    RiskGridPoint,
    _run_point,
    _run_risk_point,
    append_holdout_result,
    build_grid,
    build_risk_grid,
    write_diagnostic_short_section,
    write_risk_grid_section,
    write_tuning_log,
)


def _synthetic_ohlc(n: int = 60) -> pd.DataFrame:
    """A gently trending synthetic series - enough bars for the detectors to have
    something to find, without needing real market data for a mechanics test."""
    rows = []
    price = 50_000.0
    for i in range(n):
        price += 10.0 if i % 3 else -5.0
        rows.append((price, price + 20, price - 20, price + 5))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.index = pd.date_range("2025-01-01", periods=n, freq="5min", tz="utc")
    return df


def _sample_row(**overrides) -> dict:
    row = {
        "min_confluence": 3, "min_ob_body_ratio": 0.5, "min_fvg_gap_ratio": 0.5,
        "min_trendline_break_distance": 50.0, "min_channel_break_distance": 50.0,
        "max_trap_retest_distance": 100.0,
        "total_trades": 12, "long_trades": 12, "short_trades": 0, "win_rate_pct": 50.0, "profit_factor": 1.5,
        "total_pnl": 200.0, "roi_pct": 1.0, "max_drawdown_pct": 3.0, "avg_r_multiple": 0.4,
        "median_r_multiple": 0.3, "std_r_multiple": 0.2, "buy_hold_return_pct": 2.0,
    }
    row.update(overrides)
    return row


def _sample_risk_row(**overrides) -> dict:
    row = {
        "rr_ratio": 1.5, "min_sl_distance": 50.0,
        "total_trades": 12, "long_trades": 12, "short_trades": 0, "win_rate_pct": 50.0, "profit_factor": 1.5,
        "total_pnl": 200.0, "roi_pct": 1.0, "max_drawdown_pct": 3.0, "avg_r_multiple": 0.4,
        "median_r_multiple": 0.3, "std_r_multiple": 0.2, "buy_hold_return_pct": 2.0,
    }
    row.update(overrides)
    return row


class TestBuildGrid(unittest.TestCase):
    def test_grid_size_matches_the_product_of_all_six_parameter_lists(self):
        grid = build_grid()
        expected = (
            len(MIN_CONFLUENCE_GRID) * len(MIN_OB_BODY_RATIO_GRID) * len(MIN_FVG_GAP_RATIO_GRID)
            * len(MIN_TRENDLINE_BREAK_DISTANCE_GRID) * len(MIN_CHANNEL_BREAK_DISTANCE_GRID)
            * len(MAX_TRAP_RETEST_DISTANCE_GRID)
        )
        self.assertEqual(len(grid), expected)
        self.assertEqual(len(grid), 1215)  # 5 confluence values x 3 values each for the other 5 params

    def test_grid_has_no_duplicate_points(self):
        grid = build_grid()
        as_tuples = {
            (p.min_confluence, p.min_ob_body_ratio, p.min_fvg_gap_ratio, p.min_trendline_break_distance,
             p.min_channel_break_distance, p.max_trap_retest_distance)
            for p in grid
        }
        self.assertEqual(len(as_tuples), len(grid))

    def test_every_grid_value_is_within_its_declared_list(self):
        grid = build_grid()
        for p in grid:
            self.assertIn(p.min_confluence, MIN_CONFLUENCE_GRID)
            self.assertIn(p.min_ob_body_ratio, MIN_OB_BODY_RATIO_GRID)
            self.assertIn(p.min_fvg_gap_ratio, MIN_FVG_GAP_RATIO_GRID)
            self.assertIn(p.min_trendline_break_distance, MIN_TRENDLINE_BREAK_DISTANCE_GRID)
            self.assertIn(p.min_channel_break_distance, MIN_CHANNEL_BREAK_DISTANCE_GRID)
            self.assertIn(p.max_trap_retest_distance, MAX_TRAP_RETEST_DISTANCE_GRID)

    def test_min_confluence_grid_spans_the_full_five_vote_range(self):
        # Phase 5 rebuild: 5 votes, confluence range -5..+5 - the grid must cover
        # all 5 achievable strictness levels, not a coarser subset.
        self.assertEqual(MIN_CONFLUENCE_GRID, [1, 2, 3, 4, 5])

    def test_min_break_distance_is_not_a_grid_field(self):
        # Removed outright, not deprioritized - see backtest/tune.py's module
        # docstring for the code-trace confirming it has no causal path to any
        # backtest output (signals/smc_aggregator.py never reads structure_bias/
        # structure_events, backtest/runner.py's stop-loss reads only order_blocks).
        self.assertNotIn("min_break_distance", GridPoint.__dataclass_fields__)

    def test_trendline_points_is_not_a_grid_field(self):
        # Fixed at its default, not swept - structural (changes what the fit
        # computes from), same category as swing_left_bars/swing_right_bars.
        self.assertNotIn("trendline_points", GridPoint.__dataclass_fields__)

    def test_max_trap_retest_distance_grid_includes_the_off_default(self):
        # DEFAULT_MAX_TRAP_RETEST_DISTANCE is float("inf") - the grid's "off"
        # baseline must reproduce that, same role 0.0 plays for the minimum-
        # distance filters.
        self.assertIn(float("inf"), MAX_TRAP_RETEST_DISTANCE_GRID)


class TestRunPoint(unittest.TestCase):
    def test_returns_params_and_full_trade_level_metrics(self):
        df = _synthetic_ohlc()
        point = GridPoint(
            min_confluence=1, min_ob_body_ratio=0.0, min_fvg_gap_ratio=0.0,
            min_trendline_break_distance=0.0, min_channel_break_distance=0.0,
            max_trap_retest_distance=float("inf"),
        )

        row = _run_point(point, df)

        self.assertEqual(row["min_confluence"], 1)
        self.assertEqual(row["min_ob_body_ratio"], 0.0)
        self.assertEqual(row["min_trendline_break_distance"], 0.0)
        self.assertEqual(row["min_channel_break_distance"], 0.0)
        self.assertEqual(row["max_trap_retest_distance"], float("inf"))
        self.assertNotIn("min_break_distance", row)
        for key in ["total_trades", "win_rate_pct", "profit_factor", "roi_pct", "avg_r_multiple", "median_r_multiple"]:
            self.assertIn(key, row)

    def test_stricter_confluence_never_produces_more_trades_than_looser_on_the_same_data(self):
        df = _synthetic_ohlc(n=90)
        loose = _run_point(GridPoint(1, 0.0, 0.0, 0.0, 0.0, float("inf")), df)
        strict = _run_point(GridPoint(5, 0.0, 0.0, 0.0, 0.0, float("inf")), df)
        self.assertLessEqual(strict["total_trades"], loose["total_trades"])


class TestWriteTuningLog(unittest.TestCase):
    def test_writes_a_markdown_table_with_every_result_row(self):
        results = [_sample_row(min_confluence=c) for c in [1, 2, 3, 4, 5]]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log(results, "2025-04-01", "2025-12-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertIn("Grid search results", content)
        for c in [1, 2, 3, 4, 5]:
            self.assertEqual(content.count(f"| {c} | 0.50 | 0.50 |"), 1)  # one row per result, not just a top-N cut

    def test_low_sample_rows_are_flagged(self):
        results = [_sample_row(min_confluence=5, total_trades=2, buy_hold_return_pct=None)]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log(results, "2025-04-01", "2025-12-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertIn("†", content)

    def test_rerunning_the_search_preserves_an_existing_holdout_section(self):
        row = _sample_row()
        results = [row]

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log(results, "2025-04-01", "2025-12-31", path=path)
            append_holdout_result(row, "2026-01-01", "2026-03-31", path=path)
            self.assertIn("Holdout evaluations", path.read_text(encoding="utf-8"))

            # Re-running the search must not erase the holdout section written above.
            write_tuning_log(results, "2025-04-01", "2025-12-31", path=path)
            content_after = path.read_text(encoding="utf-8")

        self.assertIn("Holdout evaluations", content_after)
        self.assertIn("Grid search results", content_after)


class TestDiagnosticShortSectionOrderingIsPreservedBothWays(unittest.TestCase):
    """The log has up to three sections that can each be (re)written independently
    and in any order the CLI happens to be used: grid search, diagnostic
    long+short, holdout evaluations. Each writer must preserve the sections it
    isn't responsible for, regardless of which order they were created in - this
    is exactly the class of bug a prior string-slicing mistake in this same file
    already produced once, so it's covered directly rather than just by "does the
    word appear somewhere" checks."""

    def test_diagnostic_written_after_holdout_is_inserted_before_it_not_after(self):
        long_only = [_sample_row(min_confluence=3)]
        long_short = [_sample_row(min_confluence=3, roi_pct=-2.0, short_trades=4, total_trades=16, long_trades=12)]

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log(long_only, "2025-04-01", "2025-12-31", path=path)
            append_holdout_result(_sample_row(), "2026-01-01", "2026-03-31", path=path)
            write_diagnostic_short_section(long_only, long_short, "2025-04-01", "2025-12-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertIn("Grid search results", content)
        self.assertIn("Diagnostic: long+short", content)
        self.assertIn("Holdout evaluations", content)
        # Canonical order: grid, then diagnostic, then holdout - regardless of the
        # order the CLI commands were actually run in.
        self.assertLess(content.index("Grid search results"), content.index("Diagnostic: long+short"))
        self.assertLess(content.index("Diagnostic: long+short"), content.index("Holdout evaluations"))

    def test_rerunning_diagnostic_replaces_the_old_one_without_duplicating(self):
        long_only = [_sample_row(min_confluence=3)]
        long_short_v1 = [_sample_row(min_confluence=3, roi_pct=-2.0, short_trades=4, total_trades=16)]
        long_short_v2 = [_sample_row(min_confluence=3, roi_pct=-9.0, short_trades=9, total_trades=21)]

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log(long_only, "2025-04-01", "2025-12-31", path=path)
            append_holdout_result(_sample_row(), "2026-01-01", "2026-03-31", path=path)
            write_diagnostic_short_section(long_only, long_short_v1, "2025-04-01", "2025-12-31", path=path)
            write_diagnostic_short_section(long_only, long_short_v2, "2025-04-01", "2025-12-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertEqual(content.count("Diagnostic: long+short"), 1)  # not duplicated
        self.assertIn("-9.00", content)  # the second (latest) run's numbers
        self.assertIn("Holdout evaluations", content)  # still preserved

    def test_rerunning_the_grid_search_preserves_both_diagnostic_and_holdout(self):
        long_only = [_sample_row(min_confluence=3)]
        long_short = [_sample_row(min_confluence=3, roi_pct=-2.0, short_trades=4, total_trades=16)]

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log(long_only, "2025-04-01", "2025-12-31", path=path)
            write_diagnostic_short_section(long_only, long_short, "2025-04-01", "2025-12-31", path=path)
            append_holdout_result(_sample_row(), "2026-01-01", "2026-03-31", path=path)

            write_tuning_log(long_only, "2025-04-01", "2025-12-31", path=path)  # re-run the search
            content = path.read_text(encoding="utf-8")

        self.assertIn("Grid search results", content)
        self.assertIn("Diagnostic: long+short", content)
        self.assertIn("Holdout evaluations", content)


class TestBuildRiskGrid(unittest.TestCase):
    def test_grid_size_matches_the_product_of_both_risk_parameter_lists(self):
        self.assertEqual(len(build_risk_grid()), len(RR_RATIO_GRID) * len(MIN_SL_DISTANCE_GRID))

    def test_grid_size_is_16(self):
        # 4 rr_ratio values x 4 min_sl_distance values, per the module docstring's
        # "3-4 values each" - locks the literal count so a silent grid-size change
        # is caught here, not just via the product-matches-lists check above.
        self.assertEqual(len(build_risk_grid()), 16)

    def test_rr_ratio_grid_includes_the_shipped_default(self):
        self.assertIn(1.5, RR_RATIO_GRID)

    def test_min_sl_distance_grid_includes_off_and_the_shipped_default(self):
        self.assertIn(0.0, MIN_SL_DISTANCE_GRID)
        self.assertIn(50.0, MIN_SL_DISTANCE_GRID)

    def test_every_combination_is_unique(self):
        grid = build_risk_grid()
        self.assertEqual(len(grid), len(set(grid)))


class TestRunRiskPoint(unittest.TestCase):
    def test_returns_risk_params_and_full_trade_level_metrics(self):
        df = _synthetic_ohlc()
        point = RiskGridPoint(rr_ratio=2.0, min_sl_distance=0.0)

        row = _run_risk_point(point, df)

        self.assertEqual(row["rr_ratio"], 2.0)
        self.assertEqual(row["min_sl_distance"], 0.0)
        for key in ["total_trades", "win_rate_pct", "profit_factor", "roi_pct", "avg_r_multiple"]:
            self.assertIn(key, row)
        # Detector/confluence params are never in the row - they're held fixed at
        # BEST_KNOWN_DETECTOR_POINT, not swept, so they have nothing to report per-row.
        self.assertNotIn("min_confluence", row)

    def test_holds_detector_params_fixed_at_best_known_detector_point(self):
        """A tighter min_sl_distance can only reject trades (never invent new
        ones) relative to a looser one, on the same fixed detector params and
        data - proves min_sl_distance actually reaches calculate_size(), not just
        that RiskManager accepts and ignores it."""
        df = _synthetic_ohlc(n=90)
        loose = _run_risk_point(RiskGridPoint(rr_ratio=1.5, min_sl_distance=0.0), df)
        strict = _run_risk_point(RiskGridPoint(rr_ratio=1.5, min_sl_distance=1_000_000.0), df)
        self.assertLessEqual(strict["total_trades"], loose["total_trades"])
        self.assertEqual(strict["total_trades"], 0)  # no real stop distance clears a $1M floor


class TestRiskGridSectionOrderingIsPreservedBothWays(unittest.TestCase):
    """Mirrors TestDiagnosticShortSectionOrderingIsPreservedBothWays - the risk
    grid is a fourth section that can be (re)written in any order relative to the
    other three, and must never eat a section it isn't responsible for."""

    def test_risk_grid_written_after_holdout_is_inserted_before_it_not_after(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log([_sample_row()], "2025-04-01", "2025-12-31", path=path)
            append_holdout_result(_sample_row(), "2026-01-01", "2026-03-31", path=path)
            write_risk_grid_section([_sample_risk_row()], "2025-04-01", "2025-12-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertIn("Grid search results", content)
        self.assertIn("Risk-parameter grid", content)
        self.assertIn("Holdout evaluations", content)
        self.assertLess(content.index("Grid search results"), content.index("Risk-parameter grid"))
        self.assertLess(content.index("Risk-parameter grid"), content.index("Holdout evaluations"))

    def test_risk_grid_written_after_diagnostic_lands_between_diagnostic_and_holdout(self):
        long_only = [_sample_row(min_confluence=3)]
        long_short = [_sample_row(min_confluence=3, roi_pct=-2.0, short_trades=4, total_trades=16)]

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log(long_only, "2025-04-01", "2025-12-31", path=path)
            write_diagnostic_short_section(long_only, long_short, "2025-04-01", "2025-12-31", path=path)
            append_holdout_result(_sample_row(), "2026-01-01", "2026-03-31", path=path)
            write_risk_grid_section([_sample_risk_row()], "2025-04-01", "2025-12-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertLess(content.index("Diagnostic: long+short"), content.index("Risk-parameter grid"))
        self.assertLess(content.index("Risk-parameter grid"), content.index("Holdout evaluations"))

    def test_rerunning_risk_grid_replaces_the_old_one_without_duplicating(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log([_sample_row()], "2025-04-01", "2025-12-31", path=path)
            append_holdout_result(_sample_row(), "2026-01-01", "2026-03-31", path=path)
            write_risk_grid_section([_sample_risk_row(roi_pct=-2.0)], "2025-04-01", "2025-12-31", path=path)
            write_risk_grid_section([_sample_risk_row(roi_pct=-9.0)], "2025-04-01", "2025-12-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertEqual(content.count("Risk-parameter grid"), 1)
        self.assertIn("-9.00", content)
        self.assertIn("Holdout evaluations", content)

    def test_rerunning_the_grid_search_preserves_the_risk_grid_section(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log([_sample_row()], "2025-04-01", "2025-12-31", path=path)
            write_risk_grid_section([_sample_risk_row()], "2025-04-01", "2025-12-31", path=path)
            append_holdout_result(_sample_row(), "2026-01-01", "2026-03-31", path=path)

            write_tuning_log([_sample_row()], "2025-04-01", "2025-12-31", path=path)  # re-run the search
            content = path.read_text(encoding="utf-8")

        self.assertIn("Grid search results", content)
        self.assertIn("Risk-parameter grid", content)
        self.assertIn("Holdout evaluations", content)


class TestAppendHoldoutResult(unittest.TestCase):
    def test_creates_the_file_and_section_if_neither_exists(self):
        row = _sample_row(
            min_confluence=2, min_ob_body_ratio=1.0, min_fvg_gap_ratio=1.0,
            total_trades=8, win_rate_pct=37.5, profit_factor=0.9, total_pnl=-50.0, roi_pct=-0.25,
            max_drawdown_pct=1.5, avg_r_multiple=-0.1, median_r_multiple=-0.05, std_r_multiple=0.3,
            buy_hold_return_pct=0.4,
        )

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            self.assertFalse(path.exists())
            append_holdout_result(row, "2026-01-01", "2026-03-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertIn("Holdout evaluations", content)
        self.assertIn("min_confluence=2", content)
        self.assertIn("min_trendline_break_distance=50.0", content)
        self.assertIn("min_channel_break_distance=50.0", content)
        self.assertIn("max_trap_retest_distance=100.0", content)
        # The metric table itself must actually render, not just the params prose -
        # this specifically catches a prior bug where the table was built by
        # string-slicing _format_row()'s output and silently produced garbage.
        self.assertIn("| 8 | 37.5 | 0.90 | -50 | -0.25 | 1.50 | -0.10 | -0.05 | 0.30 |", content)

    def test_appending_twice_keeps_both_entries(self):
        row = _sample_row(min_confluence=2)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            append_holdout_result(row, "2026-01-01", "2026-03-31", path=path)
            append_holdout_result({**row, "min_confluence": 3}, "2026-01-01", "2026-03-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertIn("min_confluence=2", content)
        self.assertIn("min_confluence=3", content)


if __name__ == "__main__":
    unittest.main()
