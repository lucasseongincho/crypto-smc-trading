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
    MIN_CONFLUENCE_GRID,
    MIN_FVG_GAP_RATIO_GRID,
    MIN_OB_BODY_RATIO_GRID,
    GridPoint,
    _run_point,
    append_holdout_result,
    build_grid,
    write_diagnostic_short_section,
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


class TestBuildGrid(unittest.TestCase):
    def test_grid_size_matches_the_product_of_all_three_parameter_lists(self):
        grid = build_grid()
        expected = len(MIN_CONFLUENCE_GRID) * len(MIN_OB_BODY_RATIO_GRID) * len(MIN_FVG_GAP_RATIO_GRID)
        self.assertEqual(len(grid), expected)
        self.assertEqual(len(grid), 64)  # 4 values each, as specified

    def test_grid_has_no_duplicate_points(self):
        grid = build_grid()
        as_tuples = {(p.min_confluence, p.min_ob_body_ratio, p.min_fvg_gap_ratio) for p in grid}
        self.assertEqual(len(as_tuples), len(grid))

    def test_every_grid_value_is_within_its_declared_list(self):
        grid = build_grid()
        for p in grid:
            self.assertIn(p.min_confluence, MIN_CONFLUENCE_GRID)
            self.assertIn(p.min_ob_body_ratio, MIN_OB_BODY_RATIO_GRID)
            self.assertIn(p.min_fvg_gap_ratio, MIN_FVG_GAP_RATIO_GRID)

    def test_min_break_distance_is_not_a_grid_field(self):
        # Removed outright, not deprioritized - see backtest/tune.py's module
        # docstring for the code-trace confirming it has no causal path to any
        # backtest output (signals/smc_aggregator.py never reads structure_bias/
        # structure_events, backtest/runner.py's stop-loss reads only order_blocks).
        self.assertNotIn("min_break_distance", GridPoint.__dataclass_fields__)


class TestRunPoint(unittest.TestCase):
    def test_returns_params_and_full_trade_level_metrics(self):
        df = _synthetic_ohlc()
        point = GridPoint(min_confluence=1, min_ob_body_ratio=0.0, min_fvg_gap_ratio=0.0)

        row = _run_point(point, df)

        self.assertEqual(row["min_confluence"], 1)
        self.assertEqual(row["min_ob_body_ratio"], 0.0)
        self.assertNotIn("min_break_distance", row)
        for key in ["total_trades", "win_rate_pct", "profit_factor", "roi_pct", "avg_r_multiple", "median_r_multiple"]:
            self.assertIn(key, row)

    def test_stricter_confluence_never_produces_more_trades_than_looser_on_the_same_data(self):
        df = _synthetic_ohlc(n=90)
        loose = _run_point(GridPoint(1, 0.0, 0.0), df)
        strict = _run_point(GridPoint(4, 0.0, 0.0), df)
        self.assertLessEqual(strict["total_trades"], loose["total_trades"])


class TestWriteTuningLog(unittest.TestCase):
    def test_writes_a_markdown_table_with_every_result_row(self):
        results = [
            {"min_confluence": c, "min_ob_body_ratio": 0.0, "min_fvg_gap_ratio": 0.0,
             "total_trades": 5, "win_rate_pct": 40.0, "profit_factor": 1.2, "total_pnl": 100.0, "roi_pct": 0.5,
             "max_drawdown_pct": 2.0, "avg_r_multiple": 0.3, "median_r_multiple": 0.2, "std_r_multiple": 0.1,
             "buy_hold_return_pct": 1.0}
            for c in [1, 2, 3, 4]
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log(results, "2025-04-01", "2025-12-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertIn("Grid search results", content)
        self.assertEqual(content.count("| 1 | 0.00 |"), 1)  # one row per result, not just a top-N cut
        self.assertEqual(content.count("| 2 | 0.00 |"), 1)
        self.assertEqual(content.count("| 3 | 0.00 |"), 1)
        self.assertEqual(content.count("| 4 | 0.00 |"), 1)

    def test_low_sample_rows_are_flagged(self):
        results = [
            {"min_confluence": 4, "min_ob_body_ratio": 2.0, "min_fvg_gap_ratio": 2.0,
             "total_trades": 2, "win_rate_pct": 0.0, "profit_factor": 0.0, "total_pnl": 0.0, "roi_pct": 0.0,
             "max_drawdown_pct": 0.0, "avg_r_multiple": 0.0, "median_r_multiple": 0.0, "std_r_multiple": 0.0,
             "buy_hold_return_pct": None},
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            write_tuning_log(results, "2025-04-01", "2025-12-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertIn("†", content)

    def test_rerunning_the_search_preserves_an_existing_holdout_section(self):
        row = {"min_confluence": 3, "min_ob_body_ratio": 0.5, "min_fvg_gap_ratio": 0.5,
               "total_trades": 12, "win_rate_pct": 50.0, "profit_factor": 1.5, "total_pnl": 200.0, "roi_pct": 1.0,
               "max_drawdown_pct": 3.0, "avg_r_multiple": 0.4, "median_r_multiple": 0.3, "std_r_multiple": 0.2,
               "buy_hold_return_pct": 2.0}
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


def _sample_row(**overrides) -> dict:
    row = {"min_confluence": 3, "min_ob_body_ratio": 0.5, "min_fvg_gap_ratio": 0.5,
           "total_trades": 12, "long_trades": 12, "short_trades": 0, "win_rate_pct": 50.0, "profit_factor": 1.5,
           "total_pnl": 200.0, "roi_pct": 1.0, "max_drawdown_pct": 3.0, "avg_r_multiple": 0.4,
           "median_r_multiple": 0.3, "std_r_multiple": 0.2, "buy_hold_return_pct": 2.0}
    row.update(overrides)
    return row


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


class TestAppendHoldoutResult(unittest.TestCase):
    def test_creates_the_file_and_section_if_neither_exists(self):
        row = {"min_confluence": 2, "min_ob_body_ratio": 1.0, "min_fvg_gap_ratio": 1.0,
               "total_trades": 8, "win_rate_pct": 37.5, "profit_factor": 0.9, "total_pnl": -50.0, "roi_pct": -0.25,
               "max_drawdown_pct": 1.5, "avg_r_multiple": -0.1, "median_r_multiple": -0.05, "std_r_multiple": 0.3,
               "buy_hold_return_pct": 0.4}

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            self.assertFalse(path.exists())
            append_holdout_result(row, "2026-01-01", "2026-03-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertIn("Holdout evaluations", content)
        self.assertIn("min_confluence=2", content)
        # The metric table itself must actually render, not just the params prose -
        # this specifically catches a prior bug where the table was built by
        # string-slicing _format_row()'s output and silently produced garbage.
        self.assertIn("| 8 | 37.5 | 0.90 | -50 | -0.25 | 1.50 | -0.10 | -0.05 | 0.30 |", content)

    def test_appending_twice_keeps_both_entries(self):
        row = {"min_confluence": 2, "min_ob_body_ratio": 1.0, "min_fvg_gap_ratio": 1.0,
               "total_trades": 8, "win_rate_pct": 37.5, "profit_factor": 0.9, "total_pnl": -50.0, "roi_pct": -0.25,
               "max_drawdown_pct": 1.5, "avg_r_multiple": -0.1, "median_r_multiple": -0.05, "std_r_multiple": 0.3,
               "buy_hold_return_pct": 0.4}

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning-log.md"
            append_holdout_result(row, "2026-01-01", "2026-03-31", path=path)
            append_holdout_result({**row, "min_confluence": 3}, "2026-01-01", "2026-03-31", path=path)
            content = path.read_text(encoding="utf-8")

        self.assertIn("min_confluence=2", content)
        self.assertIn("min_confluence=3", content)


if __name__ == "__main__":
    unittest.main()
