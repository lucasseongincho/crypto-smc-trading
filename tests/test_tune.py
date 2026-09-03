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
    MIN_BREAK_DISTANCE_GRID,
    MIN_CONFLUENCE_GRID,
    MIN_FVG_GAP_RATIO_GRID,
    MIN_OB_BODY_RATIO_GRID,
    GridPoint,
    _run_point,
    append_holdout_result,
    build_grid,
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
    def test_grid_size_matches_the_product_of_all_four_parameter_lists(self):
        grid = build_grid()
        expected = (
            len(MIN_CONFLUENCE_GRID) * len(MIN_OB_BODY_RATIO_GRID)
            * len(MIN_FVG_GAP_RATIO_GRID) * len(MIN_BREAK_DISTANCE_GRID)
        )
        self.assertEqual(len(grid), expected)
        self.assertEqual(len(grid), 256)  # 4 values each, as specified

    def test_grid_has_no_duplicate_points(self):
        grid = build_grid()
        as_tuples = {(p.min_confluence, p.min_ob_body_ratio, p.min_fvg_gap_ratio, p.min_break_distance) for p in grid}
        self.assertEqual(len(as_tuples), len(grid))

    def test_every_grid_value_is_within_its_declared_list(self):
        grid = build_grid()
        for p in grid:
            self.assertIn(p.min_confluence, MIN_CONFLUENCE_GRID)
            self.assertIn(p.min_ob_body_ratio, MIN_OB_BODY_RATIO_GRID)
            self.assertIn(p.min_fvg_gap_ratio, MIN_FVG_GAP_RATIO_GRID)
            self.assertIn(p.min_break_distance, MIN_BREAK_DISTANCE_GRID)


class TestRunPoint(unittest.TestCase):
    def test_returns_params_and_full_trade_level_metrics(self):
        df = _synthetic_ohlc()
        point = GridPoint(min_confluence=1, min_ob_body_ratio=0.0, min_fvg_gap_ratio=0.0, min_break_distance=0.0)

        row = _run_point(point, df)

        self.assertEqual(row["min_confluence"], 1)
        self.assertEqual(row["min_ob_body_ratio"], 0.0)
        for key in ["total_trades", "win_rate_pct", "profit_factor", "roi_pct", "avg_r_multiple", "median_r_multiple"]:
            self.assertIn(key, row)

    def test_stricter_confluence_never_produces_more_trades_than_looser_on_the_same_data(self):
        df = _synthetic_ohlc(n=90)
        loose = _run_point(GridPoint(1, 0.0, 0.0, 0.0), df)
        strict = _run_point(GridPoint(4, 0.0, 0.0, 0.0), df)
        self.assertLessEqual(strict["total_trades"], loose["total_trades"])


class TestWriteTuningLog(unittest.TestCase):
    def test_writes_a_markdown_table_with_every_result_row(self):
        results = [
            {"min_confluence": c, "min_ob_body_ratio": 0.0, "min_fvg_gap_ratio": 0.0, "min_break_distance": 0.0,
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
            {"min_confluence": 4, "min_ob_body_ratio": 2.0, "min_fvg_gap_ratio": 2.0, "min_break_distance": 100.0,
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
        row = {"min_confluence": 3, "min_ob_body_ratio": 0.5, "min_fvg_gap_ratio": 0.5, "min_break_distance": 50.0,
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


class TestAppendHoldoutResult(unittest.TestCase):
    def test_creates_the_file_and_section_if_neither_exists(self):
        row = {"min_confluence": 2, "min_ob_body_ratio": 1.0, "min_fvg_gap_ratio": 1.0, "min_break_distance": 25.0,
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
        row = {"min_confluence": 2, "min_ob_body_ratio": 1.0, "min_fvg_gap_ratio": 1.0, "min_break_distance": 25.0,
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
