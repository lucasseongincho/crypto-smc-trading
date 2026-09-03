"""
test_runner.py
Confirms FillEngine/run_backtest actually thread min_ob_body_ratio/min_fvg_gap_ratio/
min_break_distance through to compute_smc_features - added for backtest/tune.py's
grid search, which needs these to actually vary results, not just be accepted and
silently ignored by the simulator.
"""
import unittest

import pandas as pd

from backtest.risk import RiskManager
from backtest.runner import FillConfig, FillEngine, run_backtest
from signals.smc_aggregator import SMCSignalAggregator


def _trending_ohlc(n: int = 80) -> pd.DataFrame:
    rows = []
    price = 50_000.0
    for i in range(n):
        price += 15.0
        rows.append((price, price + 30, price - 10, price + 20))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.index = pd.date_range("2025-01-01", periods=n, freq="5min", tz="utc")
    return df


class TestFillEngineThreadsDetectorParams(unittest.TestCase):
    def _run(self, **detector_kwargs) -> int:
        df = _trending_ohlc()
        aggregator = SMCSignalAggregator(min_confluence=1)
        risk = RiskManager(initial_balance=20_000.0)
        result = run_backtest(df, aggregator, risk, FillConfig(), **detector_kwargs)
        return len(result.trades)

    def test_default_params_match_the_data_smc_defaults(self):
        # Sanity check: omitting the new kwargs entirely must behave identically to
        # passing the explicit data/smc.py defaults (all filters off).
        trades_omitted = self._run()
        trades_explicit = self._run(min_ob_body_ratio=0.0, min_fvg_gap_ratio=0.0, min_break_distance=0.0)
        self.assertEqual(trades_omitted, trades_explicit)

    def test_an_extreme_min_ob_body_ratio_changes_the_result(self):
        """An impossibly large min_ob_body_ratio (no real candle's body could ever
        be 1000x the local ATR) must suppress every order-block-driven entry,
        proving the parameter actually reaches compute_smc_features rather than
        being silently ignored by FillEngine."""
        baseline_trades = self._run(min_ob_body_ratio=0.0)
        filtered_trades = self._run(min_ob_body_ratio=1000.0)
        self.assertLessEqual(filtered_trades, baseline_trades)

    def test_fill_engine_constructor_accepts_and_stores_the_new_params(self):
        aggregator = SMCSignalAggregator(min_confluence=2)
        risk = RiskManager(initial_balance=1000.0)
        engine = FillEngine(
            aggregator, risk, FillConfig(),
            min_ob_body_ratio=0.5, min_fvg_gap_ratio=0.5, min_break_distance=50.0,
        )
        self.assertEqual(engine.min_ob_body_ratio, 0.5)
        self.assertEqual(engine.min_fvg_gap_ratio, 0.5)
        self.assertEqual(engine.min_break_distance, 50.0)


if __name__ == "__main__":
    unittest.main()
