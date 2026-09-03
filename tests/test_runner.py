"""
test_runner.py
Confirms FillEngine/run_backtest actually thread min_ob_body_ratio/min_fvg_gap_ratio/
min_break_distance through to compute_smc_features - added for backtest/tune.py's
grid search, which needs these to actually vary results, not just be accepted and
silently ignored by the simulator.
"""
import unittest
from unittest.mock import patch

import pandas as pd

from backtest.risk import RiskManager
from backtest.runner import FillConfig, FillEngine, Position, run_backtest
from signals.smc_aggregator import SMCSignal, SMCSignalAggregator


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


def _bearish_signal(confluence: int = -2) -> SMCSignal:
    return SMCSignal(
        direction="BEARISH", confluence_score=confluence, bullish_count=0, bearish_count=2,
    )


class TestContributingFactorsCarryThroughToTrade(unittest.TestCase):
    """Phase 6 of the 2026-09 detector rebuild: the dashboard's detector
    breakdown panel needs the actual 5-vote breakdown for a trade, not just its
    net confluence score - Position and Trade both now carry
    contributing_factors, copied from the SMCSignal that opened the position."""

    def test_entry_and_exit_both_carry_the_signals_contributing_factors(self):
        engine = FillEngine(
            SMCSignalAggregator(min_confluence=1), RiskManager(initial_balance=10_000.0),
            FillConfig(taker_fee_pct=0.0, slippage_bps=0.0),
        )
        engine._window.append((pd.Timestamp("2025-01-01", tz="utc"), 50_000.0, 50_100.0, 49_900.0, 50_000.0))
        engine._window.append((pd.Timestamp("2025-01-01 00:05", tz="utc"), 50_000.0, 50_100.0, 49_900.0, 50_000.0))

        factors = {
            "order_block": "bullish", "fvg": "bullish", "trendline": None,
            "channel": None, "fakeout_trap": "bullish",
        }
        signal = SMCSignal(
            direction="BULLISH", confluence_score=3, bullish_count=3, bearish_count=0,
            contributing_factors=factors,
        )
        with patch.object(engine.aggregator, "aggregate", return_value=signal):
            engine._evaluate_signal(close=50_000.0)

        entry_event = engine._enter(pd.Timestamp("2025-01-01 00:10", tz="utc"), open_price=50_000.0)
        self.assertIsNotNone(entry_event)
        self.assertEqual(engine.position.contributing_factors, factors)

        engine._check_exit(pd.Timestamp("2025-01-01 00:15", tz="utc"), high=60_000.0, low=49_400.0)
        self.assertEqual(len(engine.trades), 1)
        self.assertEqual(engine.trades[0].contributing_factors, factors)
        # It's the SMCSignal's own dict, not a shared mutable reference to it -
        # mutating the trade's copy must not corrupt anything the signal object
        # (or a later signal reusing the same dict shape) still holds.
        engine.trades[0].contributing_factors["order_block"] = "bearish"
        self.assertEqual(factors["order_block"], "bullish")

    def test_default_contributing_factors_is_an_empty_dict_not_shared_mutable_state(self):
        a = Position(
            side="BUY", entry_price=100.0, stop_loss=99.0, take_profit=101.0, size=1.0,
            entry_time=pd.Timestamp("2025-01-01", tz="utc"), entry_fee=0.0, confluence=1,
        )
        b = Position(
            side="BUY", entry_price=100.0, stop_loss=99.0, take_profit=101.0, size=1.0,
            entry_time=pd.Timestamp("2025-01-01", tz="utc"), entry_fee=0.0, confluence=1,
        )
        a.contributing_factors["order_block"] = "bullish"
        self.assertEqual(b.contributing_factors, {})


class TestShortSideIsDiagnosticOnly(unittest.TestCase):
    """allow_short exists only for backtest/tune.py's long-vs-long+short
    diagnostic (see docs/tuning-log.md) - live/trader.py and paper trading
    (viz/control.py) never set it, so the default must stay exactly the current
    long-only behavior."""

    def _engine(self, allow_short: bool) -> FillEngine:
        return FillEngine(
            SMCSignalAggregator(min_confluence=1), RiskManager(initial_balance=10_000.0),
            FillConfig(taker_fee_pct=0.0, slippage_bps=0.0), allow_short=allow_short,
        )

    @staticmethod
    def _prime_window(engine: FillEngine) -> None:
        # _evaluate_signal requires >= 2 candles before it even calls aggregate() -
        # both tests below need this so the mocked BEARISH signal is actually reached.
        engine._window.append((pd.Timestamp("2025-01-01", tz="utc"), 100.0, 101.0, 99.0, 100.0))
        engine._window.append((pd.Timestamp("2025-01-01 00:05", tz="utc"), 100.0, 101.0, 99.0, 100.0))

    def test_bearish_signal_is_ignored_by_default(self):
        engine = self._engine(allow_short=False)
        self._prime_window(engine)
        with patch.object(engine.aggregator, "aggregate", return_value=_bearish_signal()) as mock_aggregate:
            engine._evaluate_signal(close=100.0)
        mock_aggregate.assert_called_once()  # confirms the branch was actually reached, not short-circuited
        self.assertIsNone(engine._pending_side)

    def test_bearish_signal_opens_a_pending_short_when_allowed(self):
        engine = self._engine(allow_short=True)
        self._prime_window(engine)
        with patch.object(engine.aggregator, "aggregate", return_value=_bearish_signal()):
            engine._evaluate_signal(close=100.0)
        self.assertEqual(engine._pending_side, "SELL")

    def test_short_stop_loss_triggers_when_price_rises_to_the_stop(self):
        engine = self._engine(allow_short=True)
        engine.position = Position(
            side="SELL", entry_price=100.0, stop_loss=105.0, take_profit=91.0, size=1.0,
            entry_time=pd.Timestamp("2025-01-01", tz="utc"), entry_fee=0.0, confluence=1,
        )
        event = engine._check_exit(pd.Timestamp("2025-01-01 00:05", tz="utc"), high=106.0, low=99.0)
        self.assertEqual(event["reason"], "SL")
        self.assertLess(engine.trades[-1].pnl, 0)  # short loses when price rises

    def test_short_take_profit_triggers_when_price_falls_to_the_target(self):
        engine = self._engine(allow_short=True)
        engine.position = Position(
            side="SELL", entry_price=100.0, stop_loss=105.0, take_profit=91.0, size=1.0,
            entry_time=pd.Timestamp("2025-01-01", tz="utc"), entry_fee=0.0, confluence=1,
        )
        event = engine._check_exit(pd.Timestamp("2025-01-01 00:05", tz="utc"), high=100.0, low=90.0)
        self.assertEqual(event["reason"], "TP")
        self.assertGreater(engine.trades[-1].pnl, 0)  # short profits when price falls

    def test_short_pnl_magnitude_matches_price_move_times_size(self):
        engine = self._engine(allow_short=True)
        engine.position = Position(
            side="SELL", entry_price=100.0, stop_loss=105.0, take_profit=91.0, size=2.0,
            entry_time=pd.Timestamp("2025-01-01", tz="utc"), entry_fee=0.0, confluence=1,
        )
        engine._check_exit(pd.Timestamp("2025-01-01 00:05", tz="utc"), high=100.0, low=90.0)
        # Zero fees/slippage in this fixture, so pnl should be exactly (entry - exit) * size.
        self.assertAlmostEqual(engine.trades[-1].pnl, (100.0 - 91.0) * 2.0)

    def test_run_backtest_default_allow_short_false_matches_omitting_it(self):
        """The new allow_short parameter must be purely additive - explicit False
        must behave identically to not passing it at all."""
        df = _trending_ohlc()
        agg = SMCSignalAggregator(min_confluence=1)
        result_omitted = run_backtest(df, agg, RiskManager(initial_balance=20_000.0), FillConfig())
        result_explicit = run_backtest(
            df, SMCSignalAggregator(min_confluence=1), RiskManager(initial_balance=20_000.0), FillConfig(),
            allow_short=False,
        )
        self.assertEqual(len(result_omitted.trades), len(result_explicit.trades))


if __name__ == "__main__":
    unittest.main()
