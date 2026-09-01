"""
test_risk.py
backtest/risk.py::RiskManager tests. Covers the absolute-dollar min_sl_distance
restored from the original Crypto-Trading-Bot (see risk.py's module docstring for why
that differs from tradingagents-kr's percent-based version) plus the 1.5R take-profit
and WIN/LOSS balance bookkeeping.
"""
import unittest

from backtest.risk import RiskManager


class TestRiskManagerSizing(unittest.TestCase):
    def setUp(self):
        # Large enough balance that the spot purchasing-power cap doesn't kick in
        # for these SL distances - that cap gets its own dedicated test below.
        self.risk = RiskManager(initial_balance=100_000.0, min_sl_distance=50.0)

    def test_calculate_size_basic(self):
        # 1% of 100,000 = $1,000 risk. Entry 50000, SL 49000 -> $1000 risk/unit -> size = 1
        size = self.risk.calculate_size(entry_price=50_000, stop_loss=49_000)
        self.assertEqual(size, 1.0)

    def test_calculate_size_zero_when_sl_equals_entry(self):
        self.assertEqual(self.risk.calculate_size(entry_price=50_000, stop_loss=50_000), 0.0)

    def test_calculate_size_zero_when_entry_non_positive(self):
        self.assertEqual(self.risk.calculate_size(entry_price=0, stop_loss=-1), 0.0)

    def test_sl_tighter_than_min_distance_is_rejected(self):
        # $10 SL distance is below the $50 minimum safe threshold - reject.
        size = self.risk.calculate_size(entry_price=50_000, stop_loss=49_990)
        self.assertEqual(size, 0.0)

    def test_sl_at_min_distance_is_accepted(self):
        size = self.risk.calculate_size(entry_price=50_000, stop_loss=49_950)
        self.assertGreater(size, 0.0)

    def test_size_capped_by_spot_purchasing_power(self):
        # Deliberately huge risk_pct so ideal size would cost more than the balance.
        risk = RiskManager(initial_balance=1_000.0, risk_pct=0.50, min_sl_distance=1.0)
        size = risk.calculate_size(entry_price=50_000, stop_loss=49_999)
        max_affordable = (1_000.0 * 0.98) / 50_000
        self.assertAlmostEqual(size, max_affordable)


class TestTakeProfitPrice(unittest.TestCase):
    def setUp(self):
        self.risk = RiskManager(initial_balance=1_000.0, rr_ratio=1.5)

    def test_long_take_profit_is_above_entry(self):
        tp = self.risk.take_profit_price(entry_price=50_000, stop_loss=49_900, side="BUY")
        self.assertEqual(tp, 50_150.0)  # 100 * 1.5 above entry

    def test_short_take_profit_is_below_entry(self):
        tp = self.risk.take_profit_price(entry_price=50_000, stop_loss=50_100, side="SELL")
        self.assertEqual(tp, 49_850.0)  # 100 * 1.5 below entry


class TestApplyResult(unittest.TestCase):
    def setUp(self):
        self.risk = RiskManager(initial_balance=1000.0, risk_pct=0.01, rr_ratio=1.5)

    def test_win_adds_rr_ratio_times_risk(self):
        pnl = self.risk.apply_result("WIN")
        self.assertEqual(pnl, 15.0)
        self.assertEqual(self.risk.current_balance, 1015.0)

    def test_loss_subtracts_risk_amount(self):
        pnl = self.risk.apply_result("LOSS")
        self.assertEqual(pnl, -10.0)
        self.assertEqual(self.risk.current_balance, 990.0)


if __name__ == "__main__":
    unittest.main()
