"""
test_smc.py
Regression tests for data/smc.py's detectors, most importantly detect_bos_choch().

Ported from tradingagents-kr's tests/test_smc.py, which itself documents a real bug
found there: "structure_bias: bullish" failed to update after a swing low that broke
the prior low on wick only (not close). See the "Bug fix" note in
data/smc.py::detect_bos_choch for the full explanation - this test reproduces exactly
that scenario so the fix can't silently regress.
"""
import unittest

import pandas as pd

from data.smc import compute_smc_features, detect_bos_choch, detect_swings


def _df_from_ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.index = pd.date_range("2026-01-01", periods=len(df), freq="5min", tz="utc")
    return df


class TestBosChochStaleBiasRegression(unittest.TestCase):
    """Reproduces the real bug: a new swing low confirmed by wick only, lower than an
    existing swing low that close never actually broke, must still fire a CHoCH."""

    def setUp(self):
        # i0-6: V-shaped bottom, swing low A confirmed at i=6 (low=95, formed at i=3).
        # i7-14: wick dips to 89 at i=11 (swing low B), but close never dips below 95
        # anywhere in i0-14 (asserted directly below).
        self.rows = [
            (101, 102, 100, 101),  # 0
            (100, 101, 98, 99),    # 1
            (99, 100, 97, 98),     # 2
            (97, 98, 95, 96),      # 3  swing low candidate A (low=95)
            (97, 99, 97, 98),      # 4
            (98, 100, 98, 99),     # 5
            (99, 101, 99, 100),    # 6  A confirmed (i=6)
            (100, 101, 97, 99),    # 7
            (99, 100, 95, 98),     # 8
            (98, 99, 93, 97),      # 9
            (97, 98, 91, 97),      # 10
            (97, 99, 89, 98),      # 11 swing low candidate B (low=89 < 95, close=98 stays above 95)
            (98, 100, 91, 98),     # 12
            (98, 99, 93, 97),      # 13
            (97, 98, 95, 96),      # 14 B confirmed (i=14)
            (96, 97, 95, 96),      # 15
            (96, 97, 95, 96),      # 16
        ]
        self.df = _df_from_ohlc(self.rows)

    def test_close_never_breaks_the_old_low_before_the_new_swing_confirms(self):
        """Sanity check that this really is the "wick-only break, no close break"
        scenario - otherwise this test wouldn't reproduce the bug at all."""
        self.assertTrue((self.df["close"].iloc[:15] > 95).all(), self.df["close"].iloc[:15].tolist())

    def test_lower_low_without_close_break_still_fires_choch(self):
        swings = detect_swings(self.df, left_bars=3, right_bars=3)
        swing_lows = [s for s in swings if s["type"] == "low"]
        self.assertEqual([s["price"] for s in swing_lows], [95.0, 89.0])

        events = detect_bos_choch(self.df, swings, right_bars=3)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["type"], "CHoCH")
        self.assertEqual(event["direction"], "bearish")
        self.assertEqual(event["price"], 95.0)   # must point at the level that actually broke
        self.assertEqual(event["index"], 11)     # recorded when the new swing confirmed (i=11)

    def test_compute_smc_features_reflects_the_fix_end_to_end(self):
        """Confirms the fix holds through the full compute_smc_features pipeline,
        not just the detector in isolation."""
        feats = compute_smc_features(self.df, lookback_bars=90)
        self.assertEqual(feats["structure_bias"], "bearish")
        self.assertEqual(len(feats["structure_events"]), 1)
        self.assertEqual(feats["structure_events"][0]["direction"], "bearish")


class TestBosChochNoSpuriousEventOnTightening(unittest.TestCase):
    """Confirms the fix doesn't over-fire: a higher low (structure holding in an
    uptrend) is not a break, so the level should update with no event."""

    def setUp(self):
        self.rows = [
            (101, 102, 100, 101),  # 0
            (100, 101, 98, 99),    # 1
            (99, 100, 97, 98),     # 2
            (97, 98, 95, 96),      # 3  swing low A (95)
            (97, 99, 97, 98),      # 4
            (98, 100, 98, 99),     # 5
            (99, 101, 99, 100),    # 6  A confirmed (i=6)
            (101, 102, 100, 101),  # 7
            (100, 101, 99, 100),   # 8
            (99, 100, 98, 99),     # 9
            (98, 99, 97, 98),      # 10 swing low B candidate (low=97, higher than A - tightening)
            (98, 99, 98, 99),      # 11
            (99, 100, 99, 100),    # 12
            (100, 101, 100, 101),  # 13 B confirmed (i=13)
            (100, 101, 99, 100),   # 14
            (100, 101, 99, 100),   # 15
            (100, 101, 99, 100),   # 16
        ]
        self.df = _df_from_ohlc(self.rows)

    def test_higher_low_ratchets_with_no_event(self):
        swings = detect_swings(self.df, left_bars=3, right_bars=3)
        swing_lows = [s for s in swings if s["type"] == "low"]
        self.assertEqual([s["price"] for s in swing_lows], [95.0, 97.0])

        events = detect_bos_choch(self.df, swings, right_bars=3)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
