"""
test_smc_vectorization_equivalence.py
Permanent regression test for the numpy-array rewrite of data/smc.py's detector
internals (order blocks, FVG, swings, BOS/CHoCH previously used window.iloc[i]
loops; now read from precomputed numpy arrays - see each function's docstring).

The functions below prefixed _ref_ are a frozen, verbatim copy of the pre-
vectorization implementation (as it existed before that rewrite), kept here
specifically so this equivalence stays checked by the test suite going forward,
not just verified once in a one-off script against the full real archive at
migration time (that one-off verification - run against all 1,313,022 real rows,
diffed field-by-field - is what actually established this claim; see the git log
for that commit. This test is the ongoing regression guard for the same claim on a
smaller, fast, synthetic dataset, so a future edit to the hot-path arrays can't
silently drift from the original semantics without a test noticing).

Reasoning for why this is expected to be bit-identical, not just
tolerance-equivalent: the rewrite changed only the data-access layer
(window.iloc[i]["field"] -> array[i]) and left every comparison, arithmetic
operation, and its order completely unchanged. Reading the same underlying float64
value via .iloc vs. a numpy array involves no arithmetic, so it produces the exact
same bit pattern; the actual arithmetic (subtraction, comparison) is the same
operation on the same operands in the same order either way. _atr() itself was left
completely untouched (still pandas rolling().mean()) rather than reimplemented, for
exactly this reason - a rolling-mean reimplementation risks a different
floating-point summation order even for a "mathematically equivalent" formula.

**2026-09 scope note:** the 2026-09 detector rebuild (see docs/detector-logic.md and
../crypto-smc-bot-notes/decisions/2026-09-03-detector-rebuild-decision.md)
deliberately redefined detect_order_blocks' engulf test (body-engulf, not
close-break-high) to match a reference video's spec - a real behavior change, not a
bug fix, so bit-identical equivalence against _ref_detect_order_blocks below no
longer applies to that one detector. _ref_detect_order_blocks is kept, frozen at the
pre-rebuild definition, specifically so a canary test can confirm the two
implementations still genuinely diverge (catching an accidental revert) rather than
deleting the reference outright. fvgs/swings/BOS-CHoCH are untouched by that rebuild
and remain checked for exact equivalence as before.
"""
import random
import unittest
from typing import Any

import pandas as pd

from data import smc as new

# ---- Frozen reference implementation (pre-vectorization) --------------------------


def _ref_time_str(window: pd.DataFrame, i: int) -> str | None:
    try:
        return pd.Timestamp(window.index[i]).isoformat()
    except Exception:
        return None


def _ref_atr(window: pd.DataFrame, period: int = 14) -> pd.Series:
    high = window["high"].astype(float)
    low = window["low"].astype(float)
    prev_close = window["close"].astype(float).shift(1)
    true_range = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()


def _ref_detect_order_blocks(window: pd.DataFrame, min_ob_body_ratio: float = 0.0) -> list[dict[str, Any]]:
    ob_list: list[dict[str, Any]] = []
    if len(window) < 2:
        return ob_list
    atr = _ref_atr(window)
    for i in range(1, len(window)):
        prev = window.iloc[i - 1]
        curr = window.iloc[i]
        ts = _ref_time_str(window, i)
        prev_body = abs(float(prev["close"]) - float(prev["open"]))
        if prev_body < min_ob_body_ratio * float(atr.iloc[i - 1]):
            continue
        if float(prev["close"]) < float(prev["open"]) and float(curr["close"]) > float(curr["open"]):
            if float(curr["close"]) > float(prev["high"]):
                ob_list.append({"type": "bullish", "high": float(prev["high"]), "low": float(prev["low"]),
                                 "index": i, "time": ts})
        elif float(prev["close"]) > float(prev["open"]) and float(curr["close"]) < float(curr["open"]):
            if float(curr["close"]) < float(prev["low"]):
                ob_list.append({"type": "bearish", "high": float(prev["high"]), "low": float(prev["low"]),
                                 "index": i, "time": ts})
    return ob_list


def _ref_detect_fvg(window: pd.DataFrame, min_fvg_gap_ratio: float = 0.0) -> list[dict[str, Any]]:
    fvg_list: list[dict[str, Any]] = []
    if len(window) < 3:
        return fvg_list
    atr = _ref_atr(window)
    for i in range(2, len(window)):
        candle_1 = window.iloc[i - 2]
        candle_3 = window.iloc[i]
        ts = _ref_time_str(window, i)
        min_gap = min_fvg_gap_ratio * float(atr.iloc[i - 2])
        if float(candle_3["low"]) > float(candle_1["high"]):
            gap = float(candle_3["low"]) - float(candle_1["high"])
            if gap >= min_gap:
                fvg_list.append({"type": "bullish", "top": float(candle_3["low"]), "bottom": float(candle_1["high"]),
                                  "index": i, "time": ts})
        elif float(candle_3["high"]) < float(candle_1["low"]):
            gap = float(candle_1["low"]) - float(candle_3["high"])
            if gap >= min_gap:
                fvg_list.append({"type": "bearish", "top": float(candle_1["low"]), "bottom": float(candle_3["high"]),
                                  "index": i, "time": ts})
    return fvg_list


def _ref_detect_swings(window: pd.DataFrame, left_bars: int = 3, right_bars: int = 3) -> list[dict[str, Any]]:
    swings: list[dict[str, Any]] = []
    if len(window) < left_bars + right_bars + 1:
        return swings
    for i in range(left_bars, len(window) - right_bars):
        curr = window.iloc[i]
        ts = _ref_time_str(window, i)
        is_high = True
        for j in range(1, left_bars + 1):
            if float(window.iloc[i - j]["high"]) >= float(curr["high"]):
                is_high = False
                break
        if is_high:
            for j in range(1, right_bars + 1):
                if float(window.iloc[i + j]["high"]) >= float(curr["high"]):
                    is_high = False
                    break
        if is_high:
            swings.append({"type": "high", "price": float(curr["high"]), "index": i, "time": ts})
        is_low = True
        for j in range(1, left_bars + 1):
            if float(window.iloc[i - j]["low"]) <= float(curr["low"]):
                is_low = False
                break
        if is_low:
            for j in range(1, right_bars + 1):
                if float(window.iloc[i + j]["low"]) <= float(curr["low"]):
                    is_low = False
                    break
        if is_low:
            swings.append({"type": "low", "price": float(curr["low"]), "index": i, "time": ts})
    return swings


def _ref_detect_bos_choch(
    window: pd.DataFrame, swings: list[dict[str, Any]], right_bars: int = 3, min_break_distance: float = 0.0,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not swings:
        return events
    swings_sorted = sorted(swings, key=lambda s: s["index"])
    swing_ptr = 0
    pending_high: dict[str, Any] | None = None
    pending_low: dict[str, Any] | None = None
    bias: str | None = None
    for i in range(len(window)):
        while swing_ptr < len(swings_sorted) and swings_sorted[swing_ptr]["index"] + right_bars <= i:
            s = swings_sorted[swing_ptr]
            if s["type"] == "high":
                if pending_high is not None and s["price"] > pending_high["price"]:
                    event_type = "BOS" if bias == "bullish" else "CHoCH"
                    events.append({"type": event_type, "direction": "bullish", "price": pending_high["price"],
                                    "index": s["index"], "time": s["time"]})
                    bias = "bullish"
                pending_high = s
            else:
                if pending_low is not None and s["price"] < pending_low["price"]:
                    event_type = "BOS" if bias == "bearish" else "CHoCH"
                    events.append({"type": event_type, "direction": "bearish", "price": pending_low["price"],
                                    "index": s["index"], "time": s["time"]})
                    bias = "bearish"
                pending_low = s
            swing_ptr += 1
        close = float(window.iloc[i]["close"])
        ts = _ref_time_str(window, i)
        if pending_high is not None and close > pending_high["price"] + min_break_distance:
            event_type = "BOS" if bias == "bullish" else "CHoCH"
            events.append({"type": event_type, "direction": "bullish", "price": pending_high["price"],
                            "index": i, "time": ts})
            bias = "bullish"
            pending_high = None
        elif pending_low is not None and close < pending_low["price"] - min_break_distance:
            event_type = "BOS" if bias == "bearish" else "CHoCH"
            events.append({"type": event_type, "direction": "bearish", "price": pending_low["price"],
                            "index": i, "time": ts})
            bias = "bearish"
            pending_low = None
    return events


# ---- Synthetic fixture with real variety -----------------------------------------


def _rich_synthetic_ohlc(n: int = 400, seed: int = 1234) -> pd.DataFrame:
    """A seeded random walk with occasional large 'impulse' candles and occasional
    tiny-bodied candles, so the min_ob_body_ratio/min_fvg_gap_ratio filters have both
    candidates they should keep and candidates they should exclude to compare over -
    a plain trend or plain noise series wouldn't exercise the size filters at all."""
    rng = random.Random(seed)
    rows = []
    price = 50_000.0
    for i in range(n):
        if i % 17 == 0:
            body = rng.choice([-1, 1]) * rng.uniform(200, 400)  # large impulse candle
        elif i % 11 == 0:
            body = rng.choice([-1, 1]) * rng.uniform(0.5, 3.0)  # near-flat candle
        else:
            body = rng.uniform(-60, 60)
        open_ = price
        close = price + body
        high = max(open_, close) + rng.uniform(1, 25)
        low = min(open_, close) - rng.uniform(1, 25)
        rows.append((open_, high, low, close))
        price = close
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.index = pd.date_range("2025-01-01", periods=n, freq="5min", tz="utc")
    return df


class TestVectorizedDetectorsMatchTheFrozenReference(unittest.TestCase):
    def setUp(self):
        self.df = _rich_synthetic_ohlc()

    def test_detect_order_blocks_deliberately_diverges_from_the_frozen_reference(self):
        """detect_order_blocks was redefined in the 2026-09 detector rebuild (body-
        engulf, not close-break-high - see docs/detector-logic.md's "Order block"
        section and ../crypto-smc-bot-notes/decisions/2026-09-03-detector-rebuild-
        decision.md). _ref_detect_order_blocks below is frozen at the OLD
        (pre-rebuild) close-break-high definition, so it is no longer expected to
        match new.detect_order_blocks - that equivalence assertion was retired on
        purpose, not silently dropped. This test instead confirms the two
        implementations genuinely diverge (i.e. the redefinition is actually live,
        not accidentally reverted to the old behavior) - a canary in the other
        direction from every other test in this file."""
        old = _ref_detect_order_blocks(self.df, min_ob_body_ratio=0.5)
        new_result = new.detect_order_blocks(self.df, min_ob_body_ratio=0.5)
        self.assertNotEqual(old, new_result)

    def test_detect_fvg_matches_across_a_spread_of_ratios(self):
        for ratio in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]:
            with self.subTest(ratio=ratio):
                self.assertEqual(
                    _ref_detect_fvg(self.df, min_fvg_gap_ratio=ratio),
                    new.detect_fvg(self.df, min_fvg_gap_ratio=ratio),
                )

    def test_detect_swings_matches(self):
        self.assertEqual(_ref_detect_swings(self.df), new.detect_swings(self.df))

    def test_detect_swings_matches_with_asymmetric_left_right_bars(self):
        self.assertEqual(
            _ref_detect_swings(self.df, left_bars=2, right_bars=5),
            new.detect_swings(self.df, left_bars=2, right_bars=5),
        )

    def test_detect_bos_choch_matches_across_a_spread_of_distances(self):
        swings = new.detect_swings(self.df)
        for dist in [0.0, 10.0, 50.0, 100.0]:
            with self.subTest(dist=dist):
                self.assertEqual(
                    _ref_detect_bos_choch(self.df, swings, min_break_distance=dist),
                    new.detect_bos_choch(self.df, swings, min_break_distance=dist),
                )

    def test_compute_smc_features_matches_end_to_end(self):
        """Rebuilds the full feature dict via the frozen reference functions (the
        same way the pre-vectorization compute_smc_features assembled them) and
        compares against the live compute_smc_features, catching any integration-
        level drift the per-detector tests above wouldn't (e.g. the shared-ATR
        wiring introduced alongside the vectorization).

        order_blocks is deliberately excluded from this comparison - see
        test_detect_order_blocks_deliberately_diverges_from_the_frozen_reference
        above; the 2026-09 body-engulf redefinition means the frozen reference is
        no longer expected to agree with it, on purpose. fvgs/swings/
        structure_events are untouched by that rebuild phase and still must match
        exactly."""
        ref_fvgs = _ref_detect_fvg(self.df, min_fvg_gap_ratio=0.5)
        ref_swings = _ref_detect_swings(self.df)
        ref_events = _ref_detect_bos_choch(self.df, ref_swings, min_break_distance=25.0)

        actual = new.compute_smc_features(
            self.df, lookback_bars=len(self.df),
            min_ob_body_ratio=0.5, min_fvg_gap_ratio=0.5, min_break_distance=25.0,
        )

        self.assertEqual(actual["fvgs"], ref_fvgs)
        self.assertEqual(actual["swings"], ref_swings)
        self.assertEqual(actual["structure_events"], ref_events)

    def test_values_are_native_python_float_not_numpy_scalar(self):
        """The numpy-array rewrite must not leak numpy.float64 into the output -
        those aren't JSON-serializable by the dashboard's plain json.dumps path,
        unlike the original .iloc-based code's explicit float(...) wrapping."""
        obs = new.detect_order_blocks(self.df)
        self.assertGreater(len(obs), 0)
        self.assertIs(type(obs[0]["high"]), float)
        self.assertIs(type(obs[0]["low"]), float)

        swings = new.detect_swings(self.df)
        self.assertGreater(len(swings), 0)
        self.assertIs(type(swings[0]["price"]), float)


if __name__ == "__main__":
    unittest.main()
