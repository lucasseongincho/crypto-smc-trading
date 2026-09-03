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

from data.smc import (
    compute_smc_features,
    detect_bos_choch,
    detect_channel,
    detect_fakeout,
    detect_fvg,
    detect_order_blocks,
    detect_swings,
    detect_trendline,
)


def _df_from_ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.index = pd.date_range("2026-01-01", periods=len(df), freq="5min", tz="utc")
    return df


def _build_zigzag_ohlc(
    extremes: list[tuple[int, float, str]], start_price: float = 180.0, extreme_wick: float = 0.5,
) -> list[tuple[float, float, float, float]]:
    """Builds OHLC rows that pass through each (index, price, 'low'|'high') extreme
    as a clean, unambiguous 3-left/3-right fractal - each extreme candle gets an
    exaggerated wick beyond its own body in the extreme's direction (every other
    candle is body-only, high=max(open,close)/low=min(open,close)), and prices
    move strictly monotonically between consecutive extremes. Used for
    detect_trendline/detect_channel tests, where hand-picking OHLC values that
    both form a clean swing *and* land on a specific regression-fit price is
    otherwise fiddly to get right by hand."""
    total = extremes[-1][0] + 4
    targets: list[float | None] = [None] * total
    anchors = [(-1, start_price)] + [(idx, price) for idx, price, _ in extremes]
    for (i0, p0), (i1, p1) in zip(anchors, anchors[1:]):
        steps = i1 - i0
        for k in range(1, steps + 1):
            i = i0 + k
            targets[i] = p0 + (p1 - p0) * (k / steps)
    last_idx, last_price = anchors[-1]
    _, prev_price = anchors[-2]
    for i in range(last_idx + 1, total):
        targets[i] = last_price + (last_price - prev_price) * 0.1 * (i - last_idx)

    extreme_map = {idx: kind for idx, _, kind in extremes}
    rows: list[tuple[float, float, float, float]] = []
    prev_close = start_price
    for i in range(total):
        open_ = prev_close
        close = targets[i]
        body_hi, body_lo = max(open_, close), min(open_, close)
        if extreme_map.get(i) == "low":
            hi, lo = body_hi, body_lo - extreme_wick
        elif extreme_map.get(i) == "high":
            hi, lo = body_hi + extreme_wick, body_lo
        else:
            hi, lo = body_hi, body_lo
        rows.append((open_, hi, lo, close))
        prev_close = close
    return rows


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


class TestOrderBlockBodyEngulfRedefinition(unittest.TestCase):
    """Phase 1 of the 2026-09 detector rebuild (see
    ../crypto-smc-bot-notes/decisions/2026-09-03-detector-rebuild-decision.md):
    order blocks are now a plain body-engulf pattern, not close-break-high. These
    tests are deliberately NOT bit-identical-compatible with the old behavior -
    that's the point of the redefinition, not a regression."""

    def test_body_engulf_without_close_break_now_qualifies(self):
        # curr's body fully wraps prev's body, but curr's close (100.4) never
        # breaks prev's high (100) - the old close-break-high test would have
        # rejected this outright. The new body-engulf test accepts it.
        df = _df_from_ohlc([
            (99.8, 100, 99.5, 99.6),   # red, body [99.6, 99.8]
            (99.5, 100.5, 99.4, 100.4),  # green, body [99.5, 100.4] wraps [99.6, 99.8]
        ])
        obs = detect_order_blocks(df)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["type"], "bullish")
        self.assertEqual(obs[0]["pattern"], "single")

    def test_partial_body_overlap_without_full_wrap_is_rejected(self):
        # curr's body only partially overlaps prev's body - not a full engulf.
        df = _df_from_ohlc([
            (100, 101, 99, 99.5),   # red, body [99.5, 100]
            (99.7, 102, 99, 101.5),  # green, body [99.7, 101.5] - low 99.7 > prev low 99.5
        ])
        self.assertEqual(detect_order_blocks(df), [])

    def test_same_color_candles_never_qualify_as_engulf(self):
        df = _df_from_ohlc([
            (99, 100, 98.5, 99.8),   # green
            (99.5, 101, 99.4, 100.9),  # also green, body wraps prev's body
        ])
        self.assertEqual(detect_order_blocks(df), [])

    def test_double_order_block_upgrades_the_middle_candle(self):
        # A(bearish) -> B(bullish, engulfs A's body) -> C(bearish, engulfs B's
        # body). The base pair-engulf test (B, C) already flags B as a resistance
        # order block; because B *also* engulfed A (the opposite-direction
        # candle before it), B additionally gets a "double" pattern flag.
        df = _df_from_ohlc([
            (100, 100.2, 98.5, 99),      # A: red, body [99, 100]
            (98.8, 101, 98.7, 100.9),    # B: green, body [98.8, 100.9] wraps A's [99,100]
            (101.2, 101.3, 98, 98.5),    # C: red, body [98.5, 101.2] wraps B's [98.8,100.9]
        ])
        obs = detect_order_blocks(df)
        # A is engulfed by B -> single bullish OB at A (index 1).
        # B is engulfed by C -> single bearish OB at B (index 2), PLUS the double
        # upgrade (also index 2, since it's found while examining the (B, C) pair).
        single_obs = [o for o in obs if o["pattern"] == "single"]
        double_obs = [o for o in obs if o["pattern"] == "double"]
        self.assertEqual(len(single_obs), 2)
        self.assertEqual(len(double_obs), 1)

        bullish_single = [o for o in single_obs if o["type"] == "bullish"][0]
        self.assertEqual(bullish_single["high"], 100.2)  # A's wick high, not body top
        self.assertEqual(bullish_single["low"], 98.5)    # A's wick low, not body bottom

        bearish_single = [o for o in single_obs if o["type"] == "bearish"][0]
        double = double_obs[0]
        # Same zone (the middle candle B's own high/low) for both the base
        # single-pair result and its double-pattern upgrade - it's the same
        # candle, just carrying an extra flag, not a separate zone.
        self.assertEqual(double["type"], "bearish")
        self.assertEqual(double["high"], bearish_single["high"])
        self.assertEqual(double["low"], bearish_single["low"])
        self.assertEqual(double["high"], 101)    # B's wick high, not body top
        self.assertEqual(double["low"], 98.7)    # B's wick low, not body bottom

    def test_no_double_upgrade_when_look_back_pair_is_same_direction(self):
        # A(bearish) -> B(bearish, same direction - no engulf relationship
        # matters here since same-color pairs never register at all) -> C(bearish,
        # engulfs B). No opposite-direction engulf immediately before B, so no
        # double-OB upgrade - just the ordinary single OB from (B, C) if B is
        # itself green, which it isn't here, so nothing at all fires from this
        # pair; kept intentionally trivial/negative.
        df = _df_from_ohlc([
            (100, 100.1, 99, 99.5),   # A: red
            (99.4, 99.6, 98, 98.5),   # B: red (same direction as A)
            (98.6, 98.7, 97, 97.5),   # C: red (same direction as B)
        ])
        self.assertEqual(detect_order_blocks(df), [])


class TestDetectTrendline(unittest.TestCase):
    """Phase 2 of the 2026-09 detector rebuild: real diagonal trendlines,
    replacing classify_swing_trend's 2-swing comparison. Three rising swing lows
    at exactly (index 3, price 100), (index 13, price 110), (index 23, price 120)
    - collinear by construction, slope=1.0, intercept=97 (price = index + 97) -
    then a crash candle at index 30 whose close (101) breaks well below the
    projected support value (127) there. Three rising swing highs are also
    present as a side effect of the same fixture, giving a resistance line for
    free without a dedicated fixture."""

    def setUp(self):
        rows = [
            (140, 141, 135, 136), (136, 137, 130, 131), (131, 132, 102, 103),
            (103, 104, 100, 101),  # index 3: swing low, price 100
            (101, 110, 101, 109), (109, 120, 108, 119), (119, 130, 115, 129),
            (129, 140, 125, 139), (139, 150, 135, 149), (149, 155, 140, 150),
            (150, 151, 140, 141), (141, 142, 130, 131), (131, 132, 111, 112),
            (112, 113, 110, 111),  # index 13: swing low, price 110
            (111, 120, 111, 119), (119, 130, 118, 129), (129, 140, 125, 139),
            (139, 150, 135, 149), (149, 160, 145, 159), (159, 165, 150, 160),
            (160, 161, 150, 151), (151, 152, 140, 141), (141, 142, 121, 122),
            (122, 123, 120, 121),  # index 23: swing low, price 120
            (121, 130, 121, 129), (129, 140, 128, 139), (139, 150, 135, 149),
            (149, 160, 145, 159), (159, 175, 155, 174), (174, 180, 165, 175),
            (175, 176, 100, 101),  # index 30: crash candle, close 101
            (101, 105, 95, 100), (100, 102, 90, 95),
        ]
        self.df = _df_from_ohlc(rows)
        self.swings = detect_swings(self.df, left_bars=3, right_bars=3)

    def test_swing_fixture_is_as_designed(self):
        """Sanity check the fixture actually produces the intended swings, so a
        failure below points at detect_trendline, not a miscounted fixture."""
        lows = [(s["index"], s["price"]) for s in self.swings if s["type"] == "low"]
        highs = [(s["index"], s["price"]) for s in self.swings if s["type"] == "high"]
        self.assertEqual(lows, [(3, 100.0), (13, 110.0), (23, 120.0)])
        self.assertEqual(highs, [(9, 155.0), (19, 165.0), (29, 180.0)])

    def test_support_line_fits_the_three_swing_lows_exactly(self):
        result = detect_trendline(self.df, self.swings, right_bars=3, trendline_points=3)
        # Collinear points -> exact least-squares fit: price = 1.0 * index + 97.0
        self.assertIsNone(result["support"])  # invalidated by the break below
        self.assertEqual(len(result["breaks"]), 1)
        brk = result["breaks"][0]
        self.assertEqual(brk["type"], "SUPPORT_BREAK")
        self.assertEqual(brk["direction"], "bearish")
        self.assertEqual(brk["index"], 30)
        self.assertAlmostEqual(brk["price"], 127.0)  # slope*30 + intercept = 1*30+97

    def test_close_not_wick_is_what_triggers_the_break(self):
        # Index 30's low (100) is far below the line, but so is every candle's low
        # near the crash - the break event's recorded index/price already prove
        # it fired off `close`, not the wick: the projected value (127) equals
        # slope*i+intercept, evaluated once, independent of that bar's high/low.
        result = detect_trendline(self.df, self.swings, right_bars=3, trendline_points=3)
        brk = result["breaks"][0]
        crash_close = float(self.df.iloc[30]["close"])
        self.assertEqual(crash_close, 101.0)
        self.assertLess(crash_close, brk["price"])

    def test_resistance_line_fits_the_three_swing_highs(self):
        result = detect_trendline(self.df, self.swings, right_bars=3, trendline_points=3)
        # Swing highs (9,155), (19,165), (29,180) are not perfectly collinear
        # (155->165 is +10 over 10 bars, 165->180 is +15 over 10 bars) - just
        # confirms a resistance line gets fit independently of the support line,
        # without asserting on the exact non-round least-squares numbers.
        self.assertIsNotNone(result["resistance"])
        self.assertGreater(result["resistance"]["slope"], 0)  # still an uptrend in highs

    def test_trendline_points_two_reproduces_the_exact_two_point_line(self):
        # Only the first two swing lows exist in this truncated window - with
        # trendline_points=2, least-squares through exactly 2 points must equal
        # the raw two-point line through them (the other design option raised in
        # the rebuild decision, unified into the same parameter).
        truncated = self.df.iloc[:17]
        swings = detect_swings(truncated, left_bars=3, right_bars=3)
        result = detect_trendline(truncated, swings, right_bars=3, trendline_points=2)
        self.assertIsNotNone(result["support"])
        self.assertAlmostEqual(result["support"]["slope"], 1.0)
        self.assertAlmostEqual(result["support"]["intercept"], 97.0)

    def test_min_trendline_break_distance_filters_small_breaks(self):
        # The crash close (101) is 26 below the projected line value (127) at
        # index 30 - clears a small threshold, filtered out by a larger one.
        small = detect_trendline(
            self.df, self.swings, right_bars=3, trendline_points=3,
            min_trendline_break_distance=10.0,
        )
        self.assertEqual(len(small["breaks"]), 1)

        large = detect_trendline(
            self.df, self.swings, right_bars=3, trendline_points=3,
            min_trendline_break_distance=50.0,
        )
        self.assertEqual(large["breaks"], [])

    def test_fewer_than_trendline_points_swings_yields_no_line(self):
        truncated = self.df.iloc[:10]  # only the first swing low (index 3) exists
        swings = detect_swings(truncated, left_bars=3, right_bars=3)
        result = detect_trendline(truncated, swings, right_bars=3, trendline_points=3)
        self.assertIsNone(result["support"])
        self.assertIsNone(result["resistance"])
        self.assertEqual(result["breaks"], [])

    def test_no_swings_at_all_yields_no_line(self):
        result = detect_trendline(self.df, [], right_bars=3, trendline_points=3)
        self.assertEqual(result, {"support": None, "resistance": None, "breaks": []})

    def test_compute_smc_features_exposes_trendline_not_swing_trend(self):
        feats = compute_smc_features(self.df, lookback_bars=len(self.df))
        self.assertIn("trendline", feats)
        self.assertNotIn("swing_trend", feats)
        self.assertEqual(len(feats["trendline"]["breaks"]), 1)


class TestDetectChannel(unittest.TestCase):
    """Phase 3 of the 2026-09 detector rebuild: parallel channel boundaries around
    the Phase 2 trendlines. Fixture: perfectly collinear swing lows at (3,100),
    (13,110),(23,120) and swing highs at (9,250),(19,260),(29,270) - both slope
    1.0 by construction, support intercept 96.5 (shifted -0.5 by the extreme
    wick), resistance intercept 241.5 - so support and resistance end up
    numerically parallel here, and the ascending/descending channels end up
    identical, which is a useful cross-check, not a limitation of the test."""

    def setUp(self):
        extremes = [
            (3, 100.0, "low"), (9, 250.0, "high"), (13, 110.0, "low"),
            (19, 260.0, "high"), (23, 120.0, "low"), (29, 270.0, "high"), (33, 230.0, "low"),
        ]
        rows = _build_zigzag_ohlc(extremes)
        # Hand-appended candles isolating one touch/break per bar against the
        # boundaries implied by slope=1.0, support intercept=96.5, resistance
        # intercept=241.5 (upper = i+241.5, lower = i+96.5):
        rows += [
            (rows[-1][3], 278.5, 250.0, 270.0),  # i=37: high==upper(278.5) exactly, close stays inside -> touch upper
            (270.0, 291.0, 260.0, 290.0),         # i=38: close(290) > upper(279.5) -> break upper
            (290.0, 200.0, 135.5, 150.0),          # i=39: low==lower(135.5) exactly, close stays inside -> touch lower
            (150.0, 151.0, 99.0, 100.0),           # i=40: close(100) < lower(136.5) -> break lower
        ]
        self.df = _df_from_ohlc(rows)
        self.swings = detect_swings(self.df, left_bars=3, right_bars=3)

    def test_fixture_swings_are_as_designed(self):
        lows = [(s["index"], s["price"]) for s in self.swings if s["type"] == "low"]
        highs = [(s["index"], s["price"]) for s in self.swings if s["type"] == "high"]
        self.assertEqual(lows, [(3, 99.5), (13, 109.5), (23, 119.5)])
        self.assertEqual(highs, [(9, 250.5), (19, 260.5), (29, 270.5)])

    def test_ascending_and_descending_channels_share_the_same_slope_and_boundaries(self):
        result = detect_channel(self.df.iloc[:34], self.swings, right_bars=3, trendline_points=3)
        self.assertIsNotNone(result["ascending"])
        self.assertIsNotNone(result["descending"])
        self.assertAlmostEqual(result["ascending"]["slope"], 1.0)
        self.assertAlmostEqual(result["descending"]["slope"], 1.0)
        self.assertAlmostEqual(result["ascending"]["lower"]["intercept"], 96.5)
        self.assertAlmostEqual(result["ascending"]["upper"]["intercept"], 241.5)
        # Ascending's lower boundary is literally the support line; descending's
        # upper boundary is literally the resistance line - same numbers either way.
        self.assertAlmostEqual(
            result["ascending"]["lower"]["intercept"], result["descending"]["lower"]["intercept"],
        )
        self.assertAlmostEqual(
            result["ascending"]["upper"]["intercept"], result["descending"]["upper"]["intercept"],
        )

    def test_touch_upper_boundary_fires_on_wick_without_close_breaking(self):
        result = detect_channel(self.df.iloc[:38], self.swings, right_bars=3, trendline_points=3)
        touches = [e for e in result["events"] if e["index"] == 37]
        self.assertEqual(len(touches), 2)  # both channels share this boundary here
        for e in touches:
            self.assertEqual(e["type"], "CHANNEL_TOUCH")
            self.assertEqual(e["boundary"], "upper")
            self.assertEqual(e["direction"], "bearish")  # touching a ceiling -> potential reversal down
            self.assertAlmostEqual(e["price"], 278.5)

    def test_break_upper_boundary_fires_on_close_beyond_it(self):
        result = detect_channel(self.df.iloc[:39], self.swings, right_bars=3, trendline_points=3)
        breaks = [e for e in result["events"] if e["index"] == 38]
        self.assertEqual(len(breaks), 2)
        for e in breaks:
            self.assertEqual(e["type"], "CHANNEL_BREAK")
            self.assertEqual(e["boundary"], "upper")
            self.assertEqual(e["direction"], "bullish")  # breaking a ceiling -> bullish, mirrors RESISTANCE_BREAK
            self.assertAlmostEqual(e["price"], 279.5)

    def test_touch_lower_boundary_fires_on_wick_without_close_breaking(self):
        result = detect_channel(self.df.iloc[:40], self.swings, right_bars=3, trendline_points=3)
        touches = [e for e in result["events"] if e["index"] == 39 and e["type"] == "CHANNEL_TOUCH"]
        self.assertEqual(len(touches), 1)  # descending channel already invalidated by the i=38 break
        e = touches[0]
        self.assertEqual(e["boundary"], "lower")
        self.assertEqual(e["direction"], "bullish")  # touching a floor -> potential bounce up
        self.assertAlmostEqual(e["price"], 135.5)

    def test_break_lower_boundary_fires_on_close_beyond_it(self):
        result = detect_channel(self.df, self.swings, right_bars=3, trendline_points=3)
        breaks = [e for e in result["events"] if e["index"] == 40 and e["type"] == "CHANNEL_BREAK"]
        self.assertEqual(len(breaks), 1)
        e = breaks[0]
        self.assertEqual(e["boundary"], "lower")
        self.assertEqual(e["direction"], "bearish")  # breaking a floor -> bearish, mirrors SUPPORT_BREAK
        self.assertAlmostEqual(e["price"], 136.5)

    def test_channel_invalidates_after_its_anchor_line_breaks(self):
        # descending channel's upper boundary IS the resistance line - once that
        # breaks (i=38), the descending channel must be gone from then on.
        result = detect_channel(self.df, self.swings, right_bars=3, trendline_points=3)
        self.assertIsNone(result["descending"])
        # ascending channel's lower boundary IS the support line - it breaks too,
        # at i=40, so by the end of this fixture it's also gone.
        self.assertIsNone(result["ascending"])

    def test_fewer_than_trendline_points_of_the_other_side_yields_no_channel(self):
        # Only 2 confirmed swing lows exist yet (index 23's isn't confirmed until
        # i=26) - even though the highs already have 2 confirmed points too, no
        # channel exists without >= trendline_points on *both* sides.
        truncated = self.df.iloc[:20]
        swings = detect_swings(truncated, left_bars=3, right_bars=3)
        result = detect_channel(truncated, swings, right_bars=3, trendline_points=3)
        self.assertIsNone(result["ascending"])
        self.assertIsNone(result["descending"])

    def test_no_swings_at_all_yields_no_channel(self):
        result = detect_channel(self.df, [], right_bars=3, trendline_points=3)
        self.assertEqual(result, {"ascending": None, "descending": None, "events": []})

    def test_min_channel_break_distance_filters_small_breaks(self):
        # The i=38 close (290) clears the upper boundary (279.5) by 10.5.
        small = detect_channel(
            self.df.iloc[:39], self.swings, right_bars=3, trendline_points=3,
            min_channel_break_distance=5.0,
        )
        self.assertTrue(any(e["index"] == 38 and e["type"] == "CHANNEL_BREAK" for e in small["events"]))

        large = detect_channel(
            self.df.iloc[:39], self.swings, right_bars=3, trendline_points=3,
            min_channel_break_distance=50.0,
        )
        self.assertFalse(any(e["index"] == 38 and e["type"] == "CHANNEL_BREAK" for e in large["events"]))

    def test_compute_smc_features_exposes_channel(self):
        feats = compute_smc_features(self.df, lookback_bars=len(self.df))
        self.assertIn("channel", feats)
        self.assertIn("ascending", feats["channel"])
        self.assertIn("descending", feats["channel"])
        self.assertIn("events", feats["channel"])


class TestDetectFakeoutGeneralizedLevels(unittest.TestCase):
    """Phase 4 of the 2026-09 detector rebuild: detect_fakeout now returns
    {"fakeout": ...|None, "trap": ...|None} and checks swing/trendline/channel
    levels in that priority order for the classic 2-candle sweep-and-reclaim."""

    def test_swing_level_fires_when_no_trendline_exists_yet(self):
        rows = [
            (101, 102, 100, 101), (100, 101, 98, 99), (99, 100, 97, 98), (97, 98, 95, 96),
            (97, 99, 97, 98), (98, 100, 98, 99), (99, 101, 99, 100),
            (100, 101, 90, 92),  # prev: low=90 sweeps below the swing low (95)
            (92, 97, 92, 96),    # curr: close=96 reclaims above 95
        ]
        df = _df_from_ohlc(rows)
        swings = detect_swings(df, left_bars=3, right_bars=3)
        result = detect_fakeout(df, swings, right_bars=3, trendline_points=3)
        self.assertIsNotNone(result["fakeout"])
        self.assertEqual(result["fakeout"]["type"], "BULL_FAKEOUT")
        self.assertEqual(result["fakeout"]["source"], "swing")
        self.assertEqual(result["fakeout"]["swept_level"], 95.0)
        self.assertIsNone(result["trap"])

    def test_falls_through_to_trendline_when_the_raw_swing_level_isnt_swept(self):
        # Three collinear swing lows (support line: slope=1.0, intercept=96.5,
        # confirmed at i=26) - the crafted prev/curr candles sweep well above the
        # raw last swing low (119.5, far below current price by this point) but
        # below the trendline's current projected value (124.5 at i=28), and
        # reclaim above it.
        extremes = [
            (3, 100.0, "low"), (9, 250.0, "high"), (13, 110.0, "low"),
            (19, 260.0, "high"), (23, 120.0, "low"),
        ]
        # _build_zigzag_ohlc always sizes its output to extremes[-1][0]+4 - slice
        # to just past the last designed extreme (index 23) and hand-append the
        # rest, since the helper's own trailing extrapolation continues in the
        # incoming direction rather than reversing (fine for a fixture that ends
        # mid-trend, not fine for confirming a swing as the actual endpoint).
        rows = _build_zigzag_ohlc(extremes)[:24]
        rows += [
            (120.0, 200.0, 120.0, 195.0),  # 24
            (195.0, 260.0, 195.0, 255.0),  # 25
            (255.0, 280.0, 255.0, 275.0),  # 26: support active, line=122.5
            (135.0, 136.0, 121.0, 124.0),  # 27: prev, low=121 (>119.5 swing, <123.5 line); close 124 (no break)
            (124.0, 127.0, 122.0, 126.0),  # 28: curr, close=126 (>124.5 line -> reclaim)
        ]
        df = _df_from_ohlc(rows)
        swings = detect_swings(df, left_bars=3, right_bars=3)
        support = detect_trendline(df, swings, right_bars=3, trendline_points=3)["support"]
        self.assertIsNotNone(support)  # sanity: the line must still be alive at the end

        result = detect_fakeout(df, swings, right_bars=3, trendline_points=3)
        self.assertIsNotNone(result["fakeout"])
        self.assertEqual(result["fakeout"]["type"], "BULL_FAKEOUT")
        self.assertEqual(result["fakeout"]["source"], "trendline")
        self.assertAlmostEqual(result["fakeout"]["swept_level"], 124.5)


class TestDetectTrap(unittest.TestCase):
    """Phase 4: trap detection. Fixture: a support line breaks (crash to close=50,
    broken_level=123.5), then a double-bottom-and-reclaim sequence - first
    extreme (E1) at (index 31, price 39.5), a bounce high, a second nearby
    extreme (E2) at (index 43, price 41.5, above E1 - a shallower retest), then a
    reclaim candle closing (128) back above the broken level (123.5)."""

    def setUp(self):
        extremes = [
            (3, 100.0, "low"), (9, 250.0, "high"), (13, 110.0, "low"),
            (19, 260.0, "high"), (23, 120.0, "low"),
        ]
        rows = _build_zigzag_ohlc(extremes)[:24]
        rows += [
            (120.0, 200.0, 120.0, 195.0),
            (195.0, 260.0, 195.0, 255.0),
            (255.0, 280.0, 255.0, 275.0),
            (275.0, 276.0, 45.0, 50.0),  # index 27: crash, SUPPORT_BREAK at 123.5
        ]
        post_extremes = [(3, 40.0, "low"), (9, 70.0, "high"), (15, 42.0, "low")]
        rows += _build_zigzag_ohlc(post_extremes, start_price=50.0)[:16]  # local 0..15, absolute 28..43
        rows += [
            (42.0, 55.0, 42.0, 54.0),    # 44
            (54.0, 66.0, 54.0, 65.0),    # 45
            (65.0, 76.0, 65.0, 75.0),    # 46: confirms index 43's swing (43+3=46)
            (75.0, 130.0, 75.0, 128.0),  # 47: reclaim, close 128 > broken_level 123.5
        ]
        self.rows = rows

    def test_fixture_produces_the_designed_break_and_swings(self):
        df = _df_from_ohlc(self.rows)
        swings = detect_swings(df, left_bars=3, right_bars=3)
        lows = [(s["index"], s["price"]) for s in swings if s["type"] == "low"]
        self.assertEqual(lows, [(3, 99.5), (13, 109.5), (23, 119.5), (31, 39.5), (43, 41.5)])
        breaks = detect_trendline(df, swings, right_bars=3, trendline_points=3)["breaks"]
        self.assertEqual(breaks[0]["type"], "SUPPORT_BREAK")
        self.assertAlmostEqual(breaks[0]["price"], 123.5)
        self.assertEqual(breaks[0]["index"], 27)

    def test_full_sequence_confirms_a_support_trap_on_the_reclaim_bar(self):
        df = _df_from_ohlc(self.rows)
        swings = detect_swings(df, left_bars=3, right_bars=3)
        result = detect_fakeout(df, swings, right_bars=3, trendline_points=3)
        trap = result["trap"]
        self.assertIsNotNone(trap)
        self.assertEqual(trap["type"], "SUPPORT_TRAP")
        self.assertEqual(trap["direction"], "bullish")
        self.assertAlmostEqual(trap["broken_level"], 123.5)
        self.assertAlmostEqual(trap["stop_loss_reference"], 39.5)  # E1's price
        self.assertEqual(trap["first_extreme_index"], 31)
        self.assertEqual(trap["index"], 47)

    def test_no_trap_reported_when_reclaim_isnt_on_the_current_last_bar(self):
        # Drop the reclaim candle - the sequence is otherwise complete, but
        # nothing reclaims on what is now the last bar.
        df = _df_from_ohlc(self.rows[:-1])
        swings = detect_swings(df, left_bars=3, right_bars=3)
        result = detect_fakeout(df, swings, right_bars=3, trendline_points=3)
        self.assertIsNone(result["trap"])

    def test_trap_abandoned_when_second_extreme_extends_past_first_beyond_max_distance(self):
        # Replace E2 (index 43, price 41.5) with a lower low (a real continuation,
        # not a shallow retest) and require retests to stay within a tight distance.
        rows = list(self.rows)
        # local index 15 of the post-break zigzag is absolute index 43.
        deep_low_extremes = [(3, 40.0, "low"), (9, 70.0, "high"), (15, 20.0, "low")]
        post_rows = _build_zigzag_ohlc(deep_low_extremes, start_price=50.0)[:16]
        rows[28:44] = post_rows
        df = _df_from_ohlc(rows)
        swings = detect_swings(df, left_bars=3, right_bars=3)

        permissive = detect_fakeout(
            df, swings, right_bars=3, trendline_points=3,
            max_trap_retest_distance=float("inf"),
        )
        self.assertIsNotNone(permissive["trap"])  # default behavior - always accepted

        strict = detect_fakeout(
            df, swings, right_bars=3, trendline_points=3,
            max_trap_retest_distance=5.0,
        )
        self.assertIsNone(strict["trap"])  # 20.0 is 19.5 away from E1 (39.5) - too far

    def test_compute_smc_features_exposes_the_new_fakeout_trap_shape(self):
        df = _df_from_ohlc(self.rows)
        feats = compute_smc_features(df, lookback_bars=len(df))
        self.assertIn("fakeout", feats)
        self.assertIn("fakeout", feats["fakeout"])
        self.assertIn("trap", feats["fakeout"])
        self.assertEqual(feats["fakeout"]["trap"]["type"], "SUPPORT_TRAP")


class TestNewSizeFilterParametersDefaultToOff(unittest.TestCase):
    """min_ob_body_ratio/min_fvg_gap_ratio/min_break_distance were added so these
    previously-invisible, hardcoded-off assumptions are visible and configurable
    (see docs/detector-logic.md). Confirms each filter's default (0.0) is a true
    no-op reproducing prior behavior, and that a deliberately extreme test-only
    value actually excludes what the default would have found - not tuning a real
    threshold, just proving the wiring works."""

    def test_min_ob_body_ratio_filters_small_candles(self):
        df = _df_from_ohlc([
            (100, 101, 99, 99.5),    # red, small body (0.5)
            (99.5, 102, 99, 101.5),  # green, closes above prev high -> OB at default
        ])

        self.assertEqual(len(detect_order_blocks(df)), 1)  # default (0.0) - unfiltered
        self.assertEqual(detect_order_blocks(df, min_ob_body_ratio=5.0), [])

    def test_min_fvg_gap_ratio_filters_small_gaps(self):
        df = _df_from_ohlc([
            (99, 100, 98, 99.5),
            (99.5, 100.5, 99, 100.0),
            (100, 100.5, 100.2, 100.3),  # low=100.2 > candle_1 high=100 -> small gap
        ])

        self.assertEqual(len(detect_fvg(df)), 1)  # default (0.0) - unfiltered
        self.assertEqual(detect_fvg(df, min_fvg_gap_ratio=5.0), [])

    def test_min_break_distance_filters_small_breaks(self):
        df = _df_from_ohlc([
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (102, 102, 102, 102),  # close breaks the 101.0 level by 1.0
        ])
        swings = [{"type": "high", "price": 101.0, "index": 0, "time": None}]

        # right_bars=0 so the swing is usable immediately - isolates the
        # close-crossing distance check from swing-confirmation timing.
        events = detect_bos_choch(df, swings, right_bars=0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["price"], 101.0)

        # Distance (1.0) clears a smaller threshold - still fires.
        self.assertEqual(len(detect_bos_choch(df, swings, right_bars=0, min_break_distance=0.5)), 1)
        # Distance (1.0) doesn't clear a larger threshold - filtered out.
        self.assertEqual(detect_bos_choch(df, swings, right_bars=0, min_break_distance=2.0), [])


if __name__ == "__main__":
    unittest.main()
