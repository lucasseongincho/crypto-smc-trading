"""
smc.py
Smart Money Concepts (SMC) structure detectors: OHLCV DataFrame -> SMC feature dict.
Pure detection/computation only - no buy/sell judgment (that's signals/smc_aggregator.py).

Ported from tradingagents-kr's data/smc.py, which itself was ported from the original
crypto bot's ob_fvg.py/structure.py/trendline.py/fakeout.py. That intermediate port
target daily stock bars; this port target is back to 5-minute crypto bars, so
`lookback_bars` below is marked TODO rather than carried over as a settled value -
90 daily bars and 90 5-minute bars represent wildly different amounts of wall-clock
history, and the right lookback for structure analysis needs its own validation pass.
"""
from typing import Any

import pandas as pd


def _time_str(window: pd.DataFrame, i: int) -> str | None:
    """window.index[i] -> ISO timestamp string, or None if the index isn't datetime-like.
    Used directly only by detect_fakeout(), which only ever needs one index (O(1),
    no benefit from precomputing). The O(n) loops below use _all_time_strs()
    instead - same per-element try/except as this function, just computed once up
    front rather than re-deriving pd.Timestamp(window.index[i]) on every iteration."""
    try:
        return pd.Timestamp(window.index[i]).isoformat()
    except Exception:
        return None


def _all_time_strs(window: pd.DataFrame) -> list[str | None]:
    """Equivalent to [_time_str(window, i) for i in range(len(window))], computed
    in one pass over window.index rather than by repeated positional index access -
    the O(n) detectors below call this once and index into the result, instead of
    each iteration re-doing pd.Timestamp(window.index[i]).isoformat()."""
    result: list[str | None] = []
    for t in window.index:
        try:
            result.append(pd.Timestamp(t).isoformat())
        except Exception:
            result.append(None)
    return result


def _atr(window: pd.DataFrame, period: int = 14) -> pd.Series:
    """True-range-based ATR, used only as the volatility yardstick for the
    min_ob_body_ratio/min_fvg_gap_ratio filters below - not itself an exposed
    feature. period=14 is the standard textbook ATR default (not an SMC-specific
    tuning choice like the ratios that use it), so it isn't threaded through as its
    own configurable parameter."""
    high = window["high"].astype(float)
    low = window["low"].astype(float)
    prev_close = window["close"].astype(float).shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()


# TODO: no order-block minimum-size filter existed before this parameter was added -
# any 2-candle engulf qualified regardless of how small the candle was (see
# docs/detector-logic.md's "Order block" section). 0.0 (off) reproduces that exact
# prior behavior byte-for-byte; a real nonzero value needs its own validation pass
# for 5-minute BTC/USD bars before being trusted - this is a placeholder for making
# the assumption visible/configurable, not a chosen value.
DEFAULT_MIN_OB_BODY_RATIO = 0.0


def _body_engulf_type(
    prev_open: float, prev_close: float, curr_open: float, curr_close: float,
) -> str | None:
    """Body-engulf test shared by both the single and double order-block patterns
    below (see the "double order block" section of docs/detector-logic.md for why
    this is one function, not two): curr's body [min(open,close), max(open,close)]
    must fully wrap prev's body, AND the two candles must be opposite colors. No
    close-vs-prior-high/low requirement - see the rebuild decision note this
    replaces (2026-09, matching the reference video's plain body-engulf definition
    instead of the old close-break-high test).

    Returns "bullish" (curr green engulfing a red prev - a support/bullish OB
    candidate at prev) or "bearish" (curr red engulfing a green prev - a
    resistance/bearish OB candidate at prev), or None if the bodies don't engulf or
    the candles are the same color (a same-color "engulf" isn't a reversal pattern)."""
    prev_lo, prev_hi = min(prev_open, prev_close), max(prev_open, prev_close)
    curr_lo, curr_hi = min(curr_open, curr_close), max(curr_open, curr_close)
    if not (curr_lo <= prev_lo and curr_hi >= prev_hi):
        return None
    if prev_close < prev_open and curr_close > curr_open:
        return "bullish"
    if prev_close > prev_open and curr_close < curr_open:
        return "bearish"
    return None


def detect_order_blocks(
    window: pd.DataFrame,
    min_ob_body_ratio: float = DEFAULT_MIN_OB_BODY_RATIO,
    atr: pd.Series | None = None,
) -> list[dict[str, Any]]:
    """Order Block: a candle whose body is fully wrapped (engulfed) by the next
    candle's body, in the opposite direction, with a body at least
    min_ob_body_ratio * ATR(14) - a ratio rather than a raw price threshold since
    BTC's price scale drifts too much for a fixed dollar minimum to stay meaningful
    (same reasoning as backtest/risk.py's absolute-dollar min_sl_distance being
    flagged for re-validation, just solved with a ratio here instead). Still no
    BOS/CHoCH gating or mitigation tracking - see docs/detector-logic.md for why
    that's a separate, unaddressed gap from this size filter.

    Body-engulf, not close-break-high: this is a deliberate redefinition (2026-09)
    to match the reference video exactly - the prior version additionally required
    curr's *close* to break past prev's high/low, a stricter condition the video's
    definition doesn't use. See docs/detector-logic.md's "Order block" section for
    the full before/after comparison; this is a behavior change, not a bug fix, and
    is expected to change which candles qualify (no bit-identical guarantee here,
    unlike the earlier numpy-vectorization pass).

    Double order block: when the just-engulfed candle (the "middle" one) had itself
    engulfed the *opposite*-direction candle before it, the middle candle is flagged
    as a stronger ("double") order block on top of - not instead of - the ordinary
    single order block that pattern already produces at the earlier candle. Reuses
    _body_engulf_type() for both the single-pair check and the extra look-back
    check, rather than a separate comparison implementation - see
    docs/detector-logic.md for the full worked example and why this is "the same
    pair-engulf result, with an additional check upgrading it," not a distinct
    detector.

    atr: precomputed _atr(window), or None to compute it here. compute_smc_features
    passes its own precomputed value in, since detect_fvg needs the identical ATR
    series over the same window and recomputing it twice was, empirically, over
    half this function's runtime for no benefit - the value is identical either way
    (same window, same _atr() code path), this just avoids paying for it twice."""
    ob_list: list[dict[str, Any]] = []
    n = len(window)
    if n < 2:
        return ob_list

    opens = window["open"].to_numpy(dtype=float)
    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    closes = window["close"].to_numpy(dtype=float)
    atr_arr = (atr if atr is not None else _atr(window)).to_numpy(dtype=float)
    times = _all_time_strs(window)

    for i in range(1, n):
        prev_open, prev_high, prev_low, prev_close = opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]
        curr_open, curr_close = opens[i], closes[i]
        ts = times[i]

        prev_body = abs(prev_close - prev_open)
        if prev_body < min_ob_body_ratio * atr_arr[i - 1]:
            continue  # candle too small relative to recent volatility to count as an OB

        engulf_type = _body_engulf_type(prev_open, prev_close, curr_open, curr_close)
        if engulf_type is None:
            continue

        ob_list.append({
            "type": engulf_type, "high": float(prev_high), "low": float(prev_low),
            "index": i, "time": ts, "pattern": "single",
        })

        if i >= 2:
            prior_engulf_type = _body_engulf_type(
                opens[i - 2], closes[i - 2], prev_open, prev_close,
            )
            # The middle candle (prev, i-1) itself engulfed the opposite-direction
            # candle before it (i-2) - upgrade: flag the middle candle as a double
            # order block, same zone convention (its own high/low), same type as
            # the base result just appended above.
            if prior_engulf_type is not None and prior_engulf_type != engulf_type:
                ob_list.append({
                    "type": engulf_type, "high": float(prev_high), "low": float(prev_low),
                    "index": i, "time": ts, "pattern": "double",
                })

    return ob_list


# TODO: no FVG minimum-gap filter existed before this parameter was added - any
# nonzero gap qualified regardless of size (see docs/detector-logic.md's "FVG"
# section). 0.0 (off) reproduces that exact prior behavior byte-for-byte; a real
# nonzero value needs its own validation pass for 5-minute BTC/USD bars before
# being trusted - this is a placeholder for making the assumption visible/
# configurable, not a chosen value.
DEFAULT_MIN_FVG_GAP_RATIO = 0.0


def detect_fvg(
    window: pd.DataFrame,
    min_fvg_gap_ratio: float = DEFAULT_MIN_FVG_GAP_RATIO,
    atr: pd.Series | None = None,
) -> list[dict[str, Any]]:
    """Fair Value Gap: a 3-candle price gap where the middle candle's range is
    skipped, with a gap at least min_fvg_gap_ratio * ATR(14) - a ratio for the same
    reason as detect_order_blocks' min_ob_body_ratio (BTC's price scale drifts too
    much for a fixed dollar minimum to stay meaningful). The ATR reference point is
    candle_1's position (i-2), not candle_3's - deliberately excludes the
    gap-forming candles themselves from the volatility baseline used to judge
    whether the gap they form is significant.

    atr: precomputed _atr(window), or None to compute it here - see
    detect_order_blocks' docstring for why compute_smc_features passes one in
    (the same value either way, just computed once instead of twice)."""
    fvg_list: list[dict[str, Any]] = []
    n = len(window)
    if n < 3:
        return fvg_list

    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    atr_arr = (atr if atr is not None else _atr(window)).to_numpy(dtype=float)
    times = _all_time_strs(window)

    for i in range(2, n):
        c1_high, c1_low = highs[i - 2], lows[i - 2]
        c3_high, c3_low = highs[i], lows[i]
        ts = times[i]
        min_gap = min_fvg_gap_ratio * atr_arr[i - 2]

        if c3_low > c1_high:
            gap = c3_low - c1_high
            if gap >= min_gap:
                fvg_list.append({
                    "type": "bullish", "top": float(c3_low), "bottom": float(c1_high),
                    "index": i, "time": ts,
                })
        elif c3_high < c1_low:
            gap = c1_low - c3_high
            if gap >= min_gap:
                fvg_list.append({
                    "type": "bearish", "top": float(c1_low), "bottom": float(c3_high),
                    "index": i, "time": ts,
                })

    return fvg_list


def detect_swings(window: pd.DataFrame, left_bars: int = 3, right_bars: int = 3) -> list[dict[str, Any]]:
    """Fractal-style swing high/low detection. A swing at index i isn't confirmable
    until right_bars candles later - detect_bos_choch() accounts for that lag to avoid
    look-ahead bias."""
    swings: list[dict[str, Any]] = []
    n = len(window)
    if n < left_bars + right_bars + 1:
        return swings

    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    times = _all_time_strs(window)

    for i in range(left_bars, n - right_bars):
        curr_high = highs[i]
        curr_low = lows[i]
        ts = times[i]

        is_high = True
        for j in range(1, left_bars + 1):
            if highs[i - j] >= curr_high:
                is_high = False
                break
        if is_high:
            for j in range(1, right_bars + 1):
                if highs[i + j] >= curr_high:
                    is_high = False
                    break
        if is_high:
            swings.append({"type": "high", "price": float(curr_high), "index": i, "time": ts})

        is_low = True
        for j in range(1, left_bars + 1):
            if lows[i - j] <= curr_low:
                is_low = False
                break
        if is_low:
            for j in range(1, right_bars + 1):
                if lows[i + j] <= curr_low:
                    is_low = False
                    break
        if is_low:
            swings.append({"type": "low", "price": float(curr_low), "index": i, "time": ts})

    return swings


# classify_swing_trend was removed in the 2026-09 detector rebuild (Phase 2) - it's
# replaced by detect_trendline() below, not extended by it. The old function was a
# 2-swing higher-high/higher-low comparison that never actually fit a line despite
# its name (see git history / docs/detector-logic.md for the retired behavior);
# keeping both under similar names would have been confusing about which one a
# reader should trust. Nothing in this codebase calls classify_swing_trend anymore.


def _fit_line(points: list[tuple[int, float]]) -> tuple[float, float] | None:
    """Least-squares line (slope, intercept) through (index, price) points, i.e.
    price = slope * index + intercept. Returns None if fewer than 2 points, or (in
    principle, never actually reachable here since swing indices are always
    distinct) all points share one x value. With exactly 2 points this reduces
    exactly to the two-point line through them - see detect_trendline's docstring
    for why that's a deliberate unification, not a coincidence."""
    n = len(points)
    if n < 2:
        return None
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    sum_xx = sum(p[0] * p[0] for p in points)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return None
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


# TODO: the reference video specifies "fit a line through the swings" without
# specifying exact math (least-squares over N points vs. a raw two-point line
# through the most recent pair, extended forward). trendline_points=3 was chosen as
# the smallest N that produces a genuine least-squares fit rather than degenerating
# to the two-point case (trendline_points=2 exactly reproduces "two-point line
# through the most recent unbroken pair," since least squares through exactly 2
# points is the line through them, so both options the decision note raised are
# available via this one parameter - 2 for the raw two-point line, 3+ for an actual
# regression). Like lookback_bars, this is a functional placeholder, not a value
# validated against 5-minute BTC/USD bars.
DEFAULT_TRENDLINE_POINTS = 3

# TODO: same pattern as DEFAULT_MIN_BREAK_DISTANCE for detect_bos_choch - no
# minimum break-distance filter existed before this parameter, so 0.0 (off) is a
# placeholder for making the assumption visible/configurable, not a chosen value.
DEFAULT_MIN_TRENDLINE_BREAK_DISTANCE = 0.0


class _TrendlineTracker:
    """Shared bar-by-bar state machine for fitting/breaking the two Phase-2
    trendlines (support from confirmed swing lows, resistance from confirmed swing
    highs). Used by both detect_trendline and detect_channel (Phase 3) so the two
    can't silently drift onto different fit/break/invalidate semantics for what is,
    underneath, the exact same pair of lines - detect_channel's ascending channel
    literally reuses the support line as its lower boundary, and its descending
    channel reuses the resistance line as its upper boundary."""

    def __init__(self, swings: list[dict[str, Any]], right_bars: int, trendline_points: int):
        self.swings_sorted = sorted(swings, key=lambda s: s["index"])
        self.swing_ptr = 0
        self.right_bars = right_bars
        self.trendline_points = trendline_points
        self.low_points: list[tuple[int, float]] = []
        self.high_points: list[tuple[int, float]] = []
        self.support_line: tuple[float, float] | None = None
        self.resistance_line: tuple[float, float] | None = None

    def advance(self, i: int) -> None:
        """Confirms any swings newly available as of bar i (same right_bars lag as
        detect_swings/detect_bos_choch) and refits whichever line just gained a
        point, once that type has at least trendline_points confirmed swings."""
        while (
            self.swing_ptr < len(self.swings_sorted)
            and self.swings_sorted[self.swing_ptr]["index"] + self.right_bars <= i
        ):
            s = self.swings_sorted[self.swing_ptr]
            if s["type"] == "low":
                self.low_points.append((s["index"], s["price"]))
                if len(self.low_points) >= self.trendline_points:
                    fitted = _fit_line(self.low_points[-self.trendline_points:])
                    if fitted is not None:
                        self.support_line = fitted
            else:
                self.high_points.append((s["index"], s["price"]))
                if len(self.high_points) >= self.trendline_points:
                    fitted = _fit_line(self.high_points[-self.trendline_points:])
                    if fitted is not None:
                        self.resistance_line = fitted
            self.swing_ptr += 1

    def check_trendline_breaks(
        self, i: int, close: float, ts: str | None, min_break_distance: float,
    ) -> list[dict[str, Any]]:
        """Checks close against the current lines and invalidates (support_line/
        resistance_line -> None) whichever one broke. Two independent checks, not
        if/elif - see detect_trendline's docstring for why."""
        events: list[dict[str, Any]] = []
        if self.support_line is not None:
            slope, intercept = self.support_line
            projected = slope * i + intercept
            if close < projected - min_break_distance:
                events.append({
                    "type": "SUPPORT_BREAK", "direction": "bearish",
                    "price": float(projected), "index": i, "time": ts,
                })
                self.support_line = None
        if self.resistance_line is not None:
            slope, intercept = self.resistance_line
            projected = slope * i + intercept
            if close > projected + min_break_distance:
                events.append({
                    "type": "RESISTANCE_BREAK", "direction": "bullish",
                    "price": float(projected), "index": i, "time": ts,
                })
                self.resistance_line = None
        return events


def detect_trendline(
    window: pd.DataFrame,
    swings: list[dict[str, Any]],
    right_bars: int = 3,
    trendline_points: int = DEFAULT_TRENDLINE_POINTS,
    min_trendline_break_distance: float = DEFAULT_MIN_TRENDLINE_BREAK_DISTANCE,
) -> dict[str, Any]:
    """Real diagonal trendlines, replacing classify_swing_trend's 2-swing
    comparison (removed above - see this function's own module-level note).
    Fits two independent lines - a **support** line through confirmed swing LOWS
    (the uptrend line) and a **resistance** line through confirmed swing HIGHS
    (the downtrend line) - via least-squares over the most recent
    `trendline_points` confirmed swings of each type (see _fit_line and
    DEFAULT_TRENDLINE_POINTS above for the exact-math choice and why it subsumes
    the two-point-line alternative). Bar-by-bar, mirroring detect_bos_choch's
    structure closely (same confirmation-lag guard, same close-vs-wick discipline):

    1. Any swings newly confirmed as of bar `i` (`swing["index"] + right_bars <=
       i`, same lag as detect_swings/detect_bos_choch) are appended to a running,
       type-specific point list (`low_points` / `high_points`, chronological,
       never reset - see below). Once a type has at least `trendline_points`
       confirmed swings, the line for that type is (re)fit from the most recent
       `trendline_points` of them.
    2. The *current* fitted line (if any) is projected to bar `i`'s x-position
       (`slope * i + intercept`) and compared against `close` - **close, not the
       bar's high/low**, the same discipline detect_bos_choch's close-crossing
       phase uses, deliberately not "any wick touch breaks it."
       - Support line: `close < projected - min_trendline_break_distance` fires a
         `SUPPORT_BREAK` (`direction: "bearish"` - price broke down through the
         uptrend line) and invalidates the line (support_line = None) until
         enough new lows accumulate to refit.
       - Resistance line: `close > projected + min_trendline_break_distance`
         fires a `RESISTANCE_BREAK` (`direction: "bullish"`) and invalidates the
         line the same way.
       - These are two **independent** checks (not if/elif like
         detect_bos_choch's single pending_high/pending_low) since they concern
         two unrelated lines - both could in principle fire on the same bar.

    Deliberately **not** carried over from detect_bos_choch: there is no
    wick-based "supersession" phase (the mechanism that lets a newly-confirmed
    swing itself already being past the pending level fire an event even without
    a close-crossing). A continuous fitted line doesn't have a clean equivalent
    of "the new swing is already past the old level" - the projected line value
    changes continuously, not at discrete swing-confirmation points - so this is
    scoped to close-crossing only, a documented choice, not an oversight.

    `low_points`/`high_points` are **never reset** on a break - they keep
    accumulating every confirmed swing of that type for the life of the window,
    so the next confirmed swing after a break naturally triggers a refit from
    whatever's most recent, without needing separate "was there a break" state.

    Returns `{"support": {...} | None, "resistance": {...} | None, "breaks":
    [...]}`. `support`/`resistance`, when not None, are `{"slope", "intercept",
    "value_at_last_index"}` describing the line **as it stands at the end of the
    window** (None if invalidated by a break with nothing yet refit, or if fewer
    than `trendline_points` confirmed swings of that type ever existed). `breaks`
    is every SUPPORT_BREAK/RESISTANCE_BREAK event across the whole window,
    chronological, same shape as detect_bos_choch's events
    (`type`/`direction`/`price`/`index`/`time`)."""
    n = len(window)
    if not swings or n == 0:
        return {"support": None, "resistance": None, "breaks": []}

    tracker = _TrendlineTracker(swings, right_bars, trendline_points)
    breaks: list[dict[str, Any]] = []

    closes = window["close"].to_numpy(dtype=float)
    times = _all_time_strs(window)

    for i in range(n):
        tracker.advance(i)
        breaks.extend(tracker.check_trendline_breaks(i, closes[i], times[i], min_trendline_break_distance))

    def _line_dict(line: tuple[float, float] | None) -> dict[str, float] | None:
        if line is None:
            return None
        slope, intercept = line
        last_index = n - 1
        return {
            "slope": float(slope), "intercept": float(intercept),
            "value_at_last_index": float(slope * last_index + intercept),
        }

    return {
        "support": _line_dict(tracker.support_line),
        "resistance": _line_dict(tracker.resistance_line),
        "breaks": breaks,
    }


# TODO: no minimum break-distance filter existed before this parameter, same
# pattern as DEFAULT_MIN_TRENDLINE_BREAK_DISTANCE - 0.0 (off) is a placeholder for
# making the assumption visible/configurable, not a chosen value. Deliberately
# separate from min_trendline_break_distance even though both gate a "close beyond
# a line" check - the channel's constructed boundary and the underlying trendline
# are different lines with potentially different noise characteristics, so tying
# their thresholds together would be an unjustified assumption.
DEFAULT_MIN_CHANNEL_BREAK_DISTANCE = 0.0


def _fit_parallel_line(points: list[tuple[int, float]], slope: float) -> float | None:
    """Best-fit intercept (least-squares, slope held fixed) for a line through
    points: minimizes sum((y - (slope*x + b))^2) over b, solved directly by
    b = mean(y - slope*x). Used to connect the corresponding highs/lows to a
    channel boundary parallel to an already-fit trendline, rather than anchoring
    the boundary on a single most-recent touch point - the same least-squares-
    over-multiple-points preference detect_trendline itself makes (see
    DEFAULT_TRENDLINE_POINTS), applied here with the slope constrained instead of
    independently fit. Returns None only if points is empty."""
    if not points:
        return None
    offsets = [y - slope * x for x, y in points]
    return sum(offsets) / len(offsets)


def _channel_bar_events(
    channel: str, i: int, ts: str | None, close: float, high: float, low: float,
    lower_val: float, upper_val: float, min_channel_break_distance: float,
) -> list[dict[str, Any]]:
    """Per-bar touch/break check against one channel's two boundaries. A CLOSE
    beyond a boundary is a CHANNEL_BREAK (higher-conviction reversal signal, per
    the rebuild decision); short of that, a WICK (high/low) reaching the boundary
    without closing past it is the weaker CHANNEL_TOUCH (potential reversal) -
    break takes priority over touch on the same bar/boundary, they're not both
    reported for the same event. Direction convention (a judgment call, since
    the video doesn't specify labels for these): breaking the upper boundary is
    "bullish" (price accelerated through the ceiling) and breaking the lower is
    "bearish" (through the floor) - the same convention detect_trendline's
    RESISTANCE_BREAK/SUPPORT_BREAK already use, since an ascending channel's lower
    boundary literally *is* the support line and a descending channel's upper
    boundary literally *is* the resistance line. A touch carries the *opposite*
    direction from a break at the same boundary - touching the upper boundary
    from inside the channel is a potential rejection *down* ("bearish"), touching
    the lower boundary is a potential bounce *up* ("bullish")."""
    events: list[dict[str, Any]] = []

    if close > upper_val + min_channel_break_distance:
        events.append({
            "type": "CHANNEL_BREAK", "channel": channel, "boundary": "upper",
            "direction": "bullish", "price": float(upper_val), "index": i, "time": ts,
        })
    elif high >= upper_val:
        events.append({
            "type": "CHANNEL_TOUCH", "channel": channel, "boundary": "upper",
            "direction": "bearish", "price": float(upper_val), "index": i, "time": ts,
        })

    if close < lower_val - min_channel_break_distance:
        events.append({
            "type": "CHANNEL_BREAK", "channel": channel, "boundary": "lower",
            "direction": "bearish", "price": float(lower_val), "index": i, "time": ts,
        })
    elif low <= lower_val:
        events.append({
            "type": "CHANNEL_TOUCH", "channel": channel, "boundary": "lower",
            "direction": "bullish", "price": float(lower_val), "index": i, "time": ts,
        })

    return events


def detect_channel(
    window: pd.DataFrame,
    swings: list[dict[str, Any]],
    right_bars: int = 3,
    trendline_points: int = DEFAULT_TRENDLINE_POINTS,
    min_break_distance: float = DEFAULT_MIN_TRENDLINE_BREAK_DISTANCE,
    min_channel_break_distance: float = DEFAULT_MIN_CHANNEL_BREAK_DISTANCE,
) -> dict[str, Any]:
    """Parallel channel boundaries around the Phase 2 trendlines - Phase 3 of the
    2026-09 detector rebuild. "A parallel line to the trendline, connecting the
    corresponding highs/lows, same slope" (the rebuild decision's own phrasing),
    built two ways depending on which Phase-2 line is currently active:

    - **Ascending channel**: anchored on the *support* line (fit through swing
      lows) as its **lower** boundary - its **upper** boundary is a new line with
      the *same slope*, best-fit (via `_fit_parallel_line`) through the most
      recent `trendline_points` confirmed swing **highs**.
    - **Descending channel**: anchored on the *resistance* line (fit through
      swing highs) as its **upper** boundary - its **lower** boundary is fit the
      same way through the most recent `trendline_points` confirmed swing
      **lows**.

    Both are tracked simultaneously and independently (there's no "current trend
    direction" concept left to pick one - classify_swing_trend was removed, not
    replaced with an equivalent), using the exact same `_TrendlineTracker` state
    machine detect_trendline uses, so the two functions can't drift onto
    different ideas of what the "current" support/resistance line is - the
    ascending channel's lower boundary and detect_trendline's support line are,
    numerically, the identical line.

    Per bar `i` (after `tracker.advance(i)` confirms/refits same as
    detect_trendline): if the support line exists **and** at least
    `trendline_points` swing highs have been confirmed, build the ascending
    channel's two boundaries at this bar's x-position and check both for
    touch/break (`_channel_bar_events`); symmetrically for the descending
    channel. `tracker.check_trendline_breaks` still runs every bar (using
    `min_break_distance`, the same parameter detect_trendline takes) so the
    underlying support/resistance lines invalidate at the identical bars they
    would in detect_trendline - a channel's boundary silently goes stale
    otherwise. Trendline-native SUPPORT_BREAK/RESISTANCE_BREAK events themselves
    are **not** duplicated into this function's own `events` list - that break is
    already exactly the ascending channel's lower-boundary CHANNEL_BREAK (or the
    descending channel's upper-boundary one), reported once via the mechanism
    above, not reported again under a different name.

    Returns `{"ascending": {...} | None, "descending": {...} | None, "events":
    [...]}`. Each channel dict, when not None, is `{"slope", "lower": {
    "intercept", "value_at_last_index"}, "upper": {"intercept",
    "value_at_last_index"}}` describing both boundaries as they stand at the end
    of the window (None if the anchor line is currently invalidated, or if the
    *other* side never accumulated `trendline_points` confirmed swings).
    `events` is every CHANNEL_TOUCH/CHANNEL_BREAK across the whole window,
    chronological - see `_channel_bar_events` for the field shape and the
    touch-vs-break/direction conventions."""
    n = len(window)
    if not swings or n == 0:
        return {"ascending": None, "descending": None, "events": []}

    tracker = _TrendlineTracker(swings, right_bars, trendline_points)
    events: list[dict[str, Any]] = []

    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    closes = window["close"].to_numpy(dtype=float)
    times = _all_time_strs(window)

    for i in range(n):
        tracker.advance(i)

        if tracker.support_line is not None and len(tracker.high_points) >= trendline_points:
            slope, lower_intercept = tracker.support_line
            upper_intercept = _fit_parallel_line(tracker.high_points[-trendline_points:], slope)
            if upper_intercept is not None:
                events.extend(_channel_bar_events(
                    "ascending", i, times[i], closes[i], highs[i], lows[i],
                    slope * i + lower_intercept, slope * i + upper_intercept,
                    min_channel_break_distance,
                ))

        if tracker.resistance_line is not None and len(tracker.low_points) >= trendline_points:
            slope, upper_intercept = tracker.resistance_line
            lower_intercept = _fit_parallel_line(tracker.low_points[-trendline_points:], slope)
            if lower_intercept is not None:
                events.extend(_channel_bar_events(
                    "descending", i, times[i], closes[i], highs[i], lows[i],
                    slope * i + lower_intercept, slope * i + upper_intercept,
                    min_channel_break_distance,
                ))

        tracker.check_trendline_breaks(i, closes[i], times[i], min_break_distance)

    def _channel_dict(
        anchor_line: tuple[float, float] | None, other_points: list[tuple[int, float]], anchor_is_lower: bool,
    ) -> dict[str, Any] | None:
        if anchor_line is None or len(other_points) < trendline_points:
            return None
        slope, anchor_intercept = anchor_line
        other_intercept = _fit_parallel_line(other_points[-trendline_points:], slope)
        if other_intercept is None:
            return None
        lower_intercept = anchor_intercept if anchor_is_lower else other_intercept
        upper_intercept = other_intercept if anchor_is_lower else anchor_intercept
        last_index = n - 1
        return {
            "slope": float(slope),
            "lower": {
                "intercept": float(lower_intercept),
                "value_at_last_index": float(slope * last_index + lower_intercept),
            },
            "upper": {
                "intercept": float(upper_intercept),
                "value_at_last_index": float(slope * last_index + upper_intercept),
            },
        }

    return {
        "ascending": _channel_dict(tracker.support_line, tracker.high_points, anchor_is_lower=True),
        "descending": _channel_dict(tracker.resistance_line, tracker.low_points, anchor_is_lower=False),
        "events": events,
    }


# TODO: no minimum break-distance filter existed before this parameter was added -
# any close beyond the level by any amount, even a single tick, counted as a break
# (see docs/detector-logic.md's "BOS / CHoCH" section). 0.0 (off) reproduces that
# exact prior behavior byte-for-byte; a real nonzero value needs its own validation
# pass for 5-minute BTC/USD bars before being trusted - this is a placeholder for
# making the assumption visible/configurable, not a chosen value. Unlike the OB/FVG
# filters above, this is a raw price distance, not an ATR ratio - that's what was
# asked for; nothing stops a future pass from switching it to a ratio too.
DEFAULT_MIN_BREAK_DISTANCE = 0.0


def detect_bos_choch(
    window: pd.DataFrame,
    swings: list[dict[str, Any]],
    right_bars: int = 3,
    min_break_distance: float = DEFAULT_MIN_BREAK_DISTANCE,
) -> list[dict[str, Any]]:
    """BOS (Break of Structure, trend continuation) / CHoCH (Change of Character,
    trend reversal). Treats the prior swing high/low as a "structure level" and fires
    an event the moment close crosses it by at least min_break_distance:
      - Bullish structure, close breaks above prior swing high -> BOS (bullish continues)
      - Bullish structure, close breaks below prior swing low  -> CHoCH (turns bearish)
      - Bearish structure, close breaks below prior swing low  -> BOS (bearish continues)
      - Bearish structure, close breaks above prior swing high -> CHoCH (turns bullish)
    A broken level is consumed; the next check only uses swings confirmed after it.

    min_break_distance applies only to this close-crossing check, not to the
    wick-based swing-supersession check below (see the "Bug fix" paragraph) - that
    mechanism fires on a *new swing* already being past the old level, which is a
    different kind of break (confirmed by wick, not by this bar's close) and wasn't
    part of what this parameter was asked to gate. A nonzero min_break_distance can
    therefore still be circumvented via that path - a documented scoping choice, not
    an oversight.

    Look-ahead bias guard: a swing at index i isn't confirmed until i+right_bars (see
    detect_swings) - levels are only "known" from that point on.

    Bug fix (carried over from tradingagents-kr, 2026-08-22): swings are confirmed by
    wick (high/low), not close. If a newly-confirmed swing has already gone past the
    still-unbroken pending level - even though close never actually crossed it - the
    structure has in fact already broken. Silently swapping in the tighter level
    without firing an event would leave bias stuck on a stale value. So a new swing
    that's already past the pending level fires the event before replacing it; a swing
    that only tightens the level (higher low in an uptrend, lower high in a downtrend)
    is structure continuation, not a break, and updates the level with no event.
    """
    events: list[dict[str, Any]] = []
    if not swings:
        return events

    swings_sorted = sorted(swings, key=lambda s: s["index"])
    swing_ptr = 0
    pending_high: dict[str, Any] | None = None
    pending_low: dict[str, Any] | None = None
    bias: str | None = None

    closes = window["close"].to_numpy(dtype=float)
    times = _all_time_strs(window)

    for i in range(len(window)):
        while swing_ptr < len(swings_sorted) and swings_sorted[swing_ptr]["index"] + right_bars <= i:
            s = swings_sorted[swing_ptr]
            if s["type"] == "high":
                if pending_high is not None and s["price"] > pending_high["price"]:
                    event_type = "BOS" if bias == "bullish" else "CHoCH"
                    events.append({
                        "type": event_type, "direction": "bullish", "price": pending_high["price"],
                        "index": s["index"], "time": s["time"],
                    })
                    bias = "bullish"
                pending_high = s
            else:
                if pending_low is not None and s["price"] < pending_low["price"]:
                    event_type = "BOS" if bias == "bearish" else "CHoCH"
                    events.append({
                        "type": event_type, "direction": "bearish", "price": pending_low["price"],
                        "index": s["index"], "time": s["time"],
                    })
                    bias = "bearish"
                pending_low = s
            swing_ptr += 1

        close = closes[i]
        ts = times[i]

        if pending_high is not None and close > pending_high["price"] + min_break_distance:
            event_type = "BOS" if bias == "bullish" else "CHoCH"
            events.append({
                "type": event_type, "direction": "bullish", "price": pending_high["price"],
                "index": i, "time": ts,
            })
            bias = "bullish"
            pending_high = None
        elif pending_low is not None and close < pending_low["price"] - min_break_distance:
            event_type = "BOS" if bias == "bearish" else "CHoCH"
            events.append({
                "type": event_type, "direction": "bearish", "price": pending_low["price"],
                "index": i, "time": ts,
            })
            bias = "bearish"
            pending_low = None

    return events


# TODO: "nearby" for a trap's second retest extreme isn't given exact math by the
# reference video. Unlike every other break-distance parameter in this file (which
# gate a MINIMUM distance and default to 0.0/"off"), this one gates a MAXIMUM
# distance - "off" (no filter, accept any second extreme as a valid retest) means
# infinity, not zero; 0.0 here would require the second extreme to land at exactly
# the same price as the first, which would essentially never fire. Deliberately not
# validated against 5-minute BTC/USD bars yet - same placeholder treatment as every
# other TODO default in this file, just inverted because this default's "off" value
# is a maximum, not a minimum.
DEFAULT_MAX_TRAP_RETEST_DISTANCE = float("inf")


def _detect_trap(
    window: pd.DataFrame,
    swings: list[dict[str, Any]],
    trendline_breaks: list[dict[str, Any]],
    max_trap_retest_distance: float,
) -> dict[str, Any] | None:
    """Trap pattern (new, 2026-09 rebuild Phase 4): a level breaks, price forms a
    first extreme, bounces away from it, forms a second nearby extreme (a failed
    retest), then reclaims back past the broken level - the rebuild decision's own
    phrasing, first extreme as the stop-loss reference.

    **Scoped to Phase 2 trendline breaks specifically** - not also the raw swing
    level or the Phase 3 channel-derived boundaries the classic fakeout check in
    detect_fakeout() below additionally considers. Unifying three different break
    sources into one unambiguous "which level broke, for stop-placement purposes"
    answer needs its own design pass; this is a documented scope-narrowing for the
    trap state machine specifically, not an oversight - the classic fakeout check
    still gets the full generalization the rebuild asked for.

    State machine, run once per breaks in `trendline_breaks` (each independent -
    a SUPPORT_BREAK, bearish direction, seeds a "SUPPORT_TRAP" attempt; a
    RESISTANCE_BREAK, bullish direction, seeds a "RESISTANCE_TRAP" attempt):

    1. `broken_level` = the break event's `price`.
    2. **First extreme (E1)**: the next confirmed swing of the matching type
       after the break bar (a swing LOW after a SUPPORT_BREAK, a swing HIGH
       after a RESISTANCE_BREAK). This becomes the trap's `stop_loss_reference`.
       No such swing yet -> this break hasn't produced a trap (yet).
    3. **Bounce**: the next confirmed swing of the OPPOSITE type after E1 -
       confirms price actually moved away from E1 rather than grinding straight
       through it.
    4. **Second extreme (E2)**: the next confirmed swing of the SAME type as E1,
       after the bounce. Must be "nearby" E1 - within `max_trap_retest_distance`
       - unless E2 is actually *inside* E1 (a higher low after a SUPPORT_BREAK, a
       lower high after a RESISTANCE_BREAK), which is always accepted regardless
       of distance (that's a shallower retest, not a further breakdown). If E2
       instead *extends past* E1 (a lower low / higher high) by more than
       `max_trap_retest_distance`, the setup is abandoned - that reads as
       continuation, not a failed retest.
    5. **Reclaim**: the first bar at or after E2 whose `close` lands back on the
       original side of `broken_level`. Only reported when that first-reclaiming
       bar is the *last* bar in `window` - mirroring detect_fakeout's own "only
       checks the most recent candle" scoping, so this function always answers
       "did something confirm right now," not "did something ever confirm
       somewhere in this lookback window."

    Returns `{"type": "SUPPORT_TRAP" | "RESISTANCE_TRAP", "direction": "bullish"
    | "bearish", "broken_level": float, "stop_loss_reference": float (E1's
    price), "first_extreme_index": int, "index": int, "time": str | None}` for
    whichever break (if any) completes its full 5-step sequence with its reclaim
    landing exactly on the last bar, or `None`. Naming note: deliberately
    "SUPPORT_TRAP"/"RESISTANCE_TRAP" (named after which level broke), not the
    trader-jargon "bull trap"/"bear trap" - those terms are named after which
    side gets trapped, which is the *opposite* of the break direction, and
    mixing the two naming conventions invites exactly the kind of mislabeling
    bug this project has already hit once (see detect_bos_choch's BOS/CHoCH
    labeling divergence note)."""
    n = len(window)
    if not trendline_breaks:
        return None

    closes = window["close"].to_numpy(dtype=float)
    times = _all_time_strs(window)
    swings_sorted = sorted(swings, key=lambda s: s["index"])
    last_index = n - 1

    for brk in trendline_breaks:
        break_index = brk["index"]
        broken_level = brk["price"]
        is_support_break = brk["type"] == "SUPPORT_BREAK"
        extreme_type = "low" if is_support_break else "high"
        bounce_type = "high" if is_support_break else "low"

        after_break = [s for s in swings_sorted if s["index"] > break_index]
        e1 = next((s for s in after_break if s["type"] == extreme_type), None)
        if e1 is None:
            continue

        after_e1 = [s for s in after_break if s["index"] > e1["index"]]
        bounce = next((s for s in after_e1 if s["type"] == bounce_type), None)
        if bounce is None:
            continue

        after_bounce = [s for s in after_e1 if s["index"] > bounce["index"]]
        e2 = next((s for s in after_bounce if s["type"] == extreme_type), None)
        if e2 is None:
            continue

        extends_past_e1 = (e2["price"] < e1["price"]) if is_support_break else (e2["price"] > e1["price"])
        if extends_past_e1 and abs(e2["price"] - e1["price"]) > max_trap_retest_distance:
            continue  # further continuation past E1, not a nearby failed retest

        reclaim_index = None
        for i in range(e2["index"], n):
            close = closes[i]
            if is_support_break and close > broken_level:
                reclaim_index = i
                break
            if not is_support_break and close < broken_level:
                reclaim_index = i
                break
        if reclaim_index != last_index:
            continue  # either never reclaimed, or reclaimed on a now-stale bar

        return {
            "type": "SUPPORT_TRAP" if is_support_break else "RESISTANCE_TRAP",
            "direction": "bullish" if is_support_break else "bearish",
            "broken_level": float(broken_level),
            "stop_loss_reference": float(e1["price"]),
            "first_extreme_index": e1["index"],
            "index": reclaim_index,
            "time": times[reclaim_index],
        }

    return None


def detect_fakeout(
    window: pd.DataFrame,
    swings: list[dict[str, Any]],
    right_bars: int = 3,
    trendline_points: int = DEFAULT_TRENDLINE_POINTS,
    max_trap_retest_distance: float = DEFAULT_MAX_TRAP_RETEST_DISTANCE,
    trendline: dict[str, Any] | None = None,
    channel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Liquidity sweep / fakeout (existing concept, generalized) plus trap
    detection (new) - Phase 4 of the 2026-09 detector rebuild. Returns
    `{"fakeout": {...} | None, "trap": {...} | None}` - a **breaking output-shape
    change** from the prior single-dict-or-None return (this module's docstring
    already documents this rebuild as behavior-change-by-design, not
    bit-identical-preserving).

    **Fakeout, generalized beyond the single most-recent raw swing level.**
    Still only examines the last two candles (`curr = window.iloc[-1]`, `prev =
    window.iloc[-2]`) - that structural "2 candles, no configurable window"
    constraint is unchanged. What changed is *which levels* count: previously
    only the single most recent confirmed swing low/high; now also the current
    Phase 2 trendline (`detect_trendline`'s `support`/`resistance`, evaluated at
    the last bar) and the current Phase 3 channel's *derived* boundaries (the
    ascending channel's upper boundary, the descending channel's lower boundary
    - the two that AREN'T already identical to the trendline values). Checked in
    a fixed priority order - swing, then trendline, then channel - and the first
    level that matches wins; this is a deliberate, documented tie-break (multiple
    levels often sit close together and could all match the same 2-candle sweep),
    not an attempt to report all of them.

    **Role change - the single biggest behavioral change in this rebuild.**
    Before this rebuild, `signals/smc_aggregator.py` used this detector's output
    **only as a veto**: a fakeout opposing an already-tentative direction forced
    that direction to `NEUTRAL`, and a fakeout *agreeing* with the tentative
    direction did nothing at all - not a vote, not a confirmation. Phase 5 (the
    aggregator rewire) makes fakeout/trap a **genuine fifth confluence vote**,
    on equal footing with order block / FVG / trendline / channel - it can now
    independently push the confluence score toward BULLISH or BEARISH on its
    own, not just cancel an existing call. This function's job is unchanged
    (detection only, no aggregation logic here), but what compute_smc_features'
    caller *does* with this output is fundamentally different starting Phase 5 -
    see signals/smc_aggregator.py and its own module docstring for the new
    aggregation model once that phase lands.

    See `_detect_trap` above for the trap pattern's full state-machine
    documentation, including why it's scoped to trendline breaks specifically
    rather than the same 3-source generalization the fakeout check above gets.

    `trendline`/`channel`: precomputed `detect_trendline`/`detect_channel`
    results, or `None` to compute them here. `compute_smc_features` passes its
    own already-computed values in - same shared-computation pattern
    `detect_order_blocks`/`detect_fvg` use for `atr` - since without it,
    `compute_smc_features` would end up building 4 `_TrendlineTracker`
    instances per call (2 direct + 2 more inside this function) instead of 2."""
    result: dict[str, Any] = {"fakeout": None, "trap": None}
    n = len(window)
    if n < 3 or not swings:
        return result

    if trendline is None:
        trendline = detect_trendline(window, swings, right_bars=right_bars, trendline_points=trendline_points)
    if channel is None:
        channel = detect_channel(window, swings, right_bars=right_bars, trendline_points=trendline_points)

    curr = window.iloc[-1]
    prev = window.iloc[-2]
    i = n - 1
    ts = _time_str(window, i)

    support_levels: list[tuple[str, float]] = []
    recent_lows = [s for s in swings if s["type"] == "low"]
    if recent_lows:
        support_levels.append(("swing", recent_lows[-1]["price"]))
    if trendline["support"] is not None:
        support_levels.append(("trendline", trendline["support"]["value_at_last_index"]))
    if channel["descending"] is not None:
        support_levels.append(("channel", channel["descending"]["lower"]["value_at_last_index"]))

    resistance_levels: list[tuple[str, float]] = []
    recent_highs = [s for s in swings if s["type"] == "high"]
    if recent_highs:
        resistance_levels.append(("swing", recent_highs[-1]["price"]))
    if trendline["resistance"] is not None:
        resistance_levels.append(("trendline", trendline["resistance"]["value_at_last_index"]))
    if channel["ascending"] is not None:
        resistance_levels.append(("channel", channel["ascending"]["upper"]["value_at_last_index"]))

    for source, level in support_levels:
        if float(prev["low"]) < level and float(curr["close"]) > level:
            result["fakeout"] = {
                "type": "BULL_FAKEOUT", "source": source, "swept_level": float(level),
                "index": i, "time": ts,
            }
            break

    if result["fakeout"] is None:
        for source, level in resistance_levels:
            if float(prev["high"]) > level and float(curr["close"]) < level:
                result["fakeout"] = {
                    "type": "BEAR_FAKEOUT", "source": source, "swept_level": float(level),
                    "index": i, "time": ts,
                }
                break

    result["trap"] = _detect_trap(window, swings, trendline["breaks"], max_trap_retest_distance)
    return result


# TODO: 90 was tuned for daily stock bars (~4 months of history) in tradingagents-kr.
# For 5-minute crypto bars this needs its own validation pass before trusting it as a
# real default - left as-is only as a functional placeholder, not a chosen value.
DEFAULT_LOOKBACK_BARS = 90


def compute_smc_features(
    ohlcv: pd.DataFrame,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    swing_left_bars: int = 3,
    swing_right_bars: int = 3,
    min_ob_body_ratio: float = DEFAULT_MIN_OB_BODY_RATIO,
    min_fvg_gap_ratio: float = DEFAULT_MIN_FVG_GAP_RATIO,
    min_break_distance: float = DEFAULT_MIN_BREAK_DISTANCE,
    trendline_points: int = DEFAULT_TRENDLINE_POINTS,
    min_trendline_break_distance: float = DEFAULT_MIN_TRENDLINE_BREAK_DISTANCE,
    min_channel_break_distance: float = DEFAULT_MIN_CHANNEL_BREAK_DISTANCE,
    max_trap_retest_distance: float = DEFAULT_MAX_TRAP_RETEST_DISTANCE,
) -> dict[str, Any]:
    """OHLCV DataFrame -> SMC structure feature dict. Pure computation, no
    buy/sell judgment (see signals/smc_aggregator.py for that)."""
    if ohlcv.empty:
        return {}

    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"OHLCV missing required columns. Need: {required}, have: {set(df.columns)}")

    window = df.tail(lookback_bars)

    # Computed once and shared: detect_order_blocks and detect_fvg both need the
    # identical ATR series over this same window (see their docstrings) - passing
    # it in avoids computing it twice for no benefit.
    atr = _atr(window)
    order_blocks = detect_order_blocks(window, min_ob_body_ratio=min_ob_body_ratio, atr=atr)
    fvgs = detect_fvg(window, min_fvg_gap_ratio=min_fvg_gap_ratio, atr=atr)
    swings = detect_swings(window, left_bars=swing_left_bars, right_bars=swing_right_bars)
    structure_events = detect_bos_choch(
        window, swings, right_bars=swing_right_bars, min_break_distance=min_break_distance,
    )
    # detect_trendline and detect_channel each build their own _TrendlineTracker
    # over the same swings list and parameters - a known, deliberate duplication
    # of the confirm/refit bar loop (same category as the pre-shared-ATR
    # duplication _atr's own sharing above was written to avoid), left as-is for
    # now since Phase 2/3 aren't on the tuning grid's hot path yet (the grid isn't
    # being re-run as part of this rebuild - see docs/detector-logic.md). Worth
    # revisiting the same way if/when this path gets tuned-grid-hot again.
    trendline = detect_trendline(
        window, swings, right_bars=swing_right_bars, trendline_points=trendline_points,
        min_trendline_break_distance=min_trendline_break_distance,
    )
    channel = detect_channel(
        window, swings, right_bars=swing_right_bars, trendline_points=trendline_points,
        min_break_distance=min_trendline_break_distance,
        min_channel_break_distance=min_channel_break_distance,
    )
    fakeout = detect_fakeout(
        window, swings, right_bars=swing_right_bars, trendline_points=trendline_points,
        max_trap_retest_distance=max_trap_retest_distance, trendline=trendline, channel=channel,
    )

    return {
        "lookback_bars": len(window),
        "order_blocks": order_blocks,
        "fvgs": fvgs,
        "swings": swings,
        "trendline": trendline,
        "channel": channel,
        "structure_events": structure_events,
        "structure_bias": structure_events[-1]["direction"] if structure_events else None,
        "fakeout": fakeout,
    }
