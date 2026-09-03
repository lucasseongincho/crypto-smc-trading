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

    swings_sorted = sorted(swings, key=lambda s: s["index"])
    swing_ptr = 0
    low_points: list[tuple[int, float]] = []
    high_points: list[tuple[int, float]] = []
    support_line: tuple[float, float] | None = None
    resistance_line: tuple[float, float] | None = None
    breaks: list[dict[str, Any]] = []

    closes = window["close"].to_numpy(dtype=float)
    times = _all_time_strs(window)

    for i in range(n):
        while swing_ptr < len(swings_sorted) and swings_sorted[swing_ptr]["index"] + right_bars <= i:
            s = swings_sorted[swing_ptr]
            if s["type"] == "low":
                low_points.append((s["index"], s["price"]))
                if len(low_points) >= trendline_points:
                    fitted = _fit_line(low_points[-trendline_points:])
                    if fitted is not None:
                        support_line = fitted
            else:
                high_points.append((s["index"], s["price"]))
                if len(high_points) >= trendline_points:
                    fitted = _fit_line(high_points[-trendline_points:])
                    if fitted is not None:
                        resistance_line = fitted
            swing_ptr += 1

        close = closes[i]
        ts = times[i]

        if support_line is not None:
            slope, intercept = support_line
            projected = slope * i + intercept
            if close < projected - min_trendline_break_distance:
                breaks.append({
                    "type": "SUPPORT_BREAK", "direction": "bearish",
                    "price": float(projected), "index": i, "time": ts,
                })
                support_line = None

        if resistance_line is not None:
            slope, intercept = resistance_line
            projected = slope * i + intercept
            if close > projected + min_trendline_break_distance:
                breaks.append({
                    "type": "RESISTANCE_BREAK", "direction": "bullish",
                    "price": float(projected), "index": i, "time": ts,
                })
                resistance_line = None

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
        "support": _line_dict(support_line),
        "resistance": _line_dict(resistance_line),
        "breaks": breaks,
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


def detect_fakeout(window: pd.DataFrame, swings: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Liquidity sweep: a wick past the last swing that closes back on the other side.
    Only checks the most recent candle. Detection only - signals/smc_aggregator.py
    decides whether/how to act on it (a veto, not a bonus point)."""
    if len(window) < 3 or not swings:
        return None

    curr = window.iloc[-1]
    prev = window.iloc[-2]
    i = len(window) - 1
    ts = _time_str(window, i)

    recent_lows = [s for s in swings if s["type"] == "low"]
    if recent_lows:
        last_low = recent_lows[-1]["price"]
        if float(prev["low"]) < last_low and float(curr["close"]) > last_low:
            return {"type": "BULL_FAKEOUT", "swept_level": last_low, "index": i, "time": ts}

    recent_highs = [s for s in swings if s["type"] == "high"]
    if recent_highs:
        last_high = recent_highs[-1]["price"]
        if float(prev["high"]) > last_high and float(curr["close"]) < last_high:
            return {"type": "BEAR_FAKEOUT", "swept_level": last_high, "index": i, "time": ts}

    return None


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
    trendline = detect_trendline(
        window, swings, right_bars=swing_right_bars, trendline_points=trendline_points,
        min_trendline_break_distance=min_trendline_break_distance,
    )
    fakeout = detect_fakeout(window, swings)

    return {
        "lookback_bars": len(window),
        "order_blocks": order_blocks,
        "fvgs": fvgs,
        "swings": swings,
        "trendline": trendline,
        "structure_events": structure_events,
        "structure_bias": structure_events[-1]["direction"] if structure_events else None,
        "fakeout": fakeout,
    }
