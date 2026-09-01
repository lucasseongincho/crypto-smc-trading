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
    """window.index[i] -> ISO timestamp string, or None if the index isn't datetime-like."""
    try:
        return pd.Timestamp(window.index[i]).isoformat()
    except Exception:
        return None


def detect_order_blocks(window: pd.DataFrame) -> list[dict[str, Any]]:
    """Order Block: a candle that gets strongly overrun (engulfed) by the next candle
    in the opposite direction."""
    ob_list: list[dict[str, Any]] = []
    if len(window) < 2:
        return ob_list

    for i in range(1, len(window)):
        prev = window.iloc[i - 1]
        curr = window.iloc[i]
        ts = _time_str(window, i)

        if float(prev["close"]) < float(prev["open"]) and float(curr["close"]) > float(curr["open"]):
            if float(curr["close"]) > float(prev["high"]):
                ob_list.append({
                    "type": "bullish", "high": float(prev["high"]), "low": float(prev["low"]),
                    "index": i, "time": ts,
                })
        elif float(prev["close"]) > float(prev["open"]) and float(curr["close"]) < float(curr["open"]):
            if float(curr["close"]) < float(prev["low"]):
                ob_list.append({
                    "type": "bearish", "high": float(prev["high"]), "low": float(prev["low"]),
                    "index": i, "time": ts,
                })

    return ob_list


def detect_fvg(window: pd.DataFrame) -> list[dict[str, Any]]:
    """Fair Value Gap: a 3-candle price gap where the middle candle's range is skipped."""
    fvg_list: list[dict[str, Any]] = []
    if len(window) < 3:
        return fvg_list

    for i in range(2, len(window)):
        candle_1 = window.iloc[i - 2]
        candle_3 = window.iloc[i]
        ts = _time_str(window, i)

        if float(candle_3["low"]) > float(candle_1["high"]):
            fvg_list.append({
                "type": "bullish", "top": float(candle_3["low"]), "bottom": float(candle_1["high"]),
                "index": i, "time": ts,
            })
        elif float(candle_3["high"]) < float(candle_1["low"]):
            fvg_list.append({
                "type": "bearish", "top": float(candle_1["low"]), "bottom": float(candle_3["high"]),
                "index": i, "time": ts,
            })

    return fvg_list


def detect_swings(window: pd.DataFrame, left_bars: int = 3, right_bars: int = 3) -> list[dict[str, Any]]:
    """Fractal-style swing high/low detection. A swing at index i isn't confirmable
    until right_bars candles later - detect_bos_choch() accounts for that lag to avoid
    look-ahead bias."""
    swings: list[dict[str, Any]] = []
    if len(window) < left_bars + right_bars + 1:
        return swings

    for i in range(left_bars, len(window) - right_bars):
        curr = window.iloc[i]
        ts = _time_str(window, i)

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


def classify_swing_trend(lows: list[dict[str, Any]], highs: list[dict[str, Any]]) -> str:
    """UPTREND/DOWNTREND/RANGING from the last 2 swing highs/lows (higher-high+higher-low,
    or lower-high+lower-low). Despite the name this doesn't draw an actual diagonal
    trendline - it's a 2-swing comparison classifier, same as the original."""
    if len(lows) < 2 or len(highs) < 2:
        return "RANGING"

    hh = highs[-1]["price"] > highs[-2]["price"]
    hl = lows[-1]["price"] > lows[-2]["price"]
    lh = highs[-1]["price"] < highs[-2]["price"]
    ll = lows[-1]["price"] < lows[-2]["price"]

    if hh and hl:
        return "UPTREND"
    if lh and ll:
        return "DOWNTREND"
    return "RANGING"


def detect_bos_choch(
    window: pd.DataFrame,
    swings: list[dict[str, Any]],
    right_bars: int = 3,
) -> list[dict[str, Any]]:
    """BOS (Break of Structure, trend continuation) / CHoCH (Change of Character,
    trend reversal). Treats the prior swing high/low as a "structure level" and fires
    an event the moment close crosses it:
      - Bullish structure, close breaks above prior swing high -> BOS (bullish continues)
      - Bullish structure, close breaks below prior swing low  -> CHoCH (turns bearish)
      - Bearish structure, close breaks below prior swing low  -> BOS (bearish continues)
      - Bearish structure, close breaks above prior swing high -> CHoCH (turns bullish)
    A broken level is consumed; the next check only uses swings confirmed after it.

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

        close = float(window.iloc[i]["close"])
        ts = _time_str(window, i)

        if pending_high is not None and close > pending_high["price"]:
            event_type = "BOS" if bias == "bullish" else "CHoCH"
            events.append({
                "type": event_type, "direction": "bullish", "price": pending_high["price"],
                "index": i, "time": ts,
            })
            bias = "bullish"
            pending_high = None
        elif pending_low is not None and close < pending_low["price"]:
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

    order_blocks = detect_order_blocks(window)
    fvgs = detect_fvg(window)
    swings = detect_swings(window, left_bars=swing_left_bars, right_bars=swing_right_bars)
    lows = [s for s in swings if s["type"] == "low"]
    highs = [s for s in swings if s["type"] == "high"]
    swing_trend = classify_swing_trend(lows, highs)
    structure_events = detect_bos_choch(window, swings, right_bars=swing_right_bars)
    fakeout = detect_fakeout(window, swings)

    return {
        "lookback_bars": len(window),
        "order_blocks": order_blocks,
        "fvgs": fvgs,
        "swings": swings,
        "swing_trend": swing_trend,
        "structure_events": structure_events,
        "structure_bias": structure_events[-1]["direction"] if structure_events else None,
        "fakeout": fakeout,
    }
