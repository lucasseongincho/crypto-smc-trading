# SMC detector logic, as actually implemented

This documents exactly what `data/smc.py` checks and how `signals/smc_aggregator.py`
uses it — against the real code, not the generic textbook description of SMC. Written
because several of these diverge from the textbook version in ways that matter for
interpreting backtest results, and "I'll remember" isn't a substitute for a written
record.

All line numbers below are against `data/smc.py` and `signals/smc_aggregator.py` as of
this commit.

**Headline finding**: "liquidity sweep" and "fakeout" are not two detectors — there is
one function, `detect_fakeout`, and it is not implemented as "a BOS/CHoCH that got
invalidated." See [Liquidity sweep / fakeout](#liquidity-sweep--fakeout-one-detector-not-two)
below.

**Second headline finding**: order block validity has **no** BOS/CHoCH gating, no
mitigation tracking, and no size/volume filter anywhere in this codebase — not in the
detector, not in the aggregator, not in the one other place `order_blocks` gets read
(`backtest/runner.py`'s stop-loss placement). See
[Order block](#order-block-detect_order_blocks-lines-26-51).

---

## Swing detection (`detect_swings`, lines 79-117)

**Literal condition.** For each candidate index `i` (only indices with at least
`left_bars` candles before and `right_bars` candles after are even considered — the
loop is `for i in range(left_bars, len(window) - right_bars)`):

- `is_high` starts `True`. For every `j` in `1..left_bars`: if
  `window.iloc[i-j]["high"] >= curr["high"]`, set `is_high = False` and stop. If still
  `True`, repeat the same check forward for every `j` in `1..right_bars` against
  `window.iloc[i+j]["high"]`.
- Symmetric for `is_low`, using `"low"` and `<=` instead of `>=`.

So a swing high requires `curr["high"]` to be **strictly greater** than the high of
every one of the `left_bars` candles before it and every one of the `right_bars`
candles after it (a tie disqualifies it — `>=` fails the check). Swing low is the
mirror image on `"low"` with `<=`.

**Parameters:**

| Param | Default | Where | Status |
|---|---|---|---|
| `left_bars` | `3` | `detect_swings()` signature | Hardcoded literal default. **Not** flagged as a TODO anywhere in the code, unlike `lookback_bars` — but just as unvalidated for 5-minute crypto bars. |
| `right_bars` | `3` | `detect_swings()` signature | Same as above. |
| Both, passed through as | `swing_left_bars=3`, `swing_right_bars=3` | `compute_smc_features()` signature (lines 257-258) | Hardcoded defaults, no TODO comment. |

**Confirmation lag.** A swing at index `i` cannot be known until `i + right_bars`
candles later (you need `right_bars` future candles to confirm nothing broke it) —
`detect_bos_choch` explicitly accounts for this (see below); nothing else does.

**Textbook comparison.** This is a standard N-left/N-right fractal pivot, not a
divergence in *kind*. It *is* a divergence in specific value: `left_bars=right_bars=3`
means a candle needs to be the extreme of a **7-candle window** (3 + itself + 3) to
register. Many textbook/common fractal definitions use 2-and-2 (5-candle window) or
even 1-and-1. 3-and-3 is stricter/slower to confirm than the more common versions —
worth knowing when comparing "how many swings did this find" against another tool or
another timeframe.

### `classify_swing_trend` (lines 120-136)

Not one of the 6 detectors the question asked about by name, but it's the fourth
confluence vote (`swing_trend`) — documented here since it's small.

**Literal condition**, using only the **last two** swing highs and **last two** swing
lows in the (already lookback-limited) swing list:

```
hh = highs[-1]["price"] > highs[-2]["price"]
hl = lows[-1]["price"]  > lows[-2]["price"]
lh = highs[-1]["price"] < highs[-2]["price"]
ll = lows[-1]["price"]  < lows[-2]["price"]
if hh and hl: return "UPTREND"
if lh and ll: return "DOWNTREND"
return "RANGING"
```

Requires `len(lows) >= 2 and len(highs) >= 2`, else unconditionally `"RANGING"`. No
threshold beyond strict `>`/`<` — an exact tie (or the "neither higher-high-higher-low
nor lower-high-lower-low" case, e.g. higher high + lower low) falls through to
`"RANGING"`. No parameters. Despite the module's `classify_swing_trend` naming, this
does **not** fit or draw an actual trendline — it's a 2-point comparison, as its own
docstring already says.

---

## BOS / CHoCH (`detect_bos_choch`, lines 139-218)

This is the most stateful detector and the one most worth reading carefully, because
two of its behaviors are easy to get wrong from a paraphrase.

**State carried across the loop:** `pending_high`, `pending_low` (the current
unbroken candidate swing high/low dicts, or `None`), `bias` (`None` / `"bullish"` /
`"bearish"` — the *current* structural bias).

**Per bar `i` (0..len(window)-1), two separate mechanisms, in this order:**

**(a) Swing-confirmation phase** (lines 176-196) — a `while` loop that processes every
swing whose confirmation point has arrived: `swings_sorted[swing_ptr]["index"] +
right_bars <= i`. For each newly-confirmed swing `s`:

- If `s["type"] == "high"`: **if** `pending_high is not None and s["price"] >
  pending_high["price"]` — i.e. the new swing high is *itself* already above the
  still-unbroken pending level — fire an event: `event_type = "BOS" if bias ==
  "bullish" else "CHoCH"`, `direction="bullish"`, **`price` = the OLD `pending_high`
  price** (the level that broke), **`index`/`time` = the NEW swing's** index/time. Set
  `bias = "bullish"`. Then, **unconditionally** (whether or not the above fired):
  `pending_high = s` — the tracked level is always replaced by the newest confirmed
  high.
- Symmetric for `"low"` swings against `pending_low`, using `<` and `bias =
  "bearish"`.

**(b) Close-crossing phase** (lines 198-216) — runs every bar, independent of whether
phase (a) did anything this bar:

- `close = float(window.iloc[i]["close"])`.
- **if** `pending_high is not None and close > pending_high["price"]`: fire
  `event_type = "BOS" if bias == "bullish" else "CHoCH"`, `direction="bullish"`,
  `price` = the (old) `pending_high` price, `index=i`. Set `bias="bullish"` and — unlike
  phase (a) — **`pending_high = None`** (the level is *consumed*, not replaced; no new
  pending high exists until the next swing high is confirmed in a later bar's phase
  (a)).
- **elif** `pending_low is not None and close < pending_low["price"]`: mirror image,
  `pending_low = None`.
- This is `if`/`elif`, not two independent `if`s — only one of the two can fire per
  bar. In a coherent structure (`pending_low < pending_high`) this rarely matters in
  practice, but it's the literal behavior.

**BOS vs. CHoCH labeling — the actual rule, stated exactly:** a **high** break is
`"BOS"` **only if** `bias == "bullish"` already; **every other case — `bias ==
"bearish"` or `bias is None`** — labels it `"CHoCH"`. Symmetric for low breaks against
`bias == "bearish"`.

> **Divergence from textbook.** Textbook CHoCH specifically means *reversing an
> established trend*. This code's ternary can't distinguish "reversing an established
> bearish bias" from "this is the very first break the detector has ever seen, there
> was no established bias at all" — both produce `bias is None` (well, the first one
> obviously isn't `None`, but the *first-ever* break in a session, before any bias has
> formed, has `bias is None`, and `None != "bullish"` and `None != "bearish"` both
> evaluate the same way as an actual opposite-bias reversal) and both get labeled
> `"CHoCH"`. So the very first structural break `compute_smc_features` ever detects in
> a window is always reported as a "change of character" even though nothing changed
> from — there was no prior character to change from. This is a real labeling quirk,
> not just a naming nitpick.

**Look-ahead guard.** A swing at `index` is only usable once `index + right_bars <=
i` (matches `detect_swings`'s confirmation lag above) — this is the mechanism that
prevents future data from informing a "confirmed" swing before it could actually be
known.

**Documented bug fix (already in the code comments, restated here for completeness).**
Swings are confirmed by wick (`high`/`low`), not `close`. Phase (a)'s `s["price"] >
pending_high["price"]` check exists specifically so that a *wick-confirmed* break
(the new swing already past the old level) still fires an event even if `close` never
actually crossed that level — without it, `pending_high` would just get silently
swaped for the tighter level with no event, and `structure_bias`/`bias` would stay
stuck on a stale value. A swing that only *tightens* the level (a higher low while
bullish, a lower high while bearish) is structure continuation and updates the level
with **no** event — this is intentional, not a gap.

**Parameters:**

| Param | Default | Status |
|---|---|---|
| `right_bars` | `3`, reused from `swing_right_bars` | Not independently configurable — tied 1:1 to the swing detector's `right_bars`. Hardcoded, no TODO. |
| Break threshold | **none** | Any `close` beyond the level by any amount — a single tick — counts. No minimum-distance / minimum-% filter exists anywhere in this function. |

**What the aggregator actually reads.** `compute_smc_features` exposes
`"structure_bias": structure_events[-1]["direction"] if structure_events else None` —
just the **direction** of the chronologically last event, whatever its `type`.
`signals/smc_aggregator.py` (lines 120-125) reads only `structure_bias`
(`"bullish"`/`"bearish"`) as one of its four confluence votes.

> **Unused distinction.** The BOS-vs-CHoCH `type` label this detector computes is
> never read by the aggregator at all — only `direction`. The strategy currently
> cannot tell (and does not try to tell) a trend-continuation break from a
> trend-reversal break; both move the confluence score identically. If BOS-vs-CHoCH
> is ever meant to matter for sizing/conviction, that's unbuilt, not just untuned.

---

## Order block (`detect_order_blocks`, lines 26-51)

**Literal condition.** For each adjacent pair `prev = window.iloc[i-1]`, `curr =
window.iloc[i]`, `i` in `1..len(window)-1`:

- Bullish: `prev["close"] < prev["open"]` (prev is a down candle) **and**
  `curr["close"] > curr["open"]` (curr is an up candle) **and** `curr["close"] >
  prev["high"]` (curr's close breaks above prev's high). If all three: append
  `{"type": "bullish", "high": prev["high"], "low": prev["low"], ...}` — **the stored
  zone is `prev`'s high/low, i.e. the down candle itself**, not curr.
- Bearish: mirror image — `prev close > prev open`, `curr close < curr open`, `curr
  close < prev["low"]`; zone = `prev`'s high/low.

This is a plain 2-candle engulfing pattern, checked independently for every adjacent
pair in the window (no deduplication, no "is this still the most relevant one").

**Parameters:** none. No minimum body size, no minimum range, no volume filter, no
displacement-size requirement.

**Directly answering: is order block validity gated on a subsequent BOS here?**
**No.** `detect_order_blocks(window)` takes only `window` — it has no access to
`swings` or `structure_events` and is called (line 273) *before* `detect_swings`/
`detect_bos_choch` even run in `compute_smc_features`. There is no code path,
anywhere in this repository, that checks "did this order block precede/cause a
break of structure" before including it in `order_blocks`, before the aggregator
counts it (`smc_aggregator.py` lines 97-103, which reads only `order_blocks[-1]` —
the single most recent one, full stop), or before `backtest/runner.py`'s
`_structural_stop_loss` (lines 110-120) uses it for stop placement (same pattern:
`order_blocks[-1] if order_blocks else None`, no BOS check, no mitigation check).

> **Divergence from textbook.** Stricter ICT-style definitions typically require an
> order block to be the last opposite candle *before an impulsive move that breaks
> structure* — i.e., BOS-gated by construction — and often track whether price has
> since traded back through the zone ("mitigated") before treating it as still
> valid. None of that exists here. This implementation will flag **any** 2-candle
> engulf as an order block, valid or not, structurally significant or not, mitigated
> or not, for as long as it's the most recent one in the lookback window.

---

## FVG (`detect_fvg`, lines 54-76)

**Literal condition.** For `candle_1 = window.iloc[i-2]`, `candle_3 =
window.iloc[i]`, `i` in `2..len(window)-1` — **`candle_2` (`window.iloc[i-1]`) is
never read or assigned to a variable at all**:

- Bullish: `candle_3["low"] > candle_1["high"]` → `{"type": "bullish", "top":
  candle_3["low"], "bottom": candle_1["high"], ...}`.
- Bearish: `candle_3["high"] < candle_1["low"]` → `{"type": "bearish", "top":
  candle_1["low"], "bottom": candle_3["high"], ...}`.

This *is* the standard 3-candle FVG definition (a gap between candle 1 and candle 3
that skips candle 2's range entirely) — no divergence in kind.

**Parameters:** none. **No minimum gap size** — a gap of a single price increment
counts exactly the same as a large one. No filter on candle 2's range/momentum
(some FVG variants require candle 2 to be a large/impulsive candle; this
implementation doesn't look at candle 2 at all, for any purpose). No mitigation
check — same as order blocks, only `fvgs[-1]` (the most recent) is read by the
aggregator (lines 105-111), with no check for whether price has since traded back
through the gap.

---

## Liquidity sweep / fakeout: one detector, not two (`detect_fakeout`, lines 221-245)

The question's list treats "liquidity sweep" and "fakeout" as separate detectors.
**They are not** — there is exactly one function, `detect_fakeout`, and its output
types are literally named `"BULL_FAKEOUT"` / `"BEAR_FAKEOUT"`. Its docstring calls it
a liquidity sweep ("a wick past the last swing that closes back on the other side");
the function/type names call it a fakeout. Same code, two names, no independent
"liquidity sweep" concept exists anywhere else in this file.

**Literal condition.** Only ever looks at the **last two candles** in the window —
`curr = window.iloc[-1]`, `prev = window.iloc[-2]` — and the **most recently
confirmed swing** of each type (`recent_lows[-1]`, `recent_highs[-1]`, taken from
whatever `swings` list was passed in):

- `if recent_lows: last_low = recent_lows[-1]["price"]`; **if** `prev["low"] <
  last_low and curr["close"] > last_low`: return `{"type": "BULL_FAKEOUT",
  "swept_level": last_low, ...}`.
- `if recent_highs: last_high = recent_highs[-1]["price"]`; **if** `prev["high"] >
  last_high and curr["close"] < last_high`: return `{"type": "BEAR_FAKEOUT",
  "swept_level": last_high, ...}`.
- Otherwise `None`.

**Directly answering: is this "BOS/CHoCH later invalidated," or something else?**
**Something else, entirely independent of BOS/CHoCH.** `detect_fakeout`'s signature
is `(window, swings)` — it does not receive `structure_events` and never looks at
what `detect_bos_choch` concluded. It checks raw swing levels and two raw candles
directly. Two concrete consequences:

1. A swing level that was **only ever wick-broken** (never `close`-broken, so
   `detect_bos_choch` never fired a BOS/CHoCH event on it at all) can still be
   flagged as a fakeout here, because this function doesn't care whether a
   structural event fired — only whether the wick/close pattern matches.
2. The two detectors can and do run completely decoupled: nothing stops
   `structure_bias` from saying `"bullish"` in the same feature dict where
   `fakeout` says `"BEAR_FAKEOUT"` against a *different* swing level, or vice versa.

**"How many candles before invalidation" — the actual answer.** There is no
configurable window and no "N candles to reclaim" parameter. The pattern is
structurally fixed at exactly **2 candles**: the sweep must be in the single candle
immediately preceding the current one (`prev`), and the reclaim must be the
*current* candle's close (`curr`). A sweep that happened 2+ candles ago and only
just reclaimed now is invisible to this function — by the time `curr` is examined,
that sweep candle is no longer `prev` and the pattern doesn't match. This is a hard
structural constraint, not a tunable default.

**Parameters:** none configurable. The "2 candles" constraint above is baked into
which array indices are read, not a named parameter.

**How the aggregator uses it (`smc_aggregator.py`, lines 135-153).** Fakeout
contributes **zero points** to the confluence score directly — it is not one of the
four `bullish +=1` / `bearish +=1` factors. It only acts *after* the 4-factor
confluence has already produced a non-`NEUTRAL` `direction`, and only as a veto when
it **opposes** that direction:

```
if fakeout and direction != "NEUTRAL":
    if fk_type == "BULL_FAKEOUT" and direction == "BEARISH": veto -> NEUTRAL
    elif fk_type == "BEAR_FAKEOUT" and direction == "BULLISH": veto -> NEUTRAL
```

A fakeout that *agrees* with the tentative direction (e.g. `BULL_FAKEOUT` while
`direction == "BULLISH"`) does nothing at all — not a bonus, not a confirmation,
simply ignored. A fakeout while `direction == "NEUTRAL"` is also a no-op (the `if`
guard skips it).

---

## Summary: every parameter, in one table

| Detector | Parameter | Value | Status |
|---|---|---|---|
| Swing (`detect_swings`) | `left_bars` | `3` | Hardcoded default. **Not** TODO-flagged anywhere. |
| Swing | `right_bars` | `3` | Hardcoded default. **Not** TODO-flagged anywhere. |
| BOS/CHoCH (`detect_bos_choch`) | `right_bars` | `3` (= swing's `right_bars`) | Hardcoded, reused, not independent. |
| BOS/CHoCH | break threshold | none | Not a parameter — doesn't exist. Any `close` beyond the level, however small, breaks it. |
| Order block (`detect_order_blocks`) | any threshold | none | Not a parameter — doesn't exist. No size/volume/BOS-gating/mitigation check. |
| FVG (`detect_fvg`) | minimum gap size | none | Not a parameter — doesn't exist. Any nonzero gap counts. |
| Fakeout/sweep (`detect_fakeout`) | candles allowed before invalidation | fixed at 2 (structural, not a parameter) | Not configurable — baked into which indices (`[-1]`, `[-2]`) are read. |
| `compute_smc_features` | `lookback_bars` | `90` (`DEFAULT_LOOKBACK_BARS`) | **Explicit TODO** — tuned for daily stock bars in tradingagents-kr, flagged in-code as needing its own validation for 5m bars. |
| `SMCSignalAggregator` | `min_confluence` | none — constructor **raises `ValueError`** if not passed explicitly | **Explicit TODO by design** — no default exists on purpose; every caller must pass a value. |

Note the asymmetry: `lookback_bars` and `min_confluence` are the two parameters
that got explicit TODO comments and (for `min_confluence`) an enforced
required-argument guard. `swing_left_bars`/`swing_right_bars`/the BOS `right_bars`
are exactly as unvalidated for 5-minute BTC/USD bars, but nothing in the code flags
them the same way — they're silently trusted at their literal default of `3`.
