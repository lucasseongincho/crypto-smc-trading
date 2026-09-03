# SMC detector logic, as actually implemented

This documents exactly what `data/smc.py` checks and how `signals/smc_aggregator.py`
uses it — against the real code, not the generic textbook description of SMC, and not
the reference video this system was rebuilt to match on paper. Written because several
of these diverge from both in ways that matter for interpreting backtest results, and
"I'll remember" isn't a substitute for a written record.

**What this system is today** (2026-09): six detectors over an OHLCV window — order
block, FVG, swing (fractal pivots, feeding the two below), trendline, channel, and
fakeout/trap — computed by `data/smc.py::compute_smc_features` into one feature dict.
`signals/smc_aggregator.py::SMCSignalAggregator` turns that into a single directional
call via **5 independent confluence votes** (order block, FVG, trendline, channel,
fakeout/trap — swing itself doesn't vote, it only feeds the trendline/channel fits),
each worth `+1`/`-1`, no veto mechanism. A seventh detector, BOS/CHoCH
(`detect_bos_choch`), still runs on every call — its swing-confirmation machinery is
shared infrastructure the trendline/channel fits also rely on — but its own output
(`structure_events`/`structure_bias`) is not read by the aggregator or drawn on the
dashboard; it is fully retained code with no remaining consumer. See each section below
for the literal, current behavior; see [Design history](#design-history-why-this-looks-the-way-it-does)
near the end for the record of *why* it evolved to this shape, kept separate from the
"what it does now" sections above it so the two don't get tangled.

**Two things worth knowing before reading further, because they're easy to get wrong
from a paraphrase:**

1. "Liquidity sweep" and "fakeout" are not two detectors — there is one function,
   `detect_fakeout`, generalized (2026-09) to check multiple kinds of levels and paired
   with a new, separate trap pattern. See [Fakeout / trap](#fakeout--trap-detect_fakeout).
2. Order block validity has **no** BOS/CHoCH gating and no mitigation tracking anywhere
   in this codebase — not in the detector, not in the aggregator, not in the one other
   place `order_blocks` gets read (`backtest/runner.py`'s stop-loss placement). A
   minimum-size filter exists (`min_ob_body_ratio`); BOS-gating and mitigation tracking
   are simply not built. See [Order block](#order-block-detect_order_blocks).

---

## Swing detection (`detect_swings`)

`detect_swings`, lines 239-293. **Literal condition.** `highs`/`lows` are `window["high"]`/`window["low"]` as numpy
arrays. For each candidate index `i` (only indices with at least `left_bars` candles
before and `right_bars` candles after are even considered — the loop is `for i in
range(left_bars, n - right_bars)`):

- `is_high` starts `True`. For every `j` in `1..left_bars`: if `highs[i-j] >=
  curr_high`, set `is_high = False` and stop. If still `True`, repeat the same check
  forward for every `j` in `1..right_bars` against `highs[i+j]`.
- Symmetric for `is_low`, using `lows`/`curr_low` and `<=` instead of `>=`.

So a swing high requires `curr_high` to be **strictly greater** than the high of
every one of the `left_bars` candles before it and every one of the `right_bars`
candles after it (a tie disqualifies it — `>=` fails the check). Swing low is the
mirror image on `lows`/`curr_low` with `<=`.

**Parameters:**

| Param | Default | Where | Status |
|---|---|---|---|
| `left_bars` | `3` | `detect_swings()` signature | Hardcoded literal default. **Not** flagged as a TODO anywhere in the code, unlike `lookback_bars` — but just as unvalidated for 5-minute crypto bars. |
| `right_bars` | `3` | `detect_swings()` signature | Same as above. |
| Both, passed through as | `swing_left_bars=3`, `swing_right_bars=3` | `compute_smc_features()` signature (line 1035) | Hardcoded defaults, no TODO comment. |

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

### `classify_swing_trend` — removed, replaced by `detect_trendline` (2026-09 rebuild, Phase 2)

**This function no longer exists.** It was a 2-swing higher-high/higher-low
comparison that never actually fit a diagonal line despite its name (the exact
logic that used to live here is preserved below, for anyone diffing against an
older version of this doc, but the code itself is gone — `git log` has the removal
commit). It has been **replaced, not extended**, by a real trendline detector — see
[Trendline](#trendline-detect_trendline) below — per the rebuild
decision: keeping both under similar names would have been confusing about which
one a reader should trust. `compute_smc_features`'s output no longer has a
`swing_trend` key at all; `signals/smc_aggregator.py`'s vote on it was retired in
the same rebuild's aggregator-rewire phase (Phase 5 — see that section).

<details>
<summary>Retired logic, for reference only</summary>

Using only the **last two** swing highs and **last two** swing lows in the
(already lookback-limited) swing list:

```
hh = highs[-1]["price"] > highs[-2]["price"]
hl = lows[-1]["price"]  > lows[-2]["price"]
lh = highs[-1]["price"] < highs[-2]["price"]
ll = lows[-1]["price"]  < lows[-2]["price"]
if hh and hl: return "UPTREND"
if lh and ll: return "DOWNTREND"
return "RANGING"
```

Required `len(lows) >= 2 and len(highs) >= 2`, else unconditionally `"RANGING"`. No
threshold beyond strict `>`/`<` — an exact tie (or the "neither higher-high-higher-low
nor lower-high-lower-low" case, e.g. higher high + lower low) fell through to
`"RANGING"`. No parameters.

</details>

---

## Trendline (`detect_trendline`)

`detect_trendline`, lines 404-503; `_fit_line`, lines 294-313; `_TrendlineTracker`,
lines 334-403. Real diagonal trendlines, replacing `classify_swing_trend` above.
Fits **two independent lines**: a **support** line
through confirmed swing **lows** (the uptrend line) and a **resistance** line
through confirmed swing **highs** (the downtrend line), via least-squares over the
most recent `trendline_points` confirmed swings of each type.

**Exact-math choice (the video doesn't specify this).** The rebuild decision raised
two options: least-squares over the last N points, or a raw two-point line through
the most recent unbroken pair, extended forward. `_fit_line` unifies both: ordinary
least-squares reduces *exactly* to the two-point line when given exactly 2 points
(there's only one line through 2 points, and that's what least-squares finds), so
`trendline_points=2` reproduces the "two-point line" option exactly, while
`trendline_points=3` (the chosen default, `DEFAULT_TRENDLINE_POINTS`) gets a
genuine regression fit. Like `lookback_bars`, `3` is a functional placeholder, not
a value validated against 5-minute BTC/USD bars — same TODO treatment.

**Literal condition.** Structurally close to `detect_bos_choch` (same
confirmation-lag guard via `right_bars`, same overall bar-by-bar loop shape), with
two real differences: it tracks two lines instead of one bias, and it fits a
continuous function instead of tracking a single pending level.

- Per bar `i`, first process every swing newly confirmed as of this bar
  (`swing["index"] + right_bars <= i`, identical lag to `detect_swings`/
  `detect_bos_choch`): append it to a running, **never-reset** list
  (`low_points` for swing lows, `high_points` for swing highs). Once a list has
  at least `trendline_points` entries, refit that type's line from the most
  recent `trendline_points` of them (`_fit_line`, plain least-squares — sums of
  `x`, `y`, `xy`, `x²` over the point set, no numpy dependency added).
- Then, independently for each line that currently exists, project it to this
  bar's x-position (`slope * i + intercept`) and compare against **`close`, not
  the bar's high/low** — the same close-vs-wick discipline `detect_bos_choch`'s
  close-crossing phase uses:
  - Support: `close < projected - min_trendline_break_distance` → append
    `{"type": "SUPPORT_BREAK", "direction": "bearish", "price": projected,
    "index": i, "time": ts}`, then set the support line to `None` (invalidated;
    stays `None` until enough new lows accumulate to refit — `low_points` itself
    is never cleared, so the next confirmed low naturally triggers a refit).
  - Resistance: mirror image, `close > projected + min_trendline_break_distance`
    → `RESISTANCE_BREAK`, `direction: "bullish"`.
  - These are two **independent `if`s, not `if`/`elif`** (unlike
    `detect_bos_choch`'s single pending-level check) — a support break and a
    resistance break concern two unrelated lines and could in principle both
    fire on the same bar.

**Deliberately not carried over from `detect_bos_choch`.** No equivalent of the
wick-based "supersession" phase (the mechanism that lets a newly-confirmed swing
already being past a pending *level* fire an event even without a close-crossing).
A continuously-projected line doesn't have a clean analog of "the new swing is
already past the old level" — the projected value changes every bar, not just at
swing-confirmation points — so trendline breaks are close-crossing only. A
documented scoping choice, same treatment as `min_break_distance`'s scoping note
above.

**Parameters:**

| Param | Default | Status |
|---|---|---|
| `trendline_points` | `3` (`DEFAULT_TRENDLINE_POINTS`) | **Explicit TODO**, same pattern as `lookback_bars`. `2` exactly reproduces the "two-point line through the most recent pair" alternative (see above); `3+` is a genuine least-squares fit. |
| `min_trendline_break_distance` | `0.0` (`DEFAULT_MIN_TRENDLINE_BREAK_DISTANCE`) | **Explicit TODO**, same pattern as `min_break_distance`. `0.0` means any close beyond the projected line value, however small, counts as a break. |
| `right_bars` | `3` (reused from `swing_right_bars`) | Not independently configurable — tied 1:1 to the swing detector's `right_bars`, same as `detect_bos_choch`. |

**Output shape.** `{"support": {...} | None, "resistance": {...} | None,
"breaks": [...]}`. `support`/`resistance`, when not `None`, describe the line **as
it stands at the end of the window**: `{"slope", "intercept",
"value_at_last_index"}` — `None` means either fewer than `trendline_points`
confirmed swings of that type ever existed, or the line was invalidated by a break
with nothing yet refit. `breaks` is every event across the whole window,
chronological, same field shape as `detect_bos_choch`'s events.

Both `detect_trendline` and `detect_channel` (next section) build their own
`_TrendlineTracker` internally rather than sharing one instance — a known,
deliberate duplication of the confirm/refit bar loop inside
`compute_smc_features`, left as-is since this path isn't on the tuning grid's hot
path (the grid isn't being re-run as part of this rebuild).

---

## Channel (`detect_channel`)

`detect_channel`, lines 565-697; `_fit_parallel_line`, lines 504-518;
`_channel_bar_events`, lines 519-564. Parallel channel boundaries around the
trendlines above. The rebuild decision's own framing: "a parallel line to the
trendline, connecting the corresponding highs/lows, same slope." Built two ways,
tracked simultaneously (there's no "current trend direction" concept left to pick
just one — `classify_swing_trend` was removed, not replaced with an equivalent):

- **Ascending channel**: the **support** line (from swing lows) is the **lower**
  boundary, reused exactly as `detect_trendline` computes it. The **upper**
  boundary is a *new* line with the identical slope, best-fit via
  `_fit_parallel_line` through the most recent `trendline_points` confirmed swing
  **highs**.
- **Descending channel**: mirror image — the **resistance** line (from swing
  highs) is the **upper** boundary; the **lower** boundary is fit through the
  most recent `trendline_points` confirmed swing **lows**, same slope as
  resistance.

**Fixed-slope fit (`_fit_parallel_line`).** Given a slope and a set of `(index,
price)` points, the best intercept by least-squares is `mean(price - slope *
index)` — the closed-form minimizer of `sum((y - (slope*x + b))^2)` over `b`
alone. This connects *all* the corresponding-type points to the parallel
boundary (consistent with `detect_trendline`'s own preference for fitting over
`trendline_points` points rather than anchoring on one touch), not just the
single most recent high/low.

**Shared state, not reimplemented state.** `detect_channel` uses the exact same
`_TrendlineTracker` class `detect_trendline` uses — same `advance()`
confirm/refit logic, same `check_trendline_breaks()` invalidation logic, same
`min_break_distance`-style parameter for it (passed as `min_break_distance` here
too, not renamed). This isn't a style preference: the ascending channel's lower
boundary and `detect_trendline`'s support line are, numerically, the identical
line, computed the identical way — reusing the tracker is what *guarantees* that
identity rather than merely documenting an intent that two independent
implementations could silently drift away from.

**Per bar `i`** (after `tracker.advance(i)`): if the anchor line exists **and**
the *other* side has accumulated at least `trendline_points` confirmed swings,
project both boundaries to this bar's x-position and check touch/break
(`_channel_bar_events`) — independently for the ascending and descending
channels; both, one, or neither can be active on a given bar depending on which
lines currently exist. `tracker.check_trendline_breaks()` still runs every bar
afterward so the anchor lines invalidate at the identical point
`detect_trendline` would report — a channel's boundary would otherwise silently
go stale relative to its own anchor line.

**Touch vs. break — the actual rule.** A `CHANNEL_BREAK` (close beyond a
boundary — the "higher-conviction trend-reversal signal" the rebuild decision
asked for) takes priority over a `CHANNEL_TOUCH` (wick reaches the boundary
without close following through — "potential reversal") on the same bar/
boundary; they are not both reported for the same event. **Direction
convention** (a judgment call — the video doesn't label these): breaking the
**upper** boundary is `"bullish"` and breaking the **lower** is `"bearish"` — the
same convention `detect_trendline`'s `RESISTANCE_BREAK`/`SUPPORT_BREAK` already
use, since an ascending channel's lower boundary literally *is* the support line
and a descending channel's upper boundary literally *is* the resistance line. A
**touch** carries the *opposite* direction from a break at the same boundary —
touching the upper boundary from inside the channel implies a potential
rejection *down* (`"bearish"`), touching the lower boundary implies a potential
bounce *up* (`"bullish"`).

**Not duplicated:** a channel-anchor break (e.g. the ascending channel's lower
boundary breaking) is reported once, here, as a `CHANNEL_BREAK`. It is *not*
also separately reported as `detect_trendline`'s `SUPPORT_BREAK` under a
different name inside this function's own `events` list — that event exists in
`detect_trendline`'s own `breaks` list instead; a caller wanting both needs both
functions' outputs (`compute_smc_features` calls both, see below).

**Parameters:**

| Param | Default | Status |
|---|---|---|
| `trendline_points` | `3` (`DEFAULT_TRENDLINE_POINTS`, shared with `detect_trendline`) | Same TODO as `detect_trendline`'s. Also gates the *other*-side minimum here — an ascending channel needs `trendline_points` confirmed highs, not just an existing support line. |
| `min_break_distance` | `0.0` (`DEFAULT_MIN_TRENDLINE_BREAK_DISTANCE`, shared with `detect_trendline`) | Gates the anchor line's own invalidation via `tracker.check_trendline_breaks` — literally the same parameter/threshold `detect_trendline` uses for the identical line. |
| `min_channel_break_distance` | `0.0` (`DEFAULT_MIN_CHANNEL_BREAK_DISTANCE`) | **Explicit TODO**, deliberately a *separate* parameter from `min_break_distance` even though both gate "close beyond a line" — the constructed parallel boundary and the underlying trendline are different lines with potentially different noise characteristics, so tying their thresholds together would be an unjustified assumption. |
| `right_bars` | `3` (reused from `swing_right_bars`) | Same as `detect_trendline`. |

**Output shape.** `{"ascending": {...} | None, "descending": {...} | None,
"events": [...]}`. Each channel dict, when not `None`: `{"slope", "lower":
{"intercept", "value_at_last_index"}, "upper": {"intercept",
"value_at_last_index"}}`, describing both boundaries as they stand at the end of
the window. `events` is every `CHANNEL_TOUCH`/`CHANNEL_BREAK` across the whole
window, chronological, each `{"type", "channel": "ascending"|"descending",
"boundary": "upper"|"lower", "direction", "price", "index", "time"}`.

---

## BOS / CHoCH (`detect_bos_choch`) — retained, unread

`detect_bos_choch`, lines 698-803. Still runs on every `compute_smc_features` call; its swing-confirmation
machinery is shared infrastructure `detect_trendline`/`detect_channel` also
rely on (see the [Trendline](#trendline-detect_trendline) section's `_TrendlineTracker`). But
**nothing reads this detector's own output anymore** — not the aggregator
(retired in the 5-vote rewire; see [Aggregator](#aggregator-signalssmc_aggregatorpy)),
not the dashboard (BOS/CHoCH's chip, overlay, and palette colors were removed
outright). `structure_events`/`structure_bias` are still computed and still sit
in `compute_smc_features`'s return dict, fully live code with zero remaining
consumers — described below exactly as it behaves, in case that changes again.

This is the most stateful detector and the one most worth reading carefully, because
two of its behaviors are easy to get wrong from a paraphrase.

**State carried across the loop:** `pending_high`, `pending_low` (the current
unbroken candidate swing high/low dicts, or `None`), `bias` (`None` / `"bullish"` /
`"bearish"` — the *current* structural bias).

**Per bar `i` (0..len(window)-1), two separate mechanisms, in this order:**

**(a) Swing-confirmation phase** (lines 747-767) — a `while` loop that processes every
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
- **Not gated by `min_break_distance`** — see below.

**(b) Close-crossing phase** (lines 769-787) — runs every bar, independent of whether
phase (a) did anything this bar:

- `close = closes[i]` (`closes` is `window["close"]` as a numpy array, precomputed
  once before the loop, same as `times = _all_time_strs(window)` for the
  timestamps).
- **if** `pending_high is not None and close > pending_high["price"] +
  min_break_distance`: fire `event_type = "BOS" if bias == "bullish" else "CHoCH"`,
  `direction="bullish"`, `price` = the (old) `pending_high` price, `index=i`. Set
  `bias="bullish"` and — unlike phase (a) — **`pending_high = None`** (the level is
  *consumed*, not replaced; no new pending high exists until the next swing high is
  confirmed in a later bar's phase (a)).
- **elif** `pending_low is not None and close < pending_low["price"] -
  min_break_distance`: mirror image, `pending_low = None`.
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
| `min_break_distance` | `0.0` (`DEFAULT_MIN_BREAK_DISTANCE`, line 695) | **Explicit TODO.** Raw price distance (not an ATR ratio like the OB/FVG filters — that's what was asked for) the close must clear beyond the level. `0.0` reproduces the prior "any close beyond the level, however small, breaks it" behavior exactly. **Scoping choice, documented in the function's own docstring:** applies only to phase (b), the close-crossing check — phase (a)'s wick-based swing-supersession check is unaffected, so a nonzero `min_break_distance` can still be circumvented via that path. |

`compute_smc_features` exposes `"structure_bias":
structure_events[-1]["direction"] if structure_events else None` — just the
**direction** of the chronologically last event, whatever its `type`. As
described at the top of this section, nothing reads it.

> **Unused distinction.** The
> BOS-vs-CHoCH `type` label this detector computes was already never read by
> the old aggregator - only `direction` was. Now neither field is read by the
> aggregator at all. The strategy has never been able to tell (and still isn't
> built to tell) a trend-continuation break from a trend-reversal break.

---

## Order block (`detect_order_blocks`)

`detect_order_blocks`, lines 96-188; `_body_engulf_type`, lines 70-95.

**2026-09 redefinition — body-engulf, not close-break-high.** Before this rebuild,
qualifying as an order block additionally required curr's *close* to break past
prev's high (bullish) or low (bearish) — a strictly stronger condition than "the
bodies engulf." This was changed to match the reference video's plain body-engulf
definition exactly, per the rebuild decision (see
`../crypto-smc-bot-notes/decisions/2026-09-03-detector-rebuild-decision.md` — outside
this repo, personal reasoning only, not itself a source of truth for the code).
**This is a deliberate behavior change, not a bug fix or a refactor** — unlike the
earlier numpy-vectorization pass, there is no bit-identical guarantee here, and none
was attempted; correctness is now measured against the video's spec, not against the
old code's output. `tests/test_smc.py::TestOrderBlockBodyEngulfRedefinition` covers a
case that qualifies now but didn't before (body engulfs, close never breaks the
prior wick).

**Literal condition.** `opens`/`highs`/`lows`/`closes` are `window`'s OHLC columns as
numpy arrays. For each adjacent pair `i` in `1..n-1` (`prev_open, prev_high,
prev_low, prev_close = opens[i-1], highs[i-1], lows[i-1], closes[i-1]`;
`curr_open, curr_close = opens[i], closes[i]`):

- First, a size gate: `prev_body = abs(prev_close - prev_open)`; if `prev_body <
  min_ob_body_ratio * atr_arr[i-1]`, `continue` (skip this pair entirely, regardless
  of what the engulf check below would have found). `atr_arr` comes from the `atr`
  parameter if the caller passed one, else `_atr(window)` computed here, a standard
  14-period true-range ATR — the reference point is `atr_arr[i-1]`, i.e. ATR *as of
  the candle being sized* (`prev`), not incorporating `curr`.
- `_body_engulf_type(prev_open, prev_close, curr_open, curr_close)` — a shared
  helper, used for both the base single-pair test below and the double-order-block
  upgrade check further down, so the two patterns can't silently drift apart into
  two different engulf definitions. Returns `"bullish"` if curr's body
  `[min(open,close), max(open,close)]` fully wraps prev's body **and** the two
  candles are opposite colors with prev red/curr green (a support candidate at
  prev); `"bearish"` for the mirror case (prev green/curr red — a resistance
  candidate at prev); `None` if the bodies don't fully wrap, or if they're the same
  color (a same-color "engulf" isn't a reversal pattern and never counts, whatever
  the body sizes).
- If `_body_engulf_type` returns non-`None`: append `{"type": engulf_type, "high":
  float(prev_high), "low": float(prev_low), "index": i, "time": ts, "pattern":
  "single"}` — **the stored zone is still `prev`'s full wick high/low**, not its
  body, and not curr — this didn't change in the redefinition, only the engulf test
  itself did.

This is a plain 2-candle body-engulfing pattern (now with a size gate), checked
independently for every adjacent pair in the window (no deduplication, no "is this
still the most relevant one").

**Double order block (new).** Not a separate detector or a separate comparison —
the same `_body_engulf_type` helper, applied one candle further back, upgrading an
already-found single result rather than producing an independent one. Worked
example, three consecutive candles A (bearish), B (bullish), C (bearish):

1. The ordinary loop, at `i` pointing at B (`prev=A, curr=B`), finds
   `_body_engulf_type(A, B) == "bullish"` (B's body engulfs A's) — appends the
   normal single order block at **A** (support/bullish), exactly as it always has.
2. The ordinary loop, at `i` pointing at C (`prev=B, curr=C`), finds
   `_body_engulf_type(B, C) == "bearish"` (C's body engulfs B's) — appends the
   normal single order block at **B** (resistance/bearish). This step is *not*
   double-order-block-specific — it's the exact same base test that produced A's
   entry in step 1, just one pair later.
3. **The additional check**: while still at `i` pointing at C, also evaluate
   `_body_engulf_type` one pair further back — `(A, B)` again — via
   `opens[i-2], closes[i-2], prev_open, prev_close`. Since that's `"bullish"`
   (computed in step 1) and this pair's own result is `"bearish"` (opposite), the
   pattern upgrades: append a **second** entry, same zone as B's step-2 entry
   (B's own high/low), same type (`"bearish"`), but `"pattern": "double"` instead
   of `"single"`. B ends up with two dict entries in `order_blocks` — its ordinary
   single result and its double-pattern upgrade — rather than one entry mutated in
   place; A's entry from step 1 is untouched either way.

So "the middle bullish candle becomes a resistance order block" (B, in the example
above) is produced by the *same* pair-engulf test that runs for every adjacent pair
regardless of pattern — nothing above the single-OB logic is a distinct
"double-order-block detector." The only genuinely new code is the one extra
`_body_engulf_type` call at step 3, looking one candle further back than the base
loop already does. `min_ob_body_ratio` applies identically to both — the double
entry only exists because its underlying single entry (step 2, on B) already
cleared the size gate; there is no separate/looser gate for the double case. See
`tests/test_smc.py::TestOrderBlockBodyEngulfRedefinition::test_double_order_block_upgrades_the_middle_candle`.

**Parameters:**

| Param | Default | Status |
|---|---|---|
| `min_ob_body_ratio` | `0.0` (`DEFAULT_MIN_OB_BODY_RATIO`, line 67) | **Explicit TODO.** Ratio of `prev`'s candle body to `_atr(window)` at that point — a ratio rather than a raw price threshold, since BTC's price scale drifts too much for a fixed dollar minimum to stay meaningful (same reasoning that flagged `backtest/risk.py`'s absolute-dollar `min_sl_distance` for re-validation; solved with a ratio here instead of a dollar figure). `0.0` reproduces the prior "any body size qualifies" behavior exactly (against the *old* close-break-high definition — see the redefinition note above for why "prior behavior" no longer means bit-identical here). |

Still **no** minimum range/wick filter, no volume filter, and — as before —
**no BOS-gating or mitigation tracking at all** (see below).

**Directly answering: is order block validity gated on a subsequent BOS here?**
**No**, still. `detect_order_blocks(window, min_ob_body_ratio, atr)` has no access to
`swings` or `structure_events` and is called *before* `detect_swings`/
`detect_bos_choch` even run in `compute_smc_features`. There is no code path,
anywhere in this repository, that checks "did this order block precede/cause a
break of structure" before including it in `order_blocks`, before the aggregator
counts it, or before `backtest/runner.py`'s `_structural_stop_loss` uses it for stop
placement (same pattern: `order_blocks[-1] if order_blocks else None`, no BOS check,
no mitigation check). The redefinition and the size filter both change *which*
engulfs qualify, not *how* validity is decided.

> **Divergence from textbook.** Stricter ICT-style definitions typically require an
> order block to be the last opposite candle *before an impulsive move that breaks
> structure* — i.e., BOS-gated by construction — and often track whether price has
> since traded back through the zone ("mitigated") before treating it as still
> valid. Neither exists here. This implementation flags **any** body-engulf
> clearing the (currently off) size gate as an order block, valid or not,
> structurally significant or not, mitigated or not, for as long as it's in the
> lookback window — the aggregator and stop-placement code then only ever look at
> the single most recent one (`order_blocks[-1]`), single or double alike.

---

## FVG (`detect_fvg`)

`detect_fvg`, lines 189-238.

**Literal condition.** `highs`/`lows` are `window`'s high/low columns as numpy
arrays. For `c1_high, c1_low = highs[i-2], lows[i-2]` and `c3_high, c3_low =
highs[i], lows[i]`, `i` in `2..n-1` — **candle 2 (index `i-1`) is never read at
all**:

- `min_gap = min_fvg_gap_ratio * atr_arr[i-2]` — `atr_arr` comes from the same
  shared `atr` parameter (or self-computed `_atr(window)`) `detect_order_blocks`
  uses; the reference point is candle 1's position (`i-2`), not candle 3's,
  deliberately excluding the gap-forming candles themselves from the volatility
  baseline used to judge whether the gap they form is significant.
- Bullish: `c3_low > c1_high` **and** `gap = c3_low - c1_high >= min_gap` →
  `{"type": "bullish", "top": float(c3_low), "bottom": float(c1_high), ...}`.
- Bearish: `c3_high < c1_low` **and** `gap = c1_low - c3_high >= min_gap` →
  `{"type": "bearish", "top": float(c1_low), "bottom": float(c3_high), ...}`.

This *is* the standard 3-candle FVG definition (a gap between candle 1 and candle 3
that skips candle 2's range entirely) — no divergence in kind.

**Parameters:**

| Param | Default | Status |
|---|---|---|
| `min_fvg_gap_ratio` | `0.0` (`DEFAULT_MIN_FVG_GAP_RATIO`, line 186) | **Explicit TODO.** Same ATR-ratio unit as `min_ob_body_ratio`, for consistency. `0.0` reproduces the prior "any nonzero gap qualifies" behavior exactly. |

Still no filter on candle 2's range/momentum (some FVG variants require candle 2 to
be a large/impulsive candle; this implementation doesn't look at candle 2 at all, for
any purpose). No mitigation check — same as order blocks, only `fvgs[-1]` (the most
recent) is read by the aggregator (`signals/smc_aggregator.py` line 126), with no
check for whether price has since traded back through the gap.

---

## Fakeout / trap (`detect_fakeout`)

`detect_fakeout`, lines 921-1034; `_detect_trap`, lines 804-920. "Liquidity
sweep" and "fakeout" are not two detectors — there is exactly one
function, `detect_fakeout`, and its classic sweep-and-reclaim output types are
literally named `"BULL_FAKEOUT"` / `"BEAR_FAKEOUT"`. It now does two things: the
classic sweep-and-reclaim check (generalized across three kinds of levels) and,
separately, trap detection (a new multi-bar pattern).

> ### ⚠️ Role change: veto → vote — the single biggest behavioral change in this rebuild
>
> Before this rebuild, `signals/smc_aggregator.py` used this detector's output
> **only as a veto**: a fakeout opposing an already-tentative direction forced
> that direction to `NEUTRAL`; a fakeout *agreeing* with the tentative direction
> did nothing at all. Phase 5 (the aggregator rewire — see that section) makes
> fakeout/trap a **genuine fifth confluence vote**, on equal footing with order
> block / FVG / trendline / channel: it can now independently push the
> confluence score toward `BULLISH` or `BEARISH` on its own, not just cancel an
> existing call. Every backtest result from before Phase 5 used the old
> veto-only model; every one after uses the new 5-vote model — they are not
> comparable, and the tuning grid needs a full re-run once the whole rebuild
> (through Phase 6) lands (deliberately not re-run yet — see the top of
> `docs/tuning-log.md`).

**Return shape changed.** `detect_fakeout` now returns `{"fakeout": {...} |
None, "trap": {...} | None}` — a breaking change from the prior single-dict-or-
`None` return. `compute_smc_features`'s `"fakeout"` key holds this new nested
dict (not renamed, to minimize churn elsewhere).

### Classic fakeout — generalized beyond the single raw swing level

Still only examines the **last two candles** (`curr = window.iloc[-1]`, `prev =
window.iloc[-2]`) — that structural "2 candles, no configurable window"
constraint from before is unchanged, and still baked into which array indices
are read, not a parameter. What changed is *which levels count*:

- **Before**: only the single most recently confirmed swing low/high.
- **Now**: swing (unchanged), **plus** the current Phase 2 trendline
  (`detect_trendline`'s `support`/`resistance`, evaluated at the last bar via
  `value_at_last_index`), **plus** the current Phase 3 channel's *derived*
  boundaries specifically (the ascending channel's upper boundary, the
  descending channel's lower boundary — the two that aren't already numerically
  identical to the trendline values).

Checked in a fixed, documented priority order — **swing, then trendline, then
channel** — and the first level that matches (`prev` wicks past it, `curr`
closes back on the correct side) wins; this is a deliberate tie-break (multiple
levels often sit close together and could all match the same 2-candle sweep),
not an attempt to report every matching level. The output's `"source"` field
(`"swing"` / `"trendline"` / `"channel"`, new) names which one fired.

`trendline`/`channel` are accepted as optional precomputed parameters (`None` →
computed internally) — the same shared-computation pattern `atr` uses for
`detect_order_blocks`/`detect_fvg` — since without it, `compute_smc_features`
would build 4 `_TrendlineTracker` instances per call (2 direct + 2 more inside
this function) instead of 2.

**Directly answering: is this "BOS/CHoCH later invalidated," or something
else?** Still something else, independent of BOS/CHoCH — `detect_fakeout`
never receives `structure_events` and never looks at what `detect_bos_choch`
concluded. Both consequences from the original audit still hold: a swing level
that was only ever wick-broken (never `close`-broken) can still be flagged here,
and the two detectors can and do run completely decoupled.

### Trap (new) — a distinct multi-bar pattern, not a veto or a vote by itself

See `_detect_trap`'s own docstring in `data/smc.py` for the full state-machine
walkthrough (break → first extreme E1 → bounce → second nearby extreme E2 →
reclaim). Summary:

- **Scoped to Phase 2 trendline breaks specifically** — not also the raw swing
  level or the Phase 3 channel-derived boundaries the classic fakeout check
  above additionally considers. A documented scope-narrowing (unifying three
  break sources into one unambiguous "which level broke, for stop-placement
  purposes" answer needs its own design pass), not an oversight.
- E1 (the extreme right after the break) becomes `stop_loss_reference` — not
  currently wired into `backtest/runner.py`'s `_structural_stop_loss` (which
  still only reads `order_blocks`); that wiring is out of this rebuild's scope
  and wasn't requested.
- E2 must be "nearby" E1 — within `max_trap_retest_distance` — *unless* E2 is
  actually shallower than E1 (a higher low after a support break, a lower high
  after a resistance break), which is always accepted regardless of distance.
  `max_trap_retest_distance` defaults to `float("inf")`, not `0.0` — the only
  parameter in this file where the "off"/no-op default is a maximum rather
  than a minimum (see `DEFAULT_MAX_TRAP_RETEST_DISTANCE`'s comment for why
  `0.0` would be backwards here).
- Only reported when the reclaim bar is the **last** bar of the window —
  mirroring the classic fakeout's own "only checks the most recent candle"
  scoping, so this always answers "did something confirm right now," not "did
  something ever confirm somewhere in this lookback window."
- Named `"SUPPORT_TRAP"`/`"RESISTANCE_TRAP"` (after which level broke), **not**
  the trader-jargon "bull trap"/"bear trap" (named after which side gets
  trapped — the *opposite* of the break direction). Mixing those two naming
  conventions invites exactly the kind of mislabeling bug this project already
  hit once (see the BOS/CHoCH labeling divergence note above) — `"direction"`
  (`"bullish"`/`"bearish"`) is the field to read for which way it resolves.

**Parameters:**

| Param | Default | Status |
|---|---|---|
| `max_trap_retest_distance` | `float("inf")` (`DEFAULT_MAX_TRAP_RETEST_DISTANCE`) | **Explicit TODO**, not validated against 5-minute BTC/USD bars. Inverted "off" value vs. every other filter in this file — see above. |
| `trendline_points`, `right_bars` | Same as `detect_trendline`'s | Reused, not independently configurable here. |

**How the aggregator uses this now:** see Phase 5 below — the old veto-only
snippet that used to live in this section has been retired along with the code
it described; nothing in this codebase still runs the old veto logic once
Phase 5 lands (before Phase 5 lands, in the intermediate commits, the
aggregator's old veto code reads a key shape that no longer exists and simply
never fires — a graceful no-op, not a crash, same treatment `swing_trend`'s
removal got in Phase 2).

---

## Aggregator (`signals/smc_aggregator.py`)

**Old model: 4 votes + 1 veto.** Order block, FVG, `swing_trend`, and
`structure_bias` (BOS/CHoCH) each independently added `+1`/`-1` to a confluence
score (`-4..+4`); a separate fakeout check ran *after* that score already
produced a non-`NEUTRAL` direction and could only cancel it back to `NEUTRAL`
when it disagreed — never independently move the score, and a fakeout that
*agreed* with the tentative direction did nothing at all.

**New model: 5 independent votes, no veto.** `swing_trend` was removed outright
in Phase 2 (not carried forward — `classify_swing_trend` no longer exists) and
`structure_bias` is no longer read at all (Phase 5, this section — the detector
still runs, see the BOS/CHoCH section above). In their place: **order block,
FVG, trendline, channel, fakeout/trap** — five votes, each `+1`/`-1`,
confluence range **`-5..+5`**, no veto mechanism anywhere in this module
anymore. `SMCSignal` dropped its `veto`/`veto_reason` fields entirely — there
is nothing left for them to describe.

**Literal condition, per vote:**

| Vote | Reads | Bullish when | Bearish when |
|---|---|---|---|
| `order_block` | `order_blocks[-1]["type"]` | `"bullish"` | `"bearish"` |
| `fvg` | `fvgs[-1]["type"]` | `"bullish"` | `"bearish"` |
| `trendline` | `trendline["breaks"][-1]["direction"]` | `"bullish"` (a `RESISTANCE_BREAK`) | `"bearish"` (a `SUPPORT_BREAK`) |
| `channel` | the most recent event in `channel["events"]` whose `"type"` is `"CHANNEL_BREAK"` (`CHANNEL_TOUCH` events are skipped — see below) | `"bullish"` | `"bearish"` |
| `fakeout_trap` | `fakeout["trap"]` if not `None`, else `fakeout["fakeout"]` — one combined vote, not two | trap/fakeout `"direction"`/`"type"` bullish | bearish |

Each vote reads the **most recent** item in its list, chronologically — same
"last one wins" pattern the old `order_block`/`fvg`/`structure_bias` votes
already used, now applied consistently to all 5.

**Why `CHANNEL_TOUCH` doesn't vote.** A touch and a break at the *same*
boundary carry *opposite* directional implications under `detect_channel`'s own
convention (touching a ceiling implies a potential move down; breaking it
implies bullish continuation) — mixing both event kinds into one vote would
need its own weighting design this rebuild doesn't specify. Touches remain
available in the feature dict for the dashboard (Phase 6) but don't move the
confluence score.

**Known correlation, not a bug.** The `trendline` and `channel` votes can and
often will agree on the same bar, because an ascending channel's lower
boundary *is* the support line and a descending channel's upper boundary *is*
the resistance line (see the Channel section above) — a `SUPPORT_BREAK` is
frequently also a `CHANNEL_BREAK` on the same boundary, on the same bar. This
means a single underlying structural break can supply 2 of the 5 votes at
once. Documented as an accepted consequence of Phase 3's shared-tracker design,
not something this rewire attempts to de-duplicate.

**`min_confluence` stays enforced-required, no default** — `SMCSignalAggregator.__init__`
still raises `ValueError` if not passed explicitly, unchanged from before this
rebuild. The reference video's own informal practice — "confirm with at least 2
of the 5 concepts" — is noted in the constructor's docstring **as a loose
reference point only**, the same treatment `lookback_bars`' TODO gets elsewhere
in this codebase, not a validated or recommended value. The tuning grid is
being re-run from scratch once the whole rebuild (through Phase 6) lands, not
before — see `docs/tuning-log.md`.

---

## Summary: every parameter, in one table

| Detector | Parameter | Value | Status |
|---|---|---|---|
| Swing (`detect_swings`) | `left_bars` | `3` | Hardcoded default. **Not** TODO-flagged anywhere. |
| Swing | `right_bars` | `3` | Hardcoded default. **Not** TODO-flagged anywhere. |
| BOS/CHoCH (`detect_bos_choch`) | `right_bars` | `3` (= swing's `right_bars`) | Hardcoded, reused, not independent. Unread by anything (see the BOS/CHoCH section). |
| BOS/CHoCH | `min_break_distance` | `0.0` (`DEFAULT_MIN_BREAK_DISTANCE`) | **Explicit TODO.** Raw price distance; applies only to the close-crossing check, not the wick-based swing-supersession check (documented scoping choice). |
| Order block (`detect_order_blocks`) | `min_ob_body_ratio` | `0.0` (`DEFAULT_MIN_OB_BODY_RATIO`) | **Explicit TODO.** Ratio of candle body to `_atr(window)`. Still no BOS-gating or mitigation check. Applies identically to the single and double order-block patterns. |
| FVG (`detect_fvg`) | `min_fvg_gap_ratio` | `0.0` (`DEFAULT_MIN_FVG_GAP_RATIO`) | **Explicit TODO.** Same ATR-ratio unit as the order block filter. |
| Trendline (`detect_trendline`) | `trendline_points` | `3` (`DEFAULT_TRENDLINE_POINTS`) | **Explicit TODO.** `2` reproduces a raw two-point line exactly; `3+` is a genuine least-squares fit. |
| Trendline | `min_trendline_break_distance` | `0.0` (`DEFAULT_MIN_TRENDLINE_BREAK_DISTANCE`) | **Explicit TODO.** Same pattern as BOS/CHoCH's `min_break_distance`. |
| Trendline | `right_bars` | `3` (= swing's `right_bars`) | Reused, not independent. |
| Channel (`detect_channel`) | `trendline_points`, `min_break_distance`, `right_bars` | Reused from `detect_trendline` | Same parameters, same values — the channel's anchor line *is* the trendline. |
| Channel | `min_channel_break_distance` | `0.0` (`DEFAULT_MIN_CHANNEL_BREAK_DISTANCE`) | **Explicit TODO.** Deliberately separate from `min_break_distance` even though both gate "close beyond a line" — different lines, different noise characteristics. |
| Fakeout/trap (`detect_fakeout`) | classic-fakeout candle window | fixed at 2 (structural, not a parameter) | Not configurable — baked into which indices (`[-1]`, `[-2]`) are read. |
| Fakeout/trap | `max_trap_retest_distance` | `float("inf")` (`DEFAULT_MAX_TRAP_RETEST_DISTANCE`) | **Explicit TODO.** The one inverted default in this file — gates a *maximum* distance, so "off" is infinity, not `0.0`. |
| Fakeout/trap | `trendline_points`, `right_bars` | Reused from `detect_trendline` | Not independent. |
| `compute_smc_features` | `lookback_bars` | `90` (`DEFAULT_LOOKBACK_BARS`) | **Explicit TODO** — tuned for daily stock bars in tradingagents-kr, flagged in-code as needing its own validation for 5m bars. |
| `SMCSignalAggregator` | `min_confluence` | none — constructor **raises `ValueError`** if not passed explicitly | **Explicit TODO by design** — no default exists on purpose; every caller must pass a value. Range is `-5..+5` (5 votes), up from `-4..+4` before the Phase 5 rewire. |

Every ratio/distance filter above defaults to a value that's a true no-op (`0.0` for
minimum-distance filters, `float("inf")` for `max_trap_retest_distance`'s maximum-
distance filter) — nothing about detector *output* changes by adding a parameter that
defaults off, only what's configurable. This was verified two different ways for two
different kinds of change, and the two shouldn't be confused: (1) the original
`min_ob_body_ratio`/`min_fvg_gap_ratio`/`min_break_distance` additions, and the later
numpy-array rewrite of the same functions' internals, were both verified **bit-
identical** against the prior behavior (`tests/test_smc.py`'s
`TestNewSizeFilterParametersDefaultToOff`, `tests/test_smc_vectorization_equivalence.py`)
— those were pure visibility/performance changes, not behavior changes. (2) The 2026-09
detector rebuild (order block redefinition, trendline/channel/trap additions,
aggregator rewire) is explicitly **not** bit-identical-verified against anything prior
— it's a deliberate behavior change measured against the reference video's spec, not
the old code's output. See [Design history](#design-history-why-this-looks-the-way-it-does)
for which category each change falls into.

Note the remaining asymmetry: `lookback_bars`, `min_confluence`, and every `min_*`/`max_*`
filter above carry explicit TODO comments (and for `min_confluence`, an enforced
required-argument guard) flagging them as unvalidated for 5-minute BTC/USD bars.
`swing_left_bars`/`swing_right_bars`/the BOS `right_bars` (reused by every detector
built on top of swings) are exactly as unvalidated, but nothing in the code flags them
the same way yet — they're still silently trusted at their literal default of `3`.

---

## Design history: why this looks the way it does

This section is the *why*, kept separate from the *what* above so the two don't get
tangled. Roughly chronological.

**Detector-level size/distance filters** (`min_ob_body_ratio`, `min_fvg_gap_ratio`,
`min_break_distance`) were added to make previously-invisible, hardcoded-off
assumptions visible and configurable — before these existed, *any* engulf/gap/break
qualified regardless of size, with no way to even ask "what if small ones didn't
count." They follow `lookback_bars`' pattern (a named parameter, a no-op default, an
explicit TODO), not `min_confluence`'s (no default, enforced-required) — a deliberate
choice: these were retrofitted onto existing detectors, not built into a from-scratch
constructor whose caller was always going to have to supply *something*.

**The numpy-array rewrite** (`detect_order_blocks`, `detect_fvg`, `detect_swings`,
`detect_bos_choch`) replaced `window.iloc[i]["field"]` pandas row access with
precomputed numpy arrays inside each function's hot loop — a pure performance fix (the
original `.iloc`-per-element pattern made `backtest/tune.py`'s grid search take an
estimated ~17 hours even parallelized across 16 cores; the rewrite brought that to a
practical ~1.2 hours). This changed *how* each value is read, not the comparisons,
arithmetic, or their order. Verified bit-identical two ways: a one-off run of both the
old and new implementations against the full real 1,313,022-row merged archive at
migration time, and a permanent regression test
(`tests/test_smc_vectorization_equivalence.py`) embedding a frozen copy of the original
`.iloc`-based implementation, diffed against the live one on a synthetic fixture.
`_atr()` itself was left completely untouched (still `pandas.rolling().mean()`)
specifically to avoid any risk of a different floating-point summation order;
`compute_smc_features` computes it once and shares it with both `detect_order_blocks`
and `detect_fvg` (previously computed twice, redundantly — over half the pre-shared
runtime).

**The 2026-09 detector rebuild** — the shape described throughout this document —
redesigned several detectors to match a reference video precisely, following a
decision recorded outside this repository (personal reasoning, not itself a source of
truth for the code: `../crypto-smc-bot-notes/decisions/2026-09-03-detector-rebuild-
decision.md`). Six phases, each its own commit:

1. **Order block redefinition** — body-engulf instead of close-break-high, plus the
   double order-block pattern. See [Order block](#order-block-detect_order_blocks).
2. **Real trendline detector** — `detect_trendline` replaces `classify_swing_trend`
   outright (not extended — see the retired-logic note in the Swing section). See
   [Trendline](#trendline-detect_trendline).
3. **Channel detector** — new, built on the same `_TrendlineTracker` trendline uses.
   See [Channel](#channel-detect_channel).
4. **Fakeout generalized + trap added** — the classic sweep-and-reclaim check now
   considers trendline/channel levels, not just the raw swing; trap is a genuinely new
   multi-bar pattern. This phase also promoted fakeout/trap from a veto to a vote — the
   single biggest behavioral change in the rebuild. See
   [Fakeout / trap](#fakeout--trap-detect_fakeout).
5. **Aggregator rewire** — 4 votes + 1 veto became 5 independent votes, confluence
   range `-4..+4` became `-5..+5`. See [Aggregator](#aggregator-signalssmc_aggregatorpy).
6. **Dashboard update** — BOS/CHoCH removed from the chart entirely; the toggle row
   became 5 chips matching the 5 votes; new trendline/channel renderers; a detector
   breakdown panel and a live-page risk summary panel, both ported from
   `design-reference/`'s mockup (which had already envisioned the 5-detector shape
   before the code caught up). See `viz/static/app.js` and `viz/control.py`.

**Deliberately not bit-identical.** Unlike the numpy rewrite above, none of the six
rebuild phases claim or verify bit-identical output against the pre-rebuild code —
that would be the wrong bar. Correctness for this rebuild is measured against the
reference video's spec (already reviewed against the code before the rebuild started),
not against what the old code happened to output. Where a phase needed a design
decision the video didn't specify exact math for (the double order-block pattern's
precise candle sequence, the trendline's least-squares-vs-two-point-line choice, the
channel's fixed-slope fit, direction conventions for channel touch/break events), that
choice is documented in the relevant section above and in the code's own docstrings,
not left implicit.

**Not done as part of this rebuild, on purpose:** the 256-combination tuning grid was
not re-run — the detector definitions changed, so the old grid's results describe a
system that no longer exists, but re-running it is explicitly a separate, later step
once this rebuild and the sizing/fee fix (`backtest/risk.py`'s `skip_if_capital_capped`)
are both done. See `docs/tuning-log.md`.
