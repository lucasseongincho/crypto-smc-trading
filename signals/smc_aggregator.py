"""
smc_aggregator.py
Rule-based aggregation of data/smc.py's feature dict into a single directional
SMCSignal. No LLM involved - this is the entire decision layer for the SMC-only
strategy.

**2026-09 detector rebuild, Phase 5: 5 independent votes, no veto.** Before this
rebuild, this module ran a "4 votes + 1 veto" model: order block / FVG /
swing_trend / structure_bias each independently added +1/-1 to a confluence
score, and a separate fakeout check could only ever *cancel* an already-
non-NEUTRAL direction (a veto), never independently push the score itself. That
model is gone. The current model is **5 independent votes**, each worth +1/-1,
no veto mechanism at all: order block, FVG, trendline, channel, fakeout/trap.
`swing_trend` and `structure_bias` (BOS/CHoCH) are no longer read here at all -
`classify_swing_trend` was removed outright in Phase 2 (replaced by real
trendlines), and BOS/CHoCH's structure_events, while the underlying code still
exists for the swing detection it shares with trendline/channel fitting, no
longer feeds a confluence vote (see docs/detector-logic.md's Phase 5 section for
the full before/after). This is the single biggest behavioral change in the
rebuild - see docs/detector-logic.md's fakeout/trap section for the veto-vs-vote
distinction spelled out in detail. Every backtest result from before this phase
used the old model; every one after uses this one - they are not comparable,
and the tuning grid needs a full re-run once the whole rebuild (through Phase 6)
lands (deliberately not re-run yet).

Ported originally from tradingagents-kr's signals/smc_aggregator.py (which itself
carried over the original crypto bot's strategy.py signal-summing logic); one
inherited behavior still holds:
  - No HTF (higher-timeframe) bias comparison. The original crypto bot compared a
    5m read against a separate 6h candle fetch; tradingagents-kr's data layer only
    ever passed a single-timeframe DataFrame, so that comparison was dropped rather
    than faked with a substitute indicator. Multi-timeframe confirmation would need
    its own explicit data-layer support before re-adding here.
"""
from dataclasses import dataclass, field
from typing import Any

_FACTOR_LABELS = {
    "order_block": "Order block",
    "fvg": "FVG",
    "trendline": "Trendline",
    "channel": "Channel",
    "fakeout_trap": "Fakeout/trap",
}


@dataclass
class SMCSignal:
    direction: str                 # "BULLISH" | "BEARISH" | "NEUTRAL"
    confluence_score: int          # bullish_count - bearish_count, range -5..+5 (5 votes, no veto)
    bullish_count: int
    bearish_count: int
    contributing_factors: dict[str, Any] = field(default_factory=dict)

    def to_anchor_text(self) -> str:
        lines = [
            f"[SMC rule-based verdict] {self.direction} "
            f"(confluence {self.confluence_score:+d} = bullish {self.bullish_count} - bearish {self.bearish_count})"
        ]

        detail_lines = [
            f"  - {_FACTOR_LABELS.get(k, k)}: {v}"
            for k, v in self.contributing_factors.items()
            if v is not None
        ]
        if detail_lines:
            lines.append("Basis:")
            lines.extend(detail_lines)

        return "\n".join(lines)


class SMCSignalAggregator:
    """Stateless rule-based SMC signal aggregator - the same smc_features always
    produces the same SMCSignal."""

    def __init__(self, min_confluence: int | None = None):
        """
        min_confluence: minimum net confluence score required for a directional call.
        5 detectors vote (order block / fvg / trendline / channel / fakeout_trap),
        so the range is -5..+5 - up from -4..+4 before the Phase 5 rebuild (the old
        4-vote-plus-veto model had no equivalent range, since the veto wasn't a vote).

        TODO: tradingagents-kr's old equivalent value (3, against the prior 4-vote
        model) was itself derived from a 10-year daily stock-bar backtest across 8
        tickers, already flagged there as in-sample and domain-specific - and that
        number doesn't even transfer arithmetically now that the vote count changed
        from 4 to 5. The reference video's own informal practice was "confirm with
        at least 2 of the 5 concepts" - noted here **as a loose reference point
        only**, not a validated value, same treatment lookback_bars' TODO gets
        elsewhere in this codebase (data/smc.py). Nothing has validated a default
        for this project's 5m/BTC-USD combination with the new 5-vote model - left
        unset (None) on purpose rather than silently inheriting a stock-tuned or
        video-anecdotal number. Callers must pass an explicit value (backtest/tune.py
        should treat it as a swept parameter, not a constant) until that validation
        happens - the tuning grid itself is being re-run after this whole rebuild
        lands, not before.
        """
        if min_confluence is None:
            raise ValueError(
                "min_confluence has no validated default for 5m BTC/USD yet - pass an "
                "explicit value (see the TODO in SMCSignalAggregator.__init__)."
            )
        self.min_confluence = min_confluence

    def aggregate(self, smc_features: dict[str, Any]) -> SMCSignal:
        if not smc_features:
            return SMCSignal(
                direction="NEUTRAL", confluence_score=0, bullish_count=0, bearish_count=0,
                contributing_factors={},
            )

        factors: dict[str, Any] = {}
        bullish = 0
        bearish = 0

        order_blocks = smc_features.get("order_blocks") or []
        last_ob_dir = order_blocks[-1]["type"] if order_blocks else None
        factors["order_block"] = last_ob_dir
        if last_ob_dir == "bullish":
            bullish += 1
        elif last_ob_dir == "bearish":
            bearish += 1

        fvgs = smc_features.get("fvgs") or []
        last_fvg_dir = fvgs[-1]["type"] if fvgs else None
        factors["fvg"] = last_fvg_dir
        if last_fvg_dir == "bullish":
            bullish += 1
        elif last_fvg_dir == "bearish":
            bearish += 1

        # Trendline vote: the direction of the most recent SUPPORT_BREAK/
        # RESISTANCE_BREAK event anywhere in this window's breaks list (not just
        # a break on the very last bar) - same "most recent event's direction,
        # whatever bar it happened on" pattern the retired structure_bias vote
        # used, just against detect_trendline's breaks instead of
        # detect_bos_choch's structure_events.
        trendline = smc_features.get("trendline") or {}
        trendline_breaks = trendline.get("breaks") or []
        trendline_dir = trendline_breaks[-1]["direction"] if trendline_breaks else None
        factors["trendline"] = trendline_dir
        if trendline_dir == "bullish":
            bullish += 1
        elif trendline_dir == "bearish":
            bearish += 1

        # Channel vote: the direction of the most recent CHANNEL_BREAK event
        # (either channel, either boundary) - CHANNEL_TOUCH events are
        # deliberately excluded from the vote (informational/dashboard-only;
        # touch and break carry *opposite* directional implications at the same
        # boundary per detect_channel's own convention, and mixing the two into
        # one vote would need its own weighting design). Note this can and does
        # correlate with the trendline vote above - an ascending channel's lower
        # boundary literally *is* the support line, so a SUPPORT_BREAK often
        # fires both votes in the same direction on the same bar. That's an
        # accepted consequence of Phase 3's design (see docs/detector-logic.md's
        # Channel section), not a bug to work around here.
        channel = smc_features.get("channel") or {}
        channel_breaks = [e for e in (channel.get("events") or []) if e["type"] == "CHANNEL_BREAK"]
        channel_dir = channel_breaks[-1]["direction"] if channel_breaks else None
        factors["channel"] = channel_dir
        if channel_dir == "bullish":
            bullish += 1
        elif channel_dir == "bearish":
            bearish += 1

        # Fakeout/trap vote - ONE combined vote source, per the rebuild's own
        # "5 independent votes: ... fakeout/trap" framing (not two separate
        # votes). A confirmed trap takes priority over the classic fakeout when
        # both happen to be present (a trap is the rarer, higher-conviction
        # pattern - see data/smc.py's _detect_trap docstring).
        fakeout_trap = smc_features.get("fakeout") or {}
        trap = fakeout_trap.get("trap")
        fakeout = fakeout_trap.get("fakeout")
        if trap is not None:
            fakeout_trap_dir = trap["direction"]
        elif fakeout is not None:
            fakeout_trap_dir = "bullish" if fakeout["type"] == "BULL_FAKEOUT" else "bearish"
        else:
            fakeout_trap_dir = None
        factors["fakeout_trap"] = fakeout_trap_dir
        if fakeout_trap_dir == "bullish":
            bullish += 1
        elif fakeout_trap_dir == "bearish":
            bearish += 1

        confluence = bullish - bearish
        if confluence >= self.min_confluence:
            direction = "BULLISH"
        elif confluence <= -self.min_confluence:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return SMCSignal(
            direction=direction,
            confluence_score=confluence,
            bullish_count=bullish,
            bearish_count=bearish,
            contributing_factors=factors,
        )
