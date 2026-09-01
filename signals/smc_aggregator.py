"""
smc_aggregator.py
Rule-based aggregation of data/smc.py's feature dict into a single directional
SMCSignal. No LLM involved - this is the entire decision layer for the SMC-only
strategy.

Ported from tradingagents-kr's signals/smc_aggregator.py (which itself carried over
the original crypto bot's strategy.py signal-summing logic), with two behaviors kept
from that port:
  1. No HTF (higher-timeframe) bias comparison. The original crypto bot compared a
     5m read against a separate 6h candle fetch; tradingagents-kr's data layer only
     ever passed a single-timeframe DataFrame, so that comparison was dropped rather
     than faked with a substitute indicator. Multi-timeframe confirmation would need
     its own explicit data-layer support before re-adding here.
  2. Fakeout is a veto, not a bonus point. The original crypto bot's FakeoutFilter
     just added +1 to the score; here, a fakeout opposing the tentative direction
     invalidates it to NEUTRAL outright.
"""
from dataclasses import dataclass, field
from typing import Any

_FACTOR_LABELS = {
    "order_block": "Recent order block",
    "fvg": "Recent FVG",
    "swing_trend": "Swing trend",
    "structure_bias": "Structure (BOS/CHoCH) bias",
}


@dataclass
class SMCSignal:
    direction: str                 # "BULLISH" | "BEARISH" | "NEUTRAL"
    confluence_score: int          # bullish_count - bearish_count, range -4..+4
    bullish_count: int
    bearish_count: int
    veto: bool
    veto_reason: str | None
    contributing_factors: dict[str, Any] = field(default_factory=dict)

    def to_anchor_text(self) -> str:
        lines = [
            f"[SMC rule-based verdict] {self.direction} "
            f"(confluence {self.confluence_score:+d} = bullish {self.bullish_count} - bearish {self.bearish_count})"
        ]
        if self.veto:
            lines.append(f"⚠ Fakeout veto triggered: {self.veto_reason}")

        detail_lines = [
            f"  - {_FACTOR_LABELS.get(k, k)}: {v}"
            for k, v in self.contributing_factors.items()
            if v not in (None, "RANGING")
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
        4 detectors vote (order block / fvg / swing_trend / structure_bias), so the
        range is -4..+4.

        TODO: tradingagents-kr's equivalent value (3) was derived from a 10-year daily
        stock-bar backtest across 8 tickers and is explicitly documented there as
        in-sample and domain-specific (daily bars, not 5-minute crypto bars, and stocks
        not BTC). Nothing has validated a default for this project's 5m/BTC-USD
        combination yet - left unset (None) on purpose rather than silently inheriting
        a stock-tuned number. Callers must pass an explicit value (backtest/runner.py
        should treat it as a swept parameter, not a constant) until that validation
        happens.
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
                veto=False, veto_reason=None, contributing_factors={},
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

        swing_trend = smc_features.get("swing_trend")
        factors["swing_trend"] = swing_trend
        if swing_trend == "UPTREND":
            bullish += 1
        elif swing_trend == "DOWNTREND":
            bearish += 1

        structure_bias = smc_features.get("structure_bias")
        factors["structure_bias"] = structure_bias
        if structure_bias == "bullish":
            bullish += 1
        elif structure_bias == "bearish":
            bearish += 1

        confluence = bullish - bearish
        if confluence >= self.min_confluence:
            direction = "BULLISH"
        elif confluence <= -self.min_confluence:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        veto = False
        veto_reason = None
        fakeout = smc_features.get("fakeout")
        if fakeout and direction != "NEUTRAL":
            fk_type = fakeout.get("type")
            if fk_type == "BULL_FAKEOUT" and direction == "BEARISH":
                veto_reason = (
                    f"Close recovered above the prior swing low ({fakeout.get('swept_level')}) after "
                    "sweeping it - bullish fakeout (liquidity sweep) invalidates the bearish call"
                )
                veto = True
                direction = "NEUTRAL"
            elif fk_type == "BEAR_FAKEOUT" and direction == "BULLISH":
                veto_reason = (
                    f"Close gave back the prior swing high ({fakeout.get('swept_level')}) after "
                    "sweeping it - bearish fakeout (liquidity sweep) invalidates the bullish call"
                )
                veto = True
                direction = "NEUTRAL"

        return SMCSignal(
            direction=direction,
            confluence_score=confluence,
            bullish_count=bullish,
            bearish_count=bearish,
            veto=veto,
            veto_reason=veto_reason,
            contributing_factors=factors,
        )
