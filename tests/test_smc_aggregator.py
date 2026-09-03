"""
test_smc_aggregator.py
signals/smc_aggregator.py::SMCSignalAggregator tests - Phase 5 of the 2026-09
detector rebuild (5 independent votes, no veto). No dedicated test file existed
for this module before this rebuild; written fresh alongside the rewire.
"""
import unittest

from signals.smc_aggregator import SMCSignal, SMCSignalAggregator


def _features(**overrides) -> dict:
    """Baseline smc_features with every vote source empty/None (all-NEUTRAL),
    overridable per test."""
    base = {
        "order_blocks": [],
        "fvgs": [],
        "trendline": {"support": None, "resistance": None, "breaks": []},
        "channel": {"ascending": None, "descending": None, "events": []},
        "fakeout": {"fakeout": None, "trap": None},
    }
    base.update(overrides)
    return base


class TestMinConfluenceIsEnforcedRequired(unittest.TestCase):
    def test_raises_without_an_explicit_value(self):
        with self.assertRaises(ValueError):
            SMCSignalAggregator()

    def test_accepts_an_explicit_value(self):
        agg = SMCSignalAggregator(min_confluence=2)
        self.assertEqual(agg.min_confluence, 2)


class TestEmptyFeaturesAreNeutral(unittest.TestCase):
    def test_empty_dict_returns_neutral_with_zero_counts(self):
        agg = SMCSignalAggregator(min_confluence=2)
        signal = agg.aggregate({})
        self.assertEqual(signal.direction, "NEUTRAL")
        self.assertEqual(signal.confluence_score, 0)
        self.assertEqual(signal.bullish_count, 0)
        self.assertEqual(signal.bearish_count, 0)

    def test_all_vote_sources_empty_returns_neutral(self):
        agg = SMCSignalAggregator(min_confluence=1)
        signal = agg.aggregate(_features())
        self.assertEqual(signal.direction, "NEUTRAL")
        self.assertEqual(signal.confluence_score, 0)


class TestFiveIndependentVotes(unittest.TestCase):
    """Each of the 5 vote sources checked in isolation - order block, FVG,
    trendline, channel, fakeout/trap."""

    def setUp(self):
        self.agg = SMCSignalAggregator(min_confluence=1)

    def test_order_block_vote_reads_the_most_recent_one(self):
        feats = _features(order_blocks=[
            {"type": "bearish", "pattern": "single"},
            {"type": "bullish", "pattern": "single"},
        ])
        signal = self.agg.aggregate(feats)
        self.assertEqual(signal.contributing_factors["order_block"], "bullish")
        self.assertEqual(signal.bullish_count, 1)
        self.assertEqual(signal.bearish_count, 0)

    def test_fvg_vote_reads_the_most_recent_one(self):
        feats = _features(fvgs=[{"type": "bullish"}, {"type": "bearish"}])
        signal = self.agg.aggregate(feats)
        self.assertEqual(signal.contributing_factors["fvg"], "bearish")
        self.assertEqual(signal.bearish_count, 1)

    def test_trendline_vote_reads_the_most_recent_break_direction(self):
        feats = _features(trendline={
            "support": None, "resistance": None,
            "breaks": [
                {"type": "SUPPORT_BREAK", "direction": "bearish"},
                {"type": "RESISTANCE_BREAK", "direction": "bullish"},
            ],
        })
        signal = self.agg.aggregate(feats)
        self.assertEqual(signal.contributing_factors["trendline"], "bullish")
        self.assertEqual(signal.bullish_count, 1)

    def test_trendline_vote_is_none_when_there_are_no_breaks(self):
        feats = _features(trendline={"support": {"slope": 1}, "resistance": None, "breaks": []})
        signal = self.agg.aggregate(feats)
        self.assertIsNone(signal.contributing_factors["trendline"])

    def test_channel_vote_reads_the_most_recent_break_event(self):
        feats = _features(channel={
            "ascending": None, "descending": None,
            "events": [
                {"type": "CHANNEL_TOUCH", "boundary": "upper", "direction": "bearish"},
                {"type": "CHANNEL_BREAK", "boundary": "lower", "direction": "bearish"},
            ],
        })
        signal = self.agg.aggregate(feats)
        self.assertEqual(signal.contributing_factors["channel"], "bearish")
        self.assertEqual(signal.bearish_count, 1)

    def test_channel_touch_events_do_not_vote(self):
        # Only a CHANNEL_TOUCH exists (no CHANNEL_BREAK) - the vote must be None,
        # not derived from the touch's direction.
        feats = _features(channel={
            "ascending": None, "descending": None,
            "events": [{"type": "CHANNEL_TOUCH", "boundary": "upper", "direction": "bearish"}],
        })
        signal = self.agg.aggregate(feats)
        self.assertIsNone(signal.contributing_factors["channel"])
        self.assertEqual(signal.bearish_count, 0)

    def test_fakeout_vote_from_the_classic_sweep_when_no_trap(self):
        feats = _features(fakeout={"fakeout": {"type": "BULL_FAKEOUT", "source": "swing"}, "trap": None})
        signal = self.agg.aggregate(feats)
        self.assertEqual(signal.contributing_factors["fakeout_trap"], "bullish")
        self.assertEqual(signal.bullish_count, 1)

    def test_fakeout_vote_from_bear_fakeout(self):
        feats = _features(fakeout={"fakeout": {"type": "BEAR_FAKEOUT", "source": "swing"}, "trap": None})
        signal = self.agg.aggregate(feats)
        self.assertEqual(signal.contributing_factors["fakeout_trap"], "bearish")
        self.assertEqual(signal.bearish_count, 1)

    def test_trap_takes_priority_over_a_simultaneous_classic_fakeout(self):
        feats = _features(fakeout={
            "fakeout": {"type": "BEAR_FAKEOUT", "source": "swing"},
            "trap": {"type": "SUPPORT_TRAP", "direction": "bullish"},
        })
        signal = self.agg.aggregate(feats)
        self.assertEqual(signal.contributing_factors["fakeout_trap"], "bullish")
        self.assertEqual(signal.bullish_count, 1)
        self.assertEqual(signal.bearish_count, 0)


class TestConfluenceRangeAndThreshold(unittest.TestCase):
    def _all_bullish_features(self) -> dict:
        return _features(
            order_blocks=[{"type": "bullish"}],
            fvgs=[{"type": "bullish"}],
            trendline={"support": None, "resistance": None,
                       "breaks": [{"type": "RESISTANCE_BREAK", "direction": "bullish"}]},
            channel={"ascending": None, "descending": None,
                     "events": [{"type": "CHANNEL_BREAK", "boundary": "upper", "direction": "bullish"}]},
            fakeout={"fakeout": {"type": "BULL_FAKEOUT", "source": "swing"}, "trap": None},
        )

    def test_all_five_bullish_gives_confluence_plus_five(self):
        agg = SMCSignalAggregator(min_confluence=5)
        signal = agg.aggregate(self._all_bullish_features())
        self.assertEqual(signal.confluence_score, 5)
        self.assertEqual(signal.bullish_count, 5)
        self.assertEqual(signal.bearish_count, 0)
        self.assertEqual(signal.direction, "BULLISH")

    def test_below_threshold_is_neutral(self):
        agg = SMCSignalAggregator(min_confluence=3)
        # Only 2 of 5 vote bullish - matches the video's own informal "2 of 5"
        # anecdote, deliberately below a stricter min_confluence=3 threshold here.
        feats = _features(
            order_blocks=[{"type": "bullish"}],
            fvgs=[{"type": "bullish"}],
        )
        signal = agg.aggregate(feats)
        self.assertEqual(signal.confluence_score, 2)
        self.assertEqual(signal.direction, "NEUTRAL")

    def test_at_threshold_is_directional(self):
        agg = SMCSignalAggregator(min_confluence=2)
        feats = _features(
            order_blocks=[{"type": "bullish"}],
            fvgs=[{"type": "bullish"}],
        )
        signal = agg.aggregate(feats)
        self.assertEqual(signal.direction, "BULLISH")

    def test_bearish_threshold_is_symmetric(self):
        agg = SMCSignalAggregator(min_confluence=2)
        feats = _features(
            order_blocks=[{"type": "bearish"}],
            fvgs=[{"type": "bearish"}],
        )
        signal = agg.aggregate(feats)
        self.assertEqual(signal.confluence_score, -2)
        self.assertEqual(signal.direction, "BEARISH")


class TestNoVetoMechanismExists(unittest.TestCase):
    """Phase 5's core behavioral change: a fakeout/trap agreeing OR disagreeing
    with the other 4 votes is just another vote now - it can never force an
    otherwise-non-NEUTRAL call back to NEUTRAL the way the old veto did."""

    def test_opposing_fakeout_no_longer_forces_neutral(self):
        agg = SMCSignalAggregator(min_confluence=1)
        feats = _features(
            order_blocks=[{"type": "bullish"}],
            fvgs=[{"type": "bullish"}],
            fakeout={"fakeout": {"type": "BEAR_FAKEOUT", "source": "swing"}, "trap": None},
        )
        signal = agg.aggregate(feats)
        # Old veto model: BEAR_FAKEOUT opposing a BULLISH call -> forced NEUTRAL.
        # New model: 2 bullish votes vs 1 bearish vote -> net +1, still BULLISH.
        self.assertEqual(signal.bullish_count, 2)
        self.assertEqual(signal.bearish_count, 1)
        self.assertEqual(signal.confluence_score, 1)
        self.assertEqual(signal.direction, "BULLISH")

    def test_smc_signal_has_no_veto_fields(self):
        signal = SMCSignal(direction="NEUTRAL", confluence_score=0, bullish_count=0, bearish_count=0)
        self.assertFalse(hasattr(signal, "veto"))
        self.assertFalse(hasattr(signal, "veto_reason"))


class TestToAnchorText(unittest.TestCase):
    def test_includes_direction_and_confluence_and_only_non_none_factors(self):
        agg = SMCSignalAggregator(min_confluence=1)
        feats = _features(order_blocks=[{"type": "bullish"}])
        signal = agg.aggregate(feats)
        text = signal.to_anchor_text()
        self.assertIn("BULLISH", text)
        self.assertIn("confluence +1", text)
        self.assertIn("Order block: bullish", text)
        self.assertNotIn("FVG:", text)  # None factor omitted


if __name__ == "__main__":
    unittest.main()
