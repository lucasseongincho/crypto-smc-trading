"""
test_control_paper.py
Tests ControlPanel's paper-trading candle-close detection (viz/control.py). Kraken's
WS OHLC channel repeats "update" messages for the same interval_begin as a bar
forms and only moves to a new interval_begin once it closes - _on_paper_candle()
has to coalesce those into one feed per closed bar, and skip stale bars replayed by
a post-reconnect snapshot. Exercises those private methods directly (no real network
or asyncio.create_task needed - they're synchronous callbacks).
"""
import unittest

from backtest.risk import RiskManager
from backtest.runner import FillConfig, FillEngine
from signals.smc_aggregator import SMCSignalAggregator
from viz.control import ControlPanel


def _candle(interval_begin: str, close: float, open_: float = None, high: float = None, low: float = None) -> dict:
    open_ = open_ if open_ is not None else close
    high = high if high is not None else close
    low = low if low is not None else close
    return {"interval_begin": interval_begin, "open": open_, "high": high, "low": low, "close": close}


def _make_panel() -> ControlPanel:
    broadcasts = []
    panel = ControlPanel(broadcast=broadcasts.append)
    panel._paper_engine = FillEngine(
        SMCSignalAggregator(min_confluence=2), RiskManager(initial_balance=20_000.0), FillConfig(),
    )
    panel.broadcasts = broadcasts  # stash for assertions
    return panel


class TestPartialUpdateCoalescing(unittest.TestCase):
    def test_only_feeds_engine_once_per_closed_bar_using_the_last_value_seen(self):
        panel = _make_panel()
        fed = []
        panel._paper_engine.on_candle = lambda time, o, h, l, c: fed.append((time, o, h, l, c)) or None

        # Same bar (00:00) updates three times as it forms, then a new bar (00:05) starts.
        panel._on_paper_candle(_candle("2026-01-01T00:00:00Z", 100.0))
        panel._on_paper_candle(_candle("2026-01-01T00:00:00Z", 101.0))
        panel._on_paper_candle(_candle("2026-01-01T00:00:00Z", 102.5))  # this is the true close
        self.assertEqual(fed, [])  # not fed yet - the 00:00 bar hasn't closed

        panel._on_paper_candle(_candle("2026-01-01T00:05:00Z", 103.0))  # 00:05 arriving closes 00:00
        self.assertEqual(len(fed), 1)
        self.assertEqual(fed[0][4], 102.5)  # fed the LAST close seen for 00:00, not the first or an average

    def test_second_bar_only_feeds_on_a_third_bars_arrival(self):
        panel = _make_panel()
        fed = []
        panel._paper_engine.on_candle = lambda time, o, h, l, c: fed.append(c) or None

        panel._on_paper_candle(_candle("2026-01-01T00:00:00Z", 100.0))
        panel._on_paper_candle(_candle("2026-01-01T00:05:00Z", 105.0))  # closes 00:00 -> fed=[100.0]
        panel._on_paper_candle(_candle("2026-01-01T00:05:00Z", 106.0))  # still forming 00:05
        panel._on_paper_candle(_candle("2026-01-01T00:10:00Z", 110.0))  # closes 00:05 -> fed=[100.0, 106.0]

        self.assertEqual(fed, [100.0, 106.0])


class TestReconnectReplaySkipped(unittest.TestCase):
    def test_a_bar_older_than_the_currently_tracked_one_is_ignored_not_refed(self):
        panel = _make_panel()
        fed = []
        panel._paper_engine.on_candle = lambda time, o, h, l, c: fed.append(c) or None

        panel._on_paper_candle(_candle("2026-01-01T00:00:00Z", 100.0))
        panel._on_paper_candle(_candle("2026-01-01T00:05:00Z", 105.0))  # closes 00:00 -> fed=[100.0]

        # Reconnect snapshot replays the already-closed 00:00 bar - must not re-fire.
        panel._on_paper_candle(_candle("2026-01-01T00:00:00Z", 999.0))
        self.assertEqual(fed, [100.0])

        # A genuinely new bar afterwards still closes normally (00:05, at its
        # pre-replay value of 105.0, since nothing legitimately updated it).
        panel._on_paper_candle(_candle("2026-01-01T00:10:00Z", 110.0))
        self.assertEqual(fed, [100.0, 105.0])

    def test_replaying_the_still_open_bar_is_treated_as_a_fresh_update_not_ignored(self):
        """The currently-tracked (not yet closed) bar is a different case from an
        already-closed one: a reconnect snapshot re-delivering it is legitimate
        fresher data for a bar that genuinely hasn't closed yet, not a stale replay -
        it should overwrite our view of it, same as any other in-progress update."""
        panel = _make_panel()
        fed = []
        panel._paper_engine.on_candle = lambda time, o, h, l, c: fed.append(c) or None

        panel._on_paper_candle(_candle("2026-01-01T00:00:00Z", 100.0))
        panel._on_paper_candle(_candle("2026-01-01T00:05:00Z", 105.0))  # closes 00:00, opens 00:05

        panel._on_paper_candle(_candle("2026-01-01T00:05:00Z", 999.0))  # reconnect re-delivers 00:05
        panel._on_paper_candle(_candle("2026-01-01T00:10:00Z", 110.0))  # closes 00:05

        self.assertEqual(fed, [100.0, 999.0])  # the fresher 00:05 value won, as intended


class TestStateAndBroadcast(unittest.TestCase):
    def test_closing_a_bar_updates_paper_state_and_broadcasts_status(self):
        panel = _make_panel()

        panel._on_paper_candle(_candle("2026-01-01T00:00:00Z", 100.0))
        panel._on_paper_candle(_candle("2026-01-01T00:05:00Z", 105.0))  # closes the 00:00 bar

        self.assertEqual(panel.paper.balance, panel._paper_engine.risk.current_balance)
        status_broadcasts = [b for b in panel.broadcasts if b.get("type") == "paper_status"]
        self.assertGreaterEqual(len(status_broadcasts), 1)

    def test_connection_state_change_updates_paper_connection_and_broadcasts(self):
        panel = _make_panel()
        panel.paper.connection = "reconnecting"

        panel._on_paper_connection_state("healthy", {})

        self.assertEqual(panel.paper.connection, "healthy")
        self.assertTrue(any(b.get("type") == "paper_status" for b in panel.broadcasts))


class TestRiskSummary(unittest.TestCase):
    """viz/control.py::ControlPanel.risk_summary() - Phase 6 of the 2026-09
    detector rebuild, ported into the live panel per the dashboard rebuild
    decision. Confirms honest not-configured reporting for fields that aren't
    real settings anywhere in this codebase yet (kill_switch_ready() is a
    hardcoded stub - see live/trader.py), rather than fabricated numbers."""

    def test_confluence_threshold_is_not_set_before_any_run(self):
        panel = _make_panel()
        summary = panel.risk_summary()
        self.assertEqual(summary["confluence_threshold_display"], "not set - run a backtest first")

    def test_confluence_threshold_reflects_the_most_recent_backtest_start(self):
        panel = _make_panel()
        panel._last_min_confluence = 3
        summary = panel.risk_summary()
        self.assertEqual(summary["confluence_threshold_display"], "3 of 5")

    def test_daily_loss_limit_and_drawdown_halt_are_honestly_not_configured(self):
        panel = _make_panel()
        summary = panel.risk_summary()
        self.assertIsNone(summary["daily_loss_limit"])
        self.assertIsNone(summary["max_drawdown_halt_pct"])

    def test_balance_at_arm_is_none_while_the_kill_switch_is_not_ready(self):
        # kill_switch_ready() is a hardcoded False stub today - risk_summary()
        # must not attempt a real Kraken balance call in that state.
        panel = _make_panel()
        summary = panel.risk_summary()
        self.assertIsNone(summary["balance_at_arm"])

    def test_position_size_pct_reflects_riskmanagers_own_default(self):
        from backtest.risk import RiskManager
        panel = _make_panel()
        summary = panel.risk_summary()
        self.assertAlmostEqual(summary["position_size_pct"], RiskManager().risk_pct * 100)


if __name__ == "__main__":
    unittest.main()
