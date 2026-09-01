"""
control.py
Start/stop/status for all three run modes (backtest, paper, live), sitting in front
of backtest/runner.py and live/trader.py. The dashboard is not read-only - this is
what its Backtest/Paper/Live controls actually call.

Today's scope (see the kickoff task order): backtest runs fully against pinned
snapshot data. Paper trading's start/stop/status plumbing is real, but wiring it to
Kraken's live WS feed is next session's work (viz/kraken_ws.py exists and is tested
standalone, just not hooked up here yet) - starting paper trading today raises
NotImplementedError rather than pretending to run. Live trading is permanently
refused here until live/trader.kill_switch_ready() is True, independent of whatever
the UI shows - see that function's docstring for what has to land first.
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from backtest.risk import RiskManager
from backtest.runner import FillConfig, run_backtest
from data.snapshot import load_snapshot
from live.trader import kill_switch_ready
from signals.smc_aggregator import SMCSignalAggregator

BroadcastFn = Callable[[dict[str, Any]], None]

RunStatus = Literal["idle", "running", "done", "error", "stopped"]


@dataclass
class BacktestState:
    status: RunStatus = "idle"
    run_id: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class PaperState:
    status: RunStatus = "idle"
    balance: float | None = None
    open_position: dict[str, Any] | None = None
    connection: Literal["healthy", "reconnecting", "dropped"] = "dropped"


class ControlPanel:
    """One instance shared by the FastAPI app (viz/server.py). Not thread-safe
    beyond what the GIL already gives simple attribute writes - fine for the single
    concurrent backtest / single paper session this dashboard supports."""

    def __init__(self, broadcast: BroadcastFn):
        self._broadcast = broadcast
        self.backtest = BacktestState()
        self.paper = PaperState()

    # ---- Backtest -------------------------------------------------------------

    def start_backtest(
        self,
        snapshot_path: Path,
        min_confluence: int,
        initial_balance: float,
        taker_fee_pct: float,
        slippage_bps: float,
    ) -> str:
        if self.backtest.status == "running":
            raise RuntimeError("A backtest is already running.")

        run_id = str(uuid.uuid4())
        self.backtest = BacktestState(status="running", run_id=run_id, progress={"candles_processed": 0})
        self._broadcast({"type": "backtest_status", "status": "running", "run_id": run_id})

        thread = threading.Thread(
            target=self._run_backtest_worker,
            args=(run_id, snapshot_path, min_confluence, initial_balance, taker_fee_pct, slippage_bps),
            daemon=True,
        )
        thread.start()
        return run_id

    def _run_backtest_worker(
        self,
        run_id: str,
        snapshot_path: Path,
        min_confluence: int,
        initial_balance: float,
        taker_fee_pct: float,
        slippage_bps: float,
    ) -> None:
        try:
            df = load_snapshot(snapshot_path)
            aggregator = SMCSignalAggregator(min_confluence=min_confluence)
            risk = RiskManager(initial_balance=initial_balance)
            fill_config = FillConfig(taker_fee_pct=taker_fee_pct, slippage_bps=slippage_bps)

            def on_progress(progress: dict[str, Any]) -> None:
                if self.backtest.run_id != run_id:
                    return  # a newer run superseded this one - drop stale progress
                self.backtest.progress = progress
                self._broadcast({"type": "backtest_progress", "run_id": run_id, **progress})

            result = run_backtest(df, aggregator, risk, fill_config, on_progress=on_progress)

            if self.backtest.run_id != run_id:
                return
            payload = {
                "trades": [
                    {
                        "side": t.side,
                        "entry_time": _iso(t.entry_time),
                        "exit_time": _iso(t.exit_time),
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "size": t.size,
                        "pnl": t.pnl,
                        "r_multiple": t.r_multiple,
                        "confluence": t.confluence,
                        "exit_reason": t.exit_reason,
                    }
                    for t in result.trades
                ],
                "equity_curve": result.equity_curve,
                "buy_hold_curve": result.buy_hold_curve,
                "metrics": result.metrics(),
            }
            self.backtest.status = "done"
            self.backtest.result = payload
            self._broadcast({"type": "backtest_done", "run_id": run_id, "result": payload})
        except Exception as e:  # noqa: BLE001 - any failure must reach the dashboard, not just the log
            if self.backtest.run_id == run_id:
                self.backtest.status = "error"
                self.backtest.error = str(e)
            self._broadcast({"type": "backtest_error", "run_id": run_id, "message": str(e)})

    # ---- Paper trading ----------------------------------------------------------

    def start_paper(self) -> None:
        if self.paper.status == "running":
            raise RuntimeError("Paper trading is already running.")
        # TODO(next session): wire viz/kraken_ws.stream_ohlc into a FillEngine here.
        # Deliberately not faked - starting paper trading today should fail loudly,
        # not silently do nothing while the UI claims it's running.
        raise NotImplementedError(
            "Paper trading isn't wired to the live Kraken feed yet - "
            "see viz/control.py's ControlPanel.start_paper TODO."
        )

    def stop_paper(self) -> None:
        self.paper.status = "stopped"
        self._broadcast({"type": "paper_status", "status": "stopped"})

    # ---- Live trading -------------------------------------------------------------

    def live_readiness(self) -> dict[str, Any]:
        """Read by the dashboard to decide whether the Start control is even
        rendered as enabled. live/trader.py independently re-checks
        kill_switch_ready() before placing any real order, so this isn't the only
        gate - just the one that drives the UI."""
        ready = kill_switch_ready()
        return {
            "kill_switch_ready": ready,
            "reason": "Kill switch not implemented yet" if not ready else "Ready",
        }

    def start_live(self) -> None:
        if not kill_switch_ready():
            raise PermissionError(
                "Live trading is disabled: kill_switch_ready() is False. "
                "See live/trader.py's docstring for what has to land first."
            )
        raise NotImplementedError("Live trading start path isn't built yet.")


def _iso(t: Any) -> str:
    return t.isoformat() if hasattr(t, "isoformat") else str(t)
