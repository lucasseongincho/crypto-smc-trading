"""
control.py
Start/stop/status for all three run modes (backtest, paper, live), sitting in front
of backtest/runner.py and live/trader.py. The dashboard is not read-only - this is
what its Backtest/Paper/Live controls actually call.

Backtest runs fully against pinned snapshot data. Paper trading streams Kraken's
live public WS feed (viz/kraken_ws.py) through the same FillEngine backtest uses -
one simulation engine, two data sources, per the original architecture decision.
Live trading is permanently refused here until live/trader.kill_switch_ready() is
True, independent of whatever the UI shows - see that function's docstring for what
has to land first.
"""
import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from backtest.risk import RiskManager
from backtest.runner import FillConfig, FillEngine, Position, run_backtest
from data.snapshot import load_snapshot
from live.trader import kill_switch_ready
from signals.smc_aggregator import SMCSignalAggregator
from viz.kraken_ws import stream_ohlc

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

        self._paper_task: asyncio.Task | None = None
        self._paper_stop_event: asyncio.Event | None = None
        self._paper_engine: FillEngine | None = None
        self._paper_current_interval_ts: pd.Timestamp | None = None
        self._paper_last_candle: dict[str, Any] | None = None

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
    #
    # Kraken's WS OHLC channel has no "candle closed" flag - it repeats "update"
    # messages for the same interval_begin as a bar forms, then moves on to a new
    # interval_begin once it closes. _on_paper_candle() detects that transition and
    # feeds FillEngine the *final* value seen for the bar that just closed, not
    # every partial update (feeding partial closes would mean the signal/fill logic
    # sees a different "close" price every few seconds - the same look-ahead-bias
    # problem the backtest side is careful to avoid). A reconnect (viz/kraken_ws.py)
    # replays a snapshot burst that can include the bar already closed just before
    # the drop - that one is recognized by its interval_begin being strictly older
    # than what's already been processed and is skipped, not re-fed. A replayed
    # interval_begin equal to the bar currently being tracked is different: that bar
    # hasn't closed yet, so the replay is legitimate fresher data for it, not a
    # stale duplicate - it's treated as just another in-progress update.

    def start_paper(
        self,
        min_confluence: int,
        initial_balance: float,
        taker_fee_pct: float,
        slippage_bps: float,
        symbol: str = "BTC/USD",
        interval_minutes: int = 5,
    ) -> None:
        if self.paper.status == "running":
            raise RuntimeError("Paper trading is already running.")

        aggregator = SMCSignalAggregator(min_confluence=min_confluence)
        risk = RiskManager(initial_balance=initial_balance)
        fill_config = FillConfig(taker_fee_pct=taker_fee_pct, slippage_bps=slippage_bps)

        self._paper_engine = FillEngine(aggregator, risk, fill_config)
        self._paper_stop_event = asyncio.Event()
        self._paper_current_interval_ts = None
        self._paper_last_candle = None

        self.paper = PaperState(status="running", balance=initial_balance, open_position=None, connection="reconnecting")
        self._broadcast({"type": "paper_status"})

        self._paper_task = asyncio.create_task(stream_ohlc(
            symbol, interval_minutes, self._on_paper_candle,
            on_state=self._on_paper_connection_state, stop_event=self._paper_stop_event,
        ))

    async def stop_paper(self) -> None:
        if self.paper.status != "running":
            return
        if self._paper_stop_event is not None:
            self._paper_stop_event.set()
        if self._paper_task is not None:
            await self._paper_task
        self.paper.status = "stopped"
        self.paper.connection = "dropped"
        self._broadcast({"type": "paper_status"})

    def _on_paper_connection_state(self, state: str, info: dict[str, Any]) -> None:
        self.paper.connection = state  # type: ignore[assignment]
        self._broadcast({"type": "paper_status"})

    def _on_paper_candle(self, candle: dict[str, Any]) -> None:
        interval_begin = candle.get("interval_begin")
        ts = pd.Timestamp(interval_begin)

        if self._paper_current_interval_ts is None:
            self._paper_current_interval_ts = ts
            self._paper_last_candle = candle
            return

        if ts < self._paper_current_interval_ts:
            return  # a post-reconnect snapshot replaying an already-closed bar

        if ts == self._paper_current_interval_ts:
            self._paper_last_candle = candle  # still the same forming bar
            return

        closed_candle = self._paper_last_candle
        self._paper_current_interval_ts = ts
        self._paper_last_candle = candle
        self._feed_closed_paper_candle(closed_candle)

    def _feed_closed_paper_candle(self, candle: dict[str, Any]) -> None:
        engine = self._paper_engine
        if engine is None:
            return

        event = engine.on_candle(
            pd.Timestamp(candle["interval_begin"]),
            float(candle["open"]), float(candle["high"]), float(candle["low"]), float(candle["close"]),
        )

        self.paper.balance = engine.risk.current_balance
        self.paper.open_position = _position_to_dict(engine.position)
        self._broadcast({"type": "paper_status"})
        if event:
            self._broadcast({"type": "paper_trade", **event})

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


def _position_to_dict(position: Position | None) -> dict[str, Any] | None:
    if position is None:
        return None
    return {
        "side": position.side, "entry_price": position.entry_price,
        "stop_loss": position.stop_loss, "take_profit": position.take_profit,
        "size": position.size,
    }
