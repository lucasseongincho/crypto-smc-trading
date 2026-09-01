"""
server.py
FastAPI + WebSocket dashboard backend. REST endpoints drive viz/control.py's
start/stop/status for backtest/paper/live; the WebSocket channel broadcasts
progress, fills, and status changes as they happen (an async backend so Kraken's own
WS feed can flow through without a thread-to-event-loop bridge - see the kickoff
decision on why this replaced the original Flask+SSE viewer).

    python -m viz.server
    open http://localhost:8000
"""
import asyncio
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

VIZ_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VIZ_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))  # let `python -m viz.server` import project packages

from data.smc import DEFAULT_LOOKBACK_BARS, compute_smc_features  # noqa: E402
from data.snapshot import list_snapshots, load_snapshot  # noqa: E402
from viz.control import ControlPanel  # noqa: E402

app = FastAPI(title="SMC Bot Dashboard")


class ConnectionManager:
    """Broadcasts JSON payloads to every connected dashboard client. ControlPanel
    calls broadcast() from a background worker thread (the backtest thread), not the
    event loop, so it hops over via run_coroutine_threadsafe rather than assuming
    it's already running on the loop."""

    def __init__(self) -> None:
        self._sockets: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._sockets.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._sockets.discard(ws)

    def broadcast(self, payload: dict[str, Any]) -> None:
        if self._loop is None:
            return  # server not fully started yet - nothing to broadcast to
        asyncio.run_coroutine_threadsafe(self._broadcast_async(payload), self._loop)

    async def _broadcast_async(self, payload: dict[str, Any]) -> None:
        dead = []
        for ws in self._sockets:
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001 - a broken socket shouldn't break the others
                dead.append(ws)
        for ws in dead:
            self._sockets.discard(ws)


manager = ConnectionManager()
control = ControlPanel(broadcast=manager.broadcast)

app.mount("/static", StaticFiles(directory=VIZ_DIR / "static"), name="static")


@app.on_event("startup")
async def _bind_loop() -> None:
    manager.bind_loop(asyncio.get_running_loop())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(VIZ_DIR / "static" / "index.html")


@app.get("/api/snapshots")
def api_snapshots() -> list[dict[str, str]]:
    return [{"filename": p.name} for p in list_snapshots()]


@app.get("/api/snapshot/{filename}")
def api_snapshot_detail(filename: str) -> dict[str, Any]:
    """Candles (lightweight-charts shape) + raw SMC detector output for the chart's
    overlay layer - used to confirm chart+overlay rendering against real snapshot
    data before/independent of running a full backtest."""
    path = PROJECT_ROOT / "data_snapshots" / filename
    if not path.exists():
        return {"error": f"Snapshot not found: {filename}"}

    df = load_snapshot(path)
    candles = [
        {"time": int(r.Index.timestamp()), "open": float(r.open), "high": float(r.high),
         "low": float(r.low), "close": float(r.close)}
        for r in df.itertuples()
    ]
    features = compute_smc_features(df, lookback_bars=min(len(df), DEFAULT_LOOKBACK_BARS))
    return {"candles": candles, "features": features}


class BacktestStartRequest(BaseModel):
    snapshot: str
    min_confluence: int = 2
    initial_balance: float = 20_000.0
    taker_fee_pct: float = 0.0040
    slippage_bps: float = 5.0


@app.post("/api/backtest/start")
def api_backtest_start(req: BacktestStartRequest) -> dict[str, str]:
    snapshot_path = PROJECT_ROOT / "data_snapshots" / req.snapshot
    if not snapshot_path.exists():
        return {"error": f"Snapshot not found: {req.snapshot}"}
    try:
        run_id = control.start_backtest(
            snapshot_path, req.min_confluence, req.initial_balance, req.taker_fee_pct, req.slippage_bps,
        )
        return {"run_id": run_id}
    except RuntimeError as e:
        return {"error": str(e)}


@app.get("/api/backtest/status")
def api_backtest_status() -> dict[str, Any]:
    b = control.backtest
    return {"status": b.status, "run_id": b.run_id, "progress": b.progress, "error": b.error}


@app.post("/api/paper/start")
def api_paper_start() -> dict[str, str]:
    try:
        control.start_paper()
        return {"status": "running"}
    except NotImplementedError as e:
        return {"error": str(e)}
    except RuntimeError as e:
        return {"error": str(e)}


@app.post("/api/paper/stop")
def api_paper_stop() -> dict[str, str]:
    control.stop_paper()
    return {"status": "stopped"}


@app.get("/api/paper/status")
def api_paper_status() -> dict[str, Any]:
    p = control.paper
    return {"status": p.status, "balance": p.balance, "open_position": p.open_position, "connection": p.connection}


@app.get("/api/live/readiness")
def api_live_readiness() -> dict[str, Any]:
    return control.live_readiness()


@app.post("/api/live/start")
def api_live_start() -> dict[str, str]:
    try:
        control.start_live()
        return {"status": "running"}
    except (PermissionError, NotImplementedError) as e:
        return {"error": str(e)}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "hello",
            "backtest_status": control.backtest.status,
            "paper_status": control.paper.status,
            "live_readiness": control.live_readiness(),
        })
        while True:
            await websocket.receive_text()  # dashboard doesn't send commands over WS today; just keeps the socket alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    print("SMC Bot Dashboard: http://localhost:8000")
    uvicorn.run("viz.server:app", host="0.0.0.0", port=8000, reload=False)
