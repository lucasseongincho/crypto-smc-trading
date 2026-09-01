"""
kraken_ws.py
Kraken public WebSocket API v2 client for live market data - no key required. Uses
the `websockets` library directly (CCXT Pro is excluded from this project: it's paid
and this is free).

stream_ohlc() is resilient: it reconnects with exponential backoff (capped, reset to
the initial delay after any successful connection) until explicitly stopped via
stop_event, and reports connection state through on_state() so the dashboard's
connection-health banner reflects reality instead of a hardcoded prop. This was
originally scoped as before-live work, but pulled forward because paper trading
(viz/control.py) depends on this connection actually staying up over hours, not just
on the first connect succeeding.

Note the symbol spelling mismatch with data/fetcher.py: Kraken's REST API (v0) wants
its own asset codes ("XBTUSD"), but WS API v2 takes the plain "BASE/QUOTE" spelling
("BTC/USD") directly - confirmed by connecting live. Don't reuse fetcher.py's
_resolve_pair() here; it would produce the wrong symbol for the WS channel.
"""
import asyncio
import json
from typing import Any, Awaitable, Callable

import websockets

KRAKEN_WS_URL = "wss://ws.kraken.com/v2"

OhlcCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

# "healthy": connected and confirmed receiving data.
# "reconnecting": disconnected, actively backing off and retrying.
# "dropped": stopped for good (stop_event was set) - not a "gave up" state, since
# this function otherwise retries forever.
ConnectionState = str
ConnectionStateCallback = Callable[[ConnectionState, dict[str, Any]], None]

DEFAULT_INITIAL_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_BACKOFF_SECONDS = 30.0


def _notify(on_state: ConnectionStateCallback | None, state: ConnectionState, info: dict[str, Any]) -> None:
    if on_state is not None:
        on_state(state, info)


async def _run_one_connection(
    symbol: str,
    interval: int,
    on_candle: OhlcCallback,
    on_state: ConnectionStateCallback | None,
    stop_event: asyncio.Event,
    connect_fn: Callable[..., Any],
) -> None:
    """One connection attempt: connect, subscribe, stream until the socket closes,
    an error is raised, or stop_event is set. Only reports "healthy" once real data
    has actually arrived - a bare TCP/TLS handshake succeeding doesn't mean Kraken is
    going to send anything."""
    async with connect_fn(KRAKEN_WS_URL) as ws:
        await ws.send(json.dumps({
            "method": "subscribe",
            "params": {"channel": "ohlc", "symbol": [symbol], "interval": interval},
        }))

        while not stop_event.is_set():
            raw = await ws.recv()
            msg = json.loads(raw)

            if msg.get("channel") != "ohlc" or "data" not in msg:
                continue  # status/subscribe-ack/heartbeat messages - not candle data

            _notify(on_state, "healthy", {})
            for candle in msg["data"]:
                result = on_candle(candle)
                if asyncio.iscoroutine(result):
                    await result


async def stream_ohlc(
    symbol: str,
    interval: int,
    on_candle: OhlcCallback,
    on_state: ConnectionStateCallback | None = None,
    stop_event: asyncio.Event | None = None,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_backoff: float = DEFAULT_MAX_BACKOFF_SECONDS,
    connect_fn: Callable[..., Any] = websockets.connect,
) -> None:
    """Subscribes to OHLC candles for `symbol` (Kraken's WS "BASE/QUOTE" spelling,
    e.g. "BTC/USD") at `interval` minutes and calls on_candle(candle_dict) for every
    candle in every message, reconnecting on any disconnect until stop_event is set.

    Each candle_dict has Kraken's native fields: symbol, open, high, low, close,
    volume, vwap, trades, interval, interval_begin, timestamp. Kraken does not mark
    candles as closed/unclosed in this payload - it keeps sending "update" messages
    for the same interval_begin as a bar forms, then moves on to a new
    interval_begin once that bar closes. Detecting the close is the caller's job
    (see viz/control.py's PaperEngine for how paper trading does it).

    connect_fn defaults to websockets.connect and exists so tests can inject a fake
    connection factory instead of hitting the real network - see tests/test_kraken_ws.py.
    """
    stop_event = stop_event or asyncio.Event()
    backoff = initial_backoff

    while not stop_event.is_set():
        try:
            await _run_one_connection(symbol, interval, on_candle, on_state, stop_event, connect_fn)
            backoff = initial_backoff  # a clean run means the connection worked - reset for next time
        except Exception as e:  # noqa: BLE001 - any disconnect/network failure should trigger a reconnect
            if stop_event.is_set():
                break
            _notify(on_state, "reconnecting", {"error": str(e), "retry_in_seconds": backoff})
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass  # backoff elapsed normally - loop around and retry
            backoff = min(backoff * 2, max_backoff)

    _notify(on_state, "dropped", {"reason": "stopped"})


if __name__ == "__main__":
    async def _print_candle(candle: dict[str, Any]) -> None:
        print(candle)

    def _print_state(state: str, info: dict[str, Any]) -> None:
        print(f"[connection] {state} {info}")

    asyncio.run(stream_ohlc("BTC/USD", 5, _print_candle, on_state=_print_state))
