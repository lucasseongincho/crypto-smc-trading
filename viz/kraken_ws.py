"""
kraken_ws.py
Kraken public WebSocket API v2 client for live market data - no key required. Uses
the `websockets` library directly (CCXT Pro is excluded from this project: it's paid
and this is free).

This is scaffolding for the next session's live-streaming work (see the kickoff
task order - today's dashboard renders against snapshot data only, live WS
integration into the dashboard is explicitly deferred). It's a standalone,
independently testable client: connect, subscribe to OHLC candles for a pair, and
call back on every snapshot/update message. No reconnect/backoff logic yet - that's
[BEFORE-LIVE] scope (see live/trader.py's docstring for the full list of what has to
land before unattended live streaming is safe).
"""
import asyncio
import json
from typing import Any, Callable, Awaitable

import websockets

KRAKEN_WS_URL = "wss://ws.kraken.com/v2"

OhlcCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


async def stream_ohlc(
    symbol: str,
    interval: int,
    on_candle: OhlcCallback,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Connects to Kraken's public WS v2, subscribes to OHLC candles for `symbol`
    (Kraken's own "BASE/QUOTE" spelling, e.g. "BTC/USD") at `interval` minutes, and
    calls on_candle(candle_dict) for every candle in every snapshot/update message
    until stop_event is set (or the connection drops - no auto-reconnect).

    Each candle_dict has Kraken's native fields: symbol, open, high, low, close,
    volume, vwap, trades, interval, interval_begin, timestamp.
    """
    async with websockets.connect(KRAKEN_WS_URL) as ws:
        await ws.send(json.dumps({
            "method": "subscribe",
            "params": {"channel": "ohlc", "symbol": [symbol], "interval": interval},
        }))

        while stop_event is None or not stop_event.is_set():
            raw = await ws.recv()
            msg = json.loads(raw)

            if msg.get("channel") != "ohlc" or "data" not in msg:
                continue  # status/subscribe-ack/heartbeat messages - not candle data

            for candle in msg["data"]:
                result = on_candle(candle)
                if asyncio.iscoroutine(result):
                    await result


if __name__ == "__main__":
    async def _print_candle(candle: dict[str, Any]) -> None:
        print(candle)

    asyncio.run(stream_ohlc("BTC/USD", 5, _print_candle))
