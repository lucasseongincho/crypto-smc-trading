"""
test_kraken_ws.py
Tests viz/kraken_ws.py's reconnect/backoff state machine against a fake WebSocket
connection factory - no real network needed. This is exactly the "reliable once it's
streaming continuously" behavior paper trading depends on, so it's covered before
paper trading gets wired to it.
"""
import asyncio
import json
import unittest

from viz.kraken_ws import stream_ohlc


class _FakeConnection:
    """Stands in for a websockets.connect(...) async context manager. recv() raises
    once its scripted messages run out, simulating the connection dropping."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def send(self, data):
        pass

    async def recv(self):
        if not self._messages:
            raise ConnectionError("simulated connection drop")
        return self._messages.pop(0)


def _ohlc_message(interval_begin: str, close: float) -> str:
    return json.dumps({
        "channel": "ohlc",
        "type": "update",
        "data": [{
            "symbol": "BTC/USD", "open": close, "high": close, "low": close, "close": close,
            "interval_begin": interval_begin, "interval": 5,
        }],
    })


def _non_ohlc_message() -> str:
    return json.dumps({"channel": "status", "type": "update", "data": [{"system": "online"}]})


class TestStreamOhlcReconnect(unittest.TestCase):
    def test_reconnects_after_a_drop_and_keeps_delivering_candles(self):
        scripts = [
            [],  # connection 1: dies immediately, no data at all
            [_non_ohlc_message(), _ohlc_message("2026-01-01T00:00:00Z", 100.0)],  # connection 2
            [_ohlc_message("2026-01-01T00:05:00Z", 101.0)],  # connection 3
        ]
        connect_calls = []

        def connect_fn(url):
            connect_calls.append(url)
            return _FakeConnection(scripts.pop(0))

        received: list[float] = []
        states: list[str] = []
        stop_event = asyncio.Event()

        def on_candle(candle):
            received.append(candle["close"])
            if len(received) == 2:
                stop_event.set()

        def on_state(state, info):
            states.append(state)

        asyncio.run(stream_ohlc(
            "BTC/USD", 5, on_candle, on_state=on_state, stop_event=stop_event,
            initial_backoff=0.001, max_backoff=0.002, connect_fn=connect_fn,
        ))

        self.assertEqual(received, [100.0, 101.0])
        self.assertEqual(len(connect_calls), 3)  # proves it actually reconnected twice
        self.assertEqual(states.count("reconnecting"), 2)
        self.assertIn("healthy", states)
        self.assertEqual(states[-1], "dropped")  # final state is always "dropped" once stopped

    def test_non_ohlc_messages_do_not_trigger_healthy(self):
        """A status/heartbeat message shouldn't be mistaken for confirmed data flow."""
        scripts = [[_non_ohlc_message()]]

        def connect_fn(url):
            return _FakeConnection(scripts.pop(0))

        states: list[str] = []
        stop_event = asyncio.Event()

        def on_candle(candle):
            pass  # never called in this test

        def on_state(state, info):
            states.append(state)
            if state == "reconnecting":  # the connection drops right after the status message
                stop_event.set()

        asyncio.run(stream_ohlc(
            "BTC/USD", 5, on_candle, on_state=on_state, stop_event=stop_event,
            initial_backoff=0.001, max_backoff=0.002, connect_fn=connect_fn,
        ))

        self.assertNotIn("healthy", states)

    def test_backoff_doubles_then_caps_at_max(self):
        # Every connection dies immediately - forces four straight reconnect attempts.
        scripts = [[], [], [], []]

        def connect_fn(url):
            return _FakeConnection(scripts.pop(0) if scripts else [])

        retry_delays: list[float] = []
        stop_event = asyncio.Event()

        def on_candle(candle):
            pass

        def on_state(state, info):
            if state == "reconnecting":
                retry_delays.append(info["retry_in_seconds"])
                if len(retry_delays) == 4:
                    stop_event.set()

        asyncio.run(stream_ohlc(
            "BTC/USD", 5, on_candle, on_state=on_state, stop_event=stop_event,
            initial_backoff=0.001, max_backoff=0.003, connect_fn=connect_fn,
        ))

        # 0.001 -> 0.002 -> 0.003 (cap) -> 0.003 (stays capped)
        self.assertEqual(retry_delays, [0.001, 0.002, 0.003, 0.003])


if __name__ == "__main__":
    unittest.main()
