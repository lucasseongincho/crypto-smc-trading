"""
trader.py
Real order placement against Kraken's private REST API. This is the only module in
the project allowed to move real money - backtest/runner.py's fill simulator (shared
by backtest and paper trading) never calls anything in here.

Gated behind kill_switch_ready(): every entrypoint that would place a real order
checks it and raises rather than executes. This is enforced in code, not just by
viz/control.py declining to show an enabled Start button in the UI - a UI-only gate
is trivially bypassed by calling this module directly (a script, a REPL, a future
API route someone adds without checking the dashboard state first).

kill_switch_ready() is hardcoded False. Flipping it to a real readiness check is
explicitly [BEFORE-LIVE] scope, not today's - see the kickoff notes. What it needs
before it can ever return True:
  - A drawdown/circuit-breaker kill switch (halt on API errors, abnormal drawdown,
    lost connectivity) - ported conceptually from the original bot's
    MAX_DRAWDOWN_PERCENT high-water-mark check in live_main.py, but not yet built here.
  - WebSocket reconnect/backoff for the live price feed this module's position
    "sentry" will depend on (see viz/kraken_ws.py).
  - The dashboard and trading engine no longer sharing one process, so a viz-layer
    bug can't take the live position-monitoring loop down with it.
"""
import base64
import hashlib
import hmac
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

import requests

from live.config import KrakenCredentials, load_kraken_credentials

KRAKEN_API_URL = "https://api.kraken.com"
_PAIR_ALIASES = {"BTC/USD": "XBTUSD"}


def kill_switch_ready() -> bool:
    """Single source of truth for whether live order placement is allowed to run at
    all. viz/control.py reads this to decide whether the dashboard's live Start
    control is enabled; execute_market_order() below re-checks it independently so
    the gate holds even if something calls this module directly."""
    return False


class KillSwitchNotReadyError(RuntimeError):
    pass


def _resolve_pair(pair: str) -> str:
    return _PAIR_ALIASES.get(pair, pair.replace("/", ""))


def _sign_request(urlpath: str, data: dict[str, Any], secret: str) -> str:
    """Kraken's private-endpoint signing scheme: HMAC-SHA512 of the URL path plus a
    SHA256 digest of (nonce + POST body), keyed by the base64-decoded API secret."""
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data["nonce"]) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    signature = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    return base64.b64encode(signature.digest()).decode()


def _private_request(urlpath: str, data: dict[str, Any], creds: KrakenCredentials) -> dict[str, Any]:
    data = {**data, "nonce": str(int(time.time() * 1000))}
    headers = {
        "API-Key": creds.api_key,
        "API-Sign": _sign_request(urlpath, data, creds.api_secret),
    }
    resp = requests.post(KRAKEN_API_URL + urlpath, data=data, headers=headers, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken private API error: {payload['error']}")
    return payload["result"]


@dataclass
class OrderResult:
    order_id: str
    raw: dict[str, Any]


def place_market_order(pair: str, side: str, volume: float, creds: KrakenCredentials | None = None) -> OrderResult:
    """Places a real market order on Kraken Spot. side is 'buy' or 'sell'. Kraken
    Spot has no margin/shorting for this project's scope, so 'sell' only ever means
    closing an existing long, never opening a naked short - callers are responsible
    for that invariant (see backtest/runner.py's same constraint on the simulated
    side)."""
    if not kill_switch_ready():
        raise KillSwitchNotReadyError(
            "Live order placement is disabled: kill_switch_ready() is False. "
            "See this module's docstring for what has to be built before it can flip."
        )

    creds = creds or load_kraken_credentials()
    result = _private_request(
        "/0/private/AddOrder",
        {"pair": _resolve_pair(pair), "type": side, "ordertype": "market", "volume": f"{volume:.8f}"},
        creds,
    )
    order_id = (result.get("txid") or [None])[0]
    return OrderResult(order_id=order_id, raw=result)


def get_account_balance(asset: str = "ZUSD", creds: KrakenCredentials | None = None) -> float:
    """Real USD balance from Kraken. Doesn't require kill_switch_ready - reading the
    balance isn't a trading action, and viz/control.py's Arm step needs it to show
    the risk summary ("balance at arm") before Start is even clickable."""
    creds = creds or load_kraken_credentials()
    result = _private_request("/0/private/Balance", {}, creds)
    return float(result.get(asset, 0.0))
