"""
fetcher.py
Kraken public REST client for historical OHLC candles. No API key required.

Kraken's /public/OHLC endpoint returns at most ~720 candles per call and takes a
`since` cursor (a per-pair "last" value from the previous response, not a plain
unix timestamp) to page forward. There is no bulk-download equivalent to
Binance's data.binance.vision, so a full history pull is many sequential
requests - this module handles that pagination and the public rate limit.
"""
import time
from typing import Any

import pandas as pd
import requests

KRAKEN_API_URL = "https://api.kraken.com/0/public/OHLC"

# Kraken's own pair codes don't match the common BASE/QUOTE spelling (BTC -> XBT).
_PAIR_ALIASES = {
    "BTC/USD": "XBTUSD",
}

# Kraken candle interval in minutes -> the values it actually accepts.
VALID_INTERVALS = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600}

_REQUEST_DELAY_SECONDS = 1.0  # stay well under Kraken's public rate limit


def _resolve_pair(pair: str) -> str:
    return _PAIR_ALIASES.get(pair, pair.replace("/", ""))


def fetch_ohlc_page(pair: str, interval: int, since: int | None = None) -> tuple[list[list[Any]], int]:
    """One page of Kraken OHLC candles. Returns (rows, next_since_cursor).

    Each row is Kraken's raw format: [time, open, high, low, close, vwap, volume, count].
    `next_since_cursor` is the opaque `last` value Kraken returns - pass it back in as
    `since` to fetch the next page. It can equal the input `since` when there is no more
    data yet (the caller should stop, not loop forever).
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Kraken doesn't support a {interval}-minute interval. Valid: {sorted(VALID_INTERVALS)}")

    params: dict[str, Any] = {"pair": _resolve_pair(pair), "interval": interval}
    if since is not None:
        params["since"] = since

    resp = requests.get(KRAKEN_API_URL, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("error"):
        raise RuntimeError(f"Kraken API error for {pair}: {payload['error']}")

    result = payload["result"]
    # result has one key for the resolved pair (Kraken's own spelling) plus "last".
    pair_key = next(k for k in result if k != "last")
    return result[pair_key], result["last"]


def fetch_ohlc_history(
    pair: str,
    interval: int = 5,
    since: int | None = None,
    until: int | None = None,
    max_pages: int = 10_000,
) -> pd.DataFrame:
    """Paginate fetch_ohlc_page() from `since` (unix seconds, or Kraken's earliest
    available history if None) up to `until` (unix seconds, or "now" if None).

    Returns a DataFrame indexed by UTC timestamp with columns:
    open, high, low, close, vwap, volume, trades - sorted oldest to newest, deduped.
    """
    until = until if until is not None else int(time.time())
    cursor = since
    all_rows: list[list[Any]] = []
    seen_times: set[int] = set()

    for _ in range(max_pages):
        rows, next_cursor = fetch_ohlc_page(pair, interval, since=cursor)

        new_rows = [r for r in rows if int(r[0]) not in seen_times]
        for r in new_rows:
            seen_times.add(int(r[0]))
        all_rows.extend(new_rows)

        if not rows or next_cursor == cursor:
            break  # Kraken has nothing further to page through
        cursor = next_cursor

        newest_ts = int(rows[-1][0])
        if newest_ts >= until:
            break

        time.sleep(_REQUEST_DELAY_SECONDS)

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "vwap", "volume", "trades"])

    df = pd.DataFrame(
        all_rows,
        columns=["time", "open", "high", "low", "close", "vwap", "volume", "trades"],
    )
    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if since is not None:
        df = df[df.index >= pd.Timestamp(since, unit="s", tz="utc")]
    df = df[df.index <= pd.Timestamp(until, unit="s", tz="utc")]
    return df


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    from data.snapshot import save_snapshot

    parser = argparse.ArgumentParser(description="Download a pinned Kraken OHLC snapshot")
    parser.add_argument("--pair", default="BTC/USD")
    parser.add_argument("--interval", type=int, default=5, help="minutes")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    args = parser.parse_args()

    since_ts = int(datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    until_ts = int(datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    print(f"Downloading {args.pair} {args.interval}m candles: {args.start} -> {args.end} ...")
    df = fetch_ohlc_history(args.pair, interval=args.interval, since=since_ts, until=until_ts)
    print(f"Fetched {len(df)} candles.")

    path = save_snapshot(df, args.pair, args.interval, args.start, args.end)
    print(f"Saved snapshot: {path}")
