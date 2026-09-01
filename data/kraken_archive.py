"""
kraken_archive.py
Import from Kraken's official bulk OHLCVT CSV archive (support.kraken.com/articles/
360047124832) - a single ZIP covering every pair/interval since inception, distributed
via Google Drive, plus quarterly-update ZIPs for anything traded since the last full
archive snapshot. This is the real "bulk download" equivalent to Binance's
data.binance.vision that Kraken's live REST/WS APIs don't provide on their own
(the public /OHLC endpoint only retains a rolling ~720 most-recent candles per
interval - not enough for backtesting depth. See data/fetcher.py).

The archive itself isn't fetched programmatically here - the shared Drive link has no
HTTP Range support (so partial-extraction-without-downloading isn't possible) and can
hit Google's per-file download quota. Point this module at a zip you've downloaded
yourself.
"""
import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

from data.fetcher import _resolve_pair

_COLUMNS = ["time", "open", "high", "low", "close", "volume", "trades"]


def _find_pair_entry(zf: zipfile.ZipFile, pair: str, interval: int) -> str:
    """Locate the CSV for `pair`/`interval` inside the archive, regardless of whether
    it's zipped flat or under a subfolder."""
    target = f"{_resolve_pair(pair)}_{interval}.csv".lower()
    for name in zf.namelist():
        if name.lower().endswith(target):
            return name
    raise FileNotFoundError(
        f"No entry ending in '{target}' inside {zf.filename!r}. "
        f"Available pairs/intervals are named PAIR_INTERVAL.csv (e.g. XBTUSD_5.csv)."
    )


def _parse_ohlcvt_csv(raw: bytes) -> pd.DataFrame:
    """Kraken's archive CSVs are headerless: timestamp,open,high,low,close,volume,trades.
    Parsed defensively in case a given file does carry a header row."""
    text = raw.decode("utf-8")
    first_line = text.split("\n", 1)[0]
    has_header = not first_line.split(",")[0].strip().isdigit()

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if has_header:
        rows = rows[1:]

    df = pd.DataFrame(rows, columns=_COLUMNS)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(float).astype(int)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    return df.set_index("time").sort_index()


def load_archive_pair(zip_path: Path | str, pair: str, interval: int) -> pd.DataFrame:
    """Extract and parse just the one PAIR_INTERVAL.csv entry needed from a full
    Kraken_OHLCVT.zip (or a quarterly-update zip with the same layout), without
    unzipping the rest of the archive."""
    with zipfile.ZipFile(zip_path) as zf:
        entry = _find_pair_entry(zf, pair, interval)
        raw = zf.read(entry)
    return _parse_ohlcvt_csv(raw)


def build_snapshot(
    main_zip_path: Path | str,
    pair: str,
    interval: int,
    quarterly_zip_paths: list[Path | str] | None = None,
) -> pd.DataFrame:
    """Merge the main archive with any newer quarterly-update zips into one
    deduped, sorted DataFrame - the shape data/snapshot.save_snapshot() expects.
    Quarterly data wins on overlapping timestamps (it's the more recent pull)."""
    df = load_archive_pair(main_zip_path, pair, interval)

    for qz in quarterly_zip_paths or []:
        q_df = load_archive_pair(qz, pair, interval)
        df = pd.concat([df, q_df])
        df = df[~df.index.duplicated(keep="last")]

    return df.sort_index()


if __name__ == "__main__":
    import argparse

    from data.snapshot import save_snapshot

    parser = argparse.ArgumentParser(description="Import a Kraken OHLCVT archive zip into a pinned snapshot")
    parser.add_argument("--zip", required=True, help="Path to the downloaded Kraken_OHLCVT.zip")
    parser.add_argument("--quarterly", nargs="*", default=[], help="Optional newer quarterly-update zip(s)")
    parser.add_argument("--pair", default="BTC/USD")
    parser.add_argument("--interval", type=int, default=5, help="minutes")
    args = parser.parse_args()

    print(f"Extracting {args.pair} {args.interval}m from {args.zip} ...")
    df = build_snapshot(args.zip, args.pair, args.interval, args.quarterly)
    print(f"Parsed {len(df)} candles: {df.index[0]} -> {df.index[-1]}")

    start = df.index[0].strftime("%Y-%m-%d")
    end = df.index[-1].strftime("%Y-%m-%d")
    path = save_snapshot(df, args.pair, args.interval, start, end)
    print(f"Saved snapshot: {path}")
