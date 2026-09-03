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

Gap handling: per Kraken's own docs, "the OHLCVT data only includes entries for
intervals when trades happened, so any missing candlesticks indicate that no trades
occurred during those intervals" - the archive omits a row entirely rather than
emitting a zero-volume one. See _fill_gaps() for why that's forward-filled here
rather than left as-is, and tests/test_kraken_archive.py for the fixture that locks
the decision in.
"""
import csv
import io
import zipfile
from pathlib import Path
from typing import IO

import pandas as pd

from data.fetcher import _resolve_pair

_COLUMNS = ["time", "open", "high", "low", "close", "volume", "trades"]

ZipSource = Path | str | IO[bytes]


def _find_pair_entry(zf: zipfile.ZipFile, pair: str, interval: int) -> str:
    """Locate the CSV for `pair`/`interval` inside the archive, regardless of whether
    it's zipped flat or under a subfolder.

    Matches on the entry's own basename being *exactly* PAIR_INTERVAL.csv, not on
    endswith() - a real archive contains entries like AIXBTUSD_5.csv (a genuinely
    different pair) alongside XBTUSD_5.csv, and "AIXBTUSD_5.csv".endswith("XBTUSD_5.csv")
    is true. An endswith() match silently returned AIXBTUSD_5.csv instead of
    XBTUSD_5.csv from one real archive zip (whichever entry happened to come first in
    zf.namelist() order) before this was caught - see
    tests/test_kraken_archive.py::test_a_longer_pair_code_ending_in_the_target_is_not_matched.
    """
    target = f"{_resolve_pair(pair)}_{interval}.csv".lower()
    for name in zf.namelist():
        basename = name.lower().rsplit("/", 1)[-1]
        if basename == target:
            return name
    raise FileNotFoundError(
        f"No entry named '{target}' inside {zf.filename!r}. "
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


def _fill_gaps(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Reindexes df onto a fully regular interval_minutes grid spanning its own
    first-to-last timestamp, forward-filling any missing interval as a flat candle
    (open=high=low=close=prior close, volume=0, trades=0) and flagging it with
    is_gap_fill=True.

    Why forward-fill instead of leaving gaps as-is: data/smc.py's detectors
    (order blocks, FVGs, swing fractals, BOS/CHoCH) all reason about "the
    next/previous candle" positionally (window.iloc[i-1], window.iloc[i+1], ...),
    not by elapsed time. An unfilled gap would make two candles that are actually
    10+ minutes apart look like consecutive 5-minute bars to every one of those
    detectors - e.g. a real order block could get manufactured from price action
    that never actually engulfed the "next" candle, because the true next candle
    was silently missing. A flat, zero-volume candle is inert to those detectors
    (no range to engulf, no gap to form a FVG, no new high/low to register as a
    swing), so filling this way keeps the grid regular without inventing price
    action that didn't happen.

    Why zero volume/trades rather than interpolating them: matches what Kraken
    itself means by omitting the row - "no trades occurred during that interval"
    (support.kraken.com/articles/360047124832) - this is the direct continuation of
    that semantics, not an invented convention.

    Why an explicit is_gap_fill column when trades==0 already implies it: Kraken's
    real data never emits a zero-trade row (it omits the interval instead), so
    trades==0 is already an implicit gap-fill marker in practice - but making it
    explicit means that invariant doesn't have to be re-derived (or silently
    assumed) by every downstream reader.

    Idempotency (why this matters): build_snapshot() calls this twice - once per
    file inside load_archive_pair(), then again on the merged result. A row a
    per-file pass already synthesized is, by the second pass, no longer missing
    (it has real numeric values from the first fill) - naively recomputing
    is_gap_fill from scratch on the second pass would read that as "not a gap" and
    silently overwrite the correct True with a fresh False, discarding the first
    pass's finding. Real bug, caught once real archive data actually exercised the
    per-file-then-merged path (a single quarterly file with an internal gap,
    concatenated with a main archive that already covered - and had separately
    gap-filled - that same slot). Fixed by OR-ing any is_gap_fill column already
    present in df into the freshly-computed one, so a row flagged True by any
    earlier pass stays True regardless of what a later pass sees.
    """
    if df.empty:
        return df.assign(is_gap_fill=pd.Series(dtype=bool))

    full_index = pd.date_range(df.index[0], df.index[-1], freq=f"{interval_minutes}min", tz=df.index.tz)
    full_index.name = "time"
    reindexed = df.reindex(full_index)

    newly_missing = reindexed["close"].isna()
    if "is_gap_fill" in df.columns:
        # .eq(True), not .fillna(False).astype(bool): a newly-introduced row (not
        # in df before this reindex) has NaN here, and NaN.eq(True) is already
        # False - no fillna/downcast needed.
        already_flagged = reindexed["is_gap_fill"].eq(True)
    else:
        already_flagged = pd.Series(False, index=reindexed.index)
    is_gap_fill = newly_missing | already_flagged

    reindexed["close"] = reindexed["close"].ffill()
    for col in ["open", "high", "low"]:
        reindexed[col] = reindexed[col].fillna(reindexed["close"])
    reindexed["volume"] = reindexed["volume"].fillna(0.0)
    reindexed["trades"] = reindexed["trades"].fillna(0).astype(int)
    reindexed["is_gap_fill"] = is_gap_fill

    return reindexed


def load_archive_pair(zip_path: ZipSource, pair: str, interval: int) -> pd.DataFrame:
    """Extract and parse just the one PAIR_INTERVAL.csv entry needed from a full
    Kraken_OHLCVT.zip (or a quarterly-update zip with the same layout), without
    unzipping the rest of the archive. Gap-fills the result - see _fill_gaps()."""
    with zipfile.ZipFile(zip_path) as zf:
        entry = _find_pair_entry(zf, pair, interval)
        raw = zf.read(entry)
    return _fill_gaps(_parse_ohlcvt_csv(raw), interval)


def build_snapshot(
    main_zip_path: ZipSource,
    pair: str,
    interval: int,
    quarterly_zip_paths: list[ZipSource] | None = None,
) -> pd.DataFrame:
    """Merge the main archive with any newer quarterly-update zips into one
    deduped, sorted, gap-filled DataFrame - the shape data/snapshot.save_snapshot()
    expects. Quarterly data wins on overlapping timestamps (it's the more recent
    pull). Re-runs gap-filling after the merge, not just within each source file -
    the boundary between the main archive's last row and a quarterly file's first
    row is exactly as real a gap as one inside either file alone."""
    df = load_archive_pair(main_zip_path, pair, interval)

    for qz in quarterly_zip_paths or []:
        q_df = load_archive_pair(qz, pair, interval)
        df = pd.concat([df, q_df])
        df = df[~df.index.duplicated(keep="last")]

    df = df.sort_index()
    return _fill_gaps(df, interval)


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
    gap_count = int(df["is_gap_fill"].sum())
    print(f"Parsed {len(df)} candles ({gap_count} gap-filled, no trades): {df.index[0]} -> {df.index[-1]}")

    start = df.index[0].strftime("%Y-%m-%d")
    end = df.index[-1].strftime("%Y-%m-%d")
    path = save_snapshot(df, args.pair, args.interval, start, end)
    print(f"Saved snapshot: {path}")
