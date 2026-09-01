"""
snapshot.py
Save/load pinned OHLCV CSV snapshots under data_snapshots/, so backtests are
reproducible against a fixed slice of history instead of whatever Kraken's API
returns at run time.
"""
from pathlib import Path

import pandas as pd

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data_snapshots"


def _snapshot_filename(pair: str, interval: int, start: str, end: str) -> str:
    safe_pair = pair.replace("/", "-")
    return f"{safe_pair}_{interval}m_{start}_{end}.csv"


def save_snapshot(df: pd.DataFrame, pair: str, interval: int, start: str, end: str) -> Path:
    """Write df (indexed by UTC timestamp, from fetch_ohlc_history) to a pinned CSV.
    `start`/`end` are the requested range (YYYY-MM-DD) and become part of the filename
    so multiple ranges for the same pair/interval can coexist."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / _snapshot_filename(pair, interval, start, end)
    df.to_csv(path, index_label="time")
    return path


def load_snapshot(path: Path | str) -> pd.DataFrame:
    """Read a snapshot CSV back into the same shape save_snapshot() wrote:
    UTC DatetimeIndex named 'time', float OHLCV columns."""
    df = pd.read_csv(path, index_col="time", parse_dates=["time"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("utc")
    return df


def list_snapshots() -> list[Path]:
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(SNAPSHOT_DIR.glob("*.csv"))
