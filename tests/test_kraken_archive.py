"""
test_kraken_archive.py
Tests data/kraken_archive.py's gap-handling against a small synthetic fixture
mimicking Kraken's OHLCVT archive format (tests/fixtures/kraken_xbtusd_5_with_gap.csv)
- built ahead of the real archive landing (see README's Status section for why the
real archive isn't available yet), per Kraken's own docs: "the OHLCVT data only
includes entries for intervals when trades happened, so any missing candlesticks
indicate that no trades occurred during those intervals"
(support.kraken.com/articles/360047124832).

Decision this test locks in: forward-fill missing intervals as flat, zero-volume
candles (rather than leaving gaps unfilled) and flag each with is_gap_fill=True.
See data/kraken_archive.py::_fill_gaps for the full reasoning.
"""
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from data.kraken_archive import build_snapshot, load_archive_pair

FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "kraken_xbtusd_5_with_gap.csv"


def _zip_csv(content: str, entry_name: str = "XBTUSD_5.csv") -> BytesIO:
    """Zips CSV content in-memory, mirroring the real archive's PAIR_INTERVAL.csv
    naming, so tests don't need a committed binary zip fixture."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(entry_name, content)
    buf.seek(0)
    return buf


class TestLoadArchivePairGapHandling(unittest.TestCase):
    def setUp(self):
        self.zip_buf = _zip_csv(FIXTURE_CSV.read_text())
        self.df = load_archive_pair(self.zip_buf, "BTC/USD", 5)

    def test_gap_is_forward_filled_as_a_flat_zero_volume_candle(self):
        # Fixture has 4 real rows spanning 1700000000..1700001200 at 5min spacing,
        # with 1700000900 deliberately omitted (no trades occurred that interval).
        self.assertEqual(len(self.df), 5)  # 4 real rows + 1 synthesized gap row

        gap_ts = pd.Timestamp(1700000900, unit="s", tz="utc")
        self.assertIn(gap_ts, self.df.index)

        gap_row = self.df.loc[gap_ts]
        self.assertTrue(gap_row["is_gap_fill"])
        self.assertEqual(gap_row["volume"], 0.0)
        self.assertEqual(gap_row["trades"], 0)
        # Flat candle at the prior real close (102.5) - not interpolated toward the
        # next real candle, and no invented high/low range.
        for col in ["open", "high", "low", "close"]:
            self.assertEqual(gap_row[col], 102.5)

    def test_real_rows_are_not_flagged_as_gap_fills(self):
        real_rows = self.df[~self.df["is_gap_fill"]]
        self.assertEqual(len(real_rows), 4)
        self.assertTrue((real_rows["trades"] > 0).all())

    def test_index_is_fully_regular_after_gap_fill(self):
        deltas_seconds = self.df.index.to_series().diff().dropna().dt.total_seconds().unique()
        self.assertEqual(list(deltas_seconds), [300.0])

    def test_no_gap_fill_needed_is_a_no_op(self):
        gapless_csv = "1700000000,100,101,99,100.5,10,5\n1700000300,100.5,102,100,101.5,12,6\n"
        df = load_archive_pair(_zip_csv(gapless_csv), "BTC/USD", 5)
        self.assertEqual(len(df), 2)
        self.assertFalse(df["is_gap_fill"].any())


class TestBuildSnapshotFillsGapAtTheMergeSeam(unittest.TestCase):
    """The boundary between a main archive's last row and a quarterly update's
    first row is exactly as real a gap as one inside a single file - build_snapshot
    must fill it too, not just whatever gaps existed within each source file."""

    def test_gap_between_main_and_quarterly_zip_is_filled(self):
        main_csv = "1700000000,100,101,99,100.5,10,5\n1700000300,100.5,102,100,101.5,12,6\n"
        # Quarterly update resumes two intervals later, leaving 1700000600 missing.
        quarterly_csv = "1700000900,103,104,102.5,103.5,9,3\n"

        df = build_snapshot(_zip_csv(main_csv), "BTC/USD", 5, quarterly_zip_paths=[_zip_csv(quarterly_csv)])

        self.assertEqual(len(df), 4)  # 1700000000, 300, 600(gap), 900
        gap_ts = pd.Timestamp(1700000600, unit="s", tz="utc")
        self.assertTrue(df.loc[gap_ts, "is_gap_fill"])
        self.assertEqual(df.loc[gap_ts, "close"], 101.5)  # flat at the prior real close


if __name__ == "__main__":
    unittest.main()
