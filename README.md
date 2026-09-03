# crypto-smc-bot

Smart Money Concepts (SMC) trading bot for BTC/USD on Kraken Spot, 5-minute bars, with
a FastAPI + WebSocket dashboard for backtest / paper / live control.

Ported from [tradingagents-kr](https://github.com/lucasseongincho/tradingagents-kr)
(SMC detectors, signal aggregator, journal schema) and the original
[Crypto-Trading-Bot](https://github.com/lucasseongincho/Crypto-Trading-Bot)
(absolute-dollar risk sizing, 1.5R take-profit, Telegram notifications), rebuilt
around Kraken instead of Binance/Coinbase and with the dashboard as a first-class
piece rather than an afterthought. See `design-reference/` for the dashboard's
visual/interaction spec.

## Status (as of this session)

- **Historical data**: real full history is loaded. Kraken's public `/OHLC` REST
  endpoint only retains a rolling ~720 candles (`data/fetcher.py`) - not enough for
  backtesting depth - so `data_snapshots/BTC-USD_5m_2013-10-06_2026-03-31.csv`
  (**not committed** - large; regenerate via the command below) was built from the
  official bulk OHLCVT archive plus all 13 quarterly updates through Q1 2026,
  merged via `data/kraken_archive.py`. Actual coverage: **2013-10-06 21:30 UTC
  through 2026-03-31 23:55 UTC**, 1,313,022 rows, a fully regular 5-minute grid
  (verified - no unexpected holes at any of the 13 main/quarterly file boundaries).
  257,154 rows (19.6%) are `is_gap_fill=True` - concentrated almost entirely in
  2013-2016 (up to 92.7% of 2014, Kraken's thin-liquidity early years), dropping
  below 0.3% from 2019 onward.

  Importing this for real caught two bugs the synthetic-fixture tests hadn't:
  `_find_pair_entry`'s `endswith()` match was silently matching `AIXBTUSD_5.csv`
  (a different, unrelated pair) instead of `XBTUSD_5.csv` in one archive zip -
  fixed to match the entry's exact basename. And `_fill_gaps()` being called twice
  in the real pipeline (once per source file, again on the merged result) was
  silently overwriting a real `is_gap_fill=True` from the first pass back to
  `False` on the second, because by then the row had real values and no longer
  looked missing - fixed to be idempotent (OR-ing in any pre-existing flag). Both
  fixes are covered by regression tests in `tests/test_kraken_archive.py`.
- **SMC detectors + regression test**: ported (`data/smc.py`, `tests/test_smc.py`),
  including the BOS/CHoCH wick-vs-close bug fix from tradingagents-kr.
- **Signal aggregator**: ported (`signals/smc_aggregator.py`). `min_confluence` has
  **no validated default** for 5m BTC/USD - it's a required constructor argument on
  purpose (raises if omitted). tradingagents-kr's value (3) was tuned on daily stock
  bars and does not transfer. **Stays unvalidated on purpose** until real backtest
  history is loaded - tuning against the current ~720-candle window would fit noise,
  not validate anything. Same goes for `min_sl_distance` and `rr_ratio` in
  `backtest/risk.py`.
- **Backtest/paper fill simulator**: `backtest/runner.py`'s `FillEngine` is fee- and
  slippage-aware and shared by both `run_backtest()` (static snapshot data) and paper
  trading, which now drives it candle-by-candle from `viz/kraken_ws.py`'s live feed.
- **Kraken WebSocket client**: `viz/kraken_ws.py`'s `stream_ohlc()` reconnects with
  exponential backoff (capped, reset after a successful connection) until explicitly
  stopped, and reports connection state (`healthy`/`reconnecting`/`dropped`) so the
  dashboard's connection indicator reflects reality. Covered by
  `tests/test_kraken_ws.py` against a fake connection factory (no network needed).
- **Paper trading**: real, end-to-end. `viz/control.py`'s `ControlPanel` streams
  Kraken's live public WS feed, detects each closed candle (Kraken has no explicit
  "closed" flag - see `_on_paper_candle`'s docstring), and feeds it through the same
  `FillEngine` backtest uses. A post-reconnect snapshot replaying an already-closed
  bar is recognized and skipped rather than double-fed - see
  `tests/test_control_paper.py`. Verified live against Kraken's real feed: connected,
  received real candles, and opened a real simulated position from an actual signal.
- **Dashboard**: `viz/server.py` (FastAPI+WS) + `viz/control.py` +
  `viz/static/` render candles and SMC overlays, run full backtests with live
  progress/results/trade-log, and now run/stop real paper trading sessions with a
  live connection-health indicator. Live trading is hard-gated behind
  `live/trader.kill_switch_ready()`, currently `False`.
- **Live trading**: `live/trader.py` can place real Kraken orders
  (`place_market_order`), but every entrypoint refuses to run until
  `kill_switch_ready()` returns `True` - see that function's docstring for what has
  to land first (drawdown kill-switch, process separation from the dashboard - WS
  reconnect/backoff is now done, see above).

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # fill in Kraken/Telegram credentials only if/when needed
```

Paper trading and backtesting need no credentials (Kraken's public endpoints only).
`.env` is only required once live trading is enabled.

## Running things

```
# Pull a snapshot (thin - see Status above for why real history needs the archive import)
python -m data.fetcher --pair "BTC/USD" --interval 5 --start 2026-06-01 --end 2026-08-30

# Import Kraken's official bulk archive once you have it locally. --quarterly takes
# any number of quarterly-update zips in one call, in chronological order (each one
# extends/corrects the base data - see data/kraken_archive.py's build_snapshot()).
# If the main archive arrives pre-extracted (a folder of PAIR_INTERVAL.csv files,
# not a single zip - this happened once already, see Status above), wrap just the
# one file you need into a minimal single-entry zip first:
#   python -c "import zipfile; zipfile.ZipFile('main.zip','w').write('path/to/extracted/XBTUSD_5.csv', arcname='XBTUSD_5.csv')"
python -m data.kraken_archive --zip path/to/Kraken_OHLCVT.zip --pair "BTC/USD" --interval 5 \
  --quarterly path/to/Q1_2023.zip path/to/Q2_2023.zip  # ... in chronological order

# Run tests
python -m unittest discover tests

# Dashboard
python -m viz.server
# open http://localhost:8000
```

## Known limitations / follow-up checklist

See the SMC-on-spot, bars-vs-stints, and detector-correlation caveats in this
session's kickoff notes for how to *read* any backtest result here - none of that
is a build task, just an interpretation habit.

Before this runs unattended with live streaming or real money:
- [x] WebSocket reconnect/backoff for the Kraken feed (`viz/kraken_ws.py`)
- [x] Wire paper trading (`viz/control.py::start_paper`) to the live Kraken feed
- [ ] Dashboard and trading engine split into separate processes
- [ ] Kill switch / circuit breaker (drawdown halt, API-error halt, connectivity-loss halt)
- [x] Import the real Kraken OHLCVT archive (2013-10-06 through 2026-03-31 now in
      `data_snapshots/` - not committed, regenerate via the command above)
- [ ] Re-validate `min_confluence`, `min_sl_distance`, and `rr_ratio` against real
      backtest data now that real history is imported (all currently unvalidated defaults
      or explicit TODOs - see their respective docstrings)
