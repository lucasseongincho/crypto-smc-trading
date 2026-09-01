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

- **Historical data**: Kraken's public `/OHLC` REST endpoint only retains a rolling
  ~720 candles (`data/fetcher.py`) - not enough for real backtesting depth. Real
  history comes from Kraken's official bulk OHLCVT archive
  (`data/kraken_archive.py`) - see that module's docstring for how to obtain and
  import it. `data_snapshots/` currently holds only a small live-pulled sample
  (`BTC-USD_5m_2026-06-01_2026-08-30.csv`, ~106 candles) - a placeholder, not a
  meaningful backtest dataset.
- **SMC detectors + regression test**: ported (`data/smc.py`, `tests/test_smc.py`),
  including the BOS/CHoCH wick-vs-close bug fix from tradingagents-kr.
- **Signal aggregator**: ported (`signals/smc_aggregator.py`). `min_confluence` has
  **no validated default** for 5m BTC/USD - it's a required constructor argument on
  purpose (raises if omitted). tradingagents-kr's value (3) was tuned on daily stock
  bars and does not transfer.
- **Backtest/paper fill simulator**: `backtest/runner.py`'s `FillEngine` is fee- and
  slippage-aware and shared by both `run_backtest()` (static snapshot data) and paper
  trading (meant to be driven candle-by-candle from `viz/kraken_ws.py` - see below).
- **Risk**: `backtest/risk.py` restores the original bot's absolute-dollar
  `min_sl_distance` ($50 default) and 1.5R take-profit. Not yet re-validated against
  real backtest data - see `docs/` checklist below.
- **Dashboard**: `viz/server.py` (FastAPI+WS) + `viz/control.py` +
  `viz/static/` render candles and SMC overlays against snapshot data, and can run a
  full backtest with live progress/results/trade-log over the WebSocket. Paper
  trading's start/stop/status plumbing is real but **not wired to Kraken's live feed
  yet** (`viz/kraken_ws.py` exists and works standalone - next session's work is
  connecting it through `FillEngine` into `ControlPanel.start_paper`). Live trading
  is hard-gated behind `live/trader.kill_switch_ready()`, currently `False`.
- **Live trading**: `live/trader.py` can place real Kraken orders
  (`place_market_order`), but every entrypoint refuses to run until
  `kill_switch_ready()` returns `True` - see that function's docstring for what has
  to land first (drawdown kill-switch, WS reconnect/backoff, process separation from
  the dashboard).

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

# Or import Kraken's official bulk archive once you have it locally
python -m data.kraken_archive --zip path/to/Kraken_OHLCVT.zip --pair "BTC/USD" --interval 5

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
- [ ] WebSocket reconnect/backoff for the Kraken feed (`viz/kraken_ws.py` has none yet)
- [ ] Dashboard and trading engine split into separate processes
- [ ] Kill switch / circuit breaker (drawdown halt, API-error halt, connectivity-loss halt)
- [ ] Wire paper trading (`viz/control.py::start_paper`) to the live Kraken feed
- [ ] Re-validate `min_confluence`, `min_sl_distance`, and `rr_ratio` against real
      backtest data once real history is imported (all currently unvalidated defaults
      or explicit TODOs - see their respective docstrings)
