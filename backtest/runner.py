"""
runner.py
Fee-and-slippage-aware fill simulator. FillEngine is the one execution-simulation
engine shared by `--mode backtest` (historical snapshot data, driven by run_backtest()
below) and paper trading (live Kraken WS candles, driven one at a time via
FillEngine.on_candle() - see viz/control.py) - see the kickoff decision: one
simulator, two data sources, so paper and backtest can't silently drift onto
different fill assumptions. Only live/trader.py ever calls Kraken's real
order-placement API.

Fees and slippage are modeled explicitly and are not optional: Kraken's base-tier
taker fee (~0.40%) compounds fast on a 5-minute-bar strategy with frequent entries,
and tuning against fee-free numbers would just mean redoing the tuning later.

Same-bar TP/SL ambiguity: when a single candle's high/low range crosses both the
stop-loss and take-profit, OHLC data alone can't say which happened first. This
simulator always resolves that in the stop-loss's favor (checks SL before TP) - the
conservative assumption, not an attempt to guess intrabar sequencing.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from backtest.risk import RiskManager
from data.smc import DEFAULT_LOOKBACK_BARS, compute_smc_features
from signals.smc_aggregator import SMCSignal, SMCSignalAggregator


@dataclass
class FillConfig:
    taker_fee_pct: float = 0.0040  # Kraken base-tier taker fee, ~0.40%
    slippage_bps: float = 5.0      # applied against the trader on every simulated fill


@dataclass
class Position:
    side: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    entry_time: Any
    entry_fee: float
    confluence: int


@dataclass
class Trade:
    side: str
    entry_time: Any
    exit_time: Any
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    r_multiple: float
    confluence: int
    exit_reason: str  # "TP" | "SL"


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    buy_hold_curve: list[dict[str, Any]] = field(default_factory=list)
    final_balance: float = 0.0
    initial_balance: float = 0.0

    def metrics(self) -> dict[str, Any]:
        n = len(self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        peak = self.initial_balance
        max_dd = 0.0
        for point in self.equity_curve:
            peak = max(peak, point["balance"])
            if peak > 0:
                max_dd = max(max_dd, (peak - point["balance"]) / peak)

        buy_hold_return_pct = None
        if len(self.buy_hold_curve) >= 2:
            start, end = self.buy_hold_curve[0]["balance"], self.buy_hold_curve[-1]["balance"]
            buy_hold_return_pct = (end - start) / start * 100 if start else None

        return {
            "total_trades": n,
            "win_rate_pct": (len(wins) / n * 100) if n else 0.0,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
            "total_pnl": self.final_balance - self.initial_balance,
            "roi_pct": (self.final_balance - self.initial_balance) / self.initial_balance * 100
            if self.initial_balance else 0.0,
            "max_drawdown_pct": max_dd * 100,
            "avg_r_multiple": (sum(t.r_multiple for t in self.trades) / n) if n else 0.0,
            "buy_hold_return_pct": buy_hold_return_pct,
        }


def _fill_price(price: float, action: str, slippage_bps: float) -> float:
    """Slippage always works against the trader: buying fills higher, selling fills
    lower, regardless of whether it's an entry or an exit."""
    factor = slippage_bps / 10_000
    return price * (1 + factor) if action == "BUY" else price * (1 - factor)


def _structural_stop_loss(features: dict[str, Any], side: str, current_price: float) -> float:
    """Ported from the original crypto bot's strategy.py: place the stop just past
    the most recent order block on the trade's side, if there is one and it's on the
    correct side of price; otherwise fall back to a flat 1% buffer."""
    order_blocks = features.get("order_blocks") or []
    last_ob = order_blocks[-1] if order_blocks else None

    if side == "BUY":
        if last_ob and last_ob["type"] == "bullish" and last_ob["low"] < current_price:
            return last_ob["low"]
        return current_price * 0.99

    if last_ob and last_ob["type"] == "bearish" and last_ob["high"] > current_price:
        return last_ob["high"]
    return current_price * 1.01


class FillEngine:
    """Bar-by-bar signal + fill simulation, one candle at a time. A signal computed
    on a closed candle's close can only fill at the *next* candle's open (no
    same-bar lookahead) - so a signal fires on on_candle(bar_i) and actually enters
    on the following on_candle(bar_i+1) call, using that call's open price. Kraken
    spot has no margin/shorting, so a BEARISH signal while flat is skipped rather
    than opened as a short (see live/trader.py for the execution-side version of the
    same constraint); a BEARISH signal while already long is not treated as an exit
    signal either (exits are SL/TP only) - a real simplification worth revisiting
    once enough trades exist to check whether early SMC-driven exits would help.

    run_backtest() below drives this over a static historical DataFrame; paper
    trading (viz/control.py) drives the same on_candle() one live candle at a time.

    The rolling window is bounded to lookback_bars candles (a deque, not an
    ever-growing DataFrame) - compute_smc_features only ever looks at the most
    recent lookback_bars candles anyway, and an unbounded buffer would make both a
    long historical backtest and an hours-long paper session quadratic in candle
    count.
    """

    def __init__(
        self,
        aggregator: SMCSignalAggregator,
        risk: RiskManager,
        fill_config: FillConfig | None = None,
        lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    ):
        self.aggregator = aggregator
        self.risk = risk
        self.fill_config = fill_config or FillConfig()
        self.lookback_bars = lookback_bars

        self._window: deque[tuple[Any, float, float, float, float]] = deque(maxlen=lookback_bars)
        self.position: Position | None = None
        self.trades: list[Trade] = []
        self.equity_curve: list[dict[str, Any]] = []

        self._pending_side: str | None = None
        self._pending_signal: SMCSignal | None = None
        self._pending_features: dict[str, Any] | None = None
        self._pending_current_price: float | None = None

    def on_candle(self, time: Any, open_: float, high: float, low: float, close: float) -> dict[str, Any] | None:
        """Feed one closed candle in. Returns an {"type": "entry"|"exit", ...} event
        dict when this candle caused a fill, else None - callers (e.g. the
        dashboard's WS broadcast) use that to know when something happened."""
        self._window.append((time, open_, high, low, close))

        if self._pending_side is not None:
            return self._enter(time, open_)
        if self.position is not None:
            return self._check_exit(time, high, low)

        self._evaluate_signal(close)
        return None

    def _window_df(self) -> pd.DataFrame:
        idx = [r[0] for r in self._window]
        return pd.DataFrame(
            {
                "open": [r[1] for r in self._window],
                "high": [r[2] for r in self._window],
                "low": [r[3] for r in self._window],
                "close": [r[4] for r in self._window],
            },
            index=idx,
        )

    def _evaluate_signal(self, close: float) -> None:
        if len(self._window) < 2:
            return  # not enough data for any detector to fire yet
        features = compute_smc_features(self._window_df(), lookback_bars=self.lookback_bars)
        signal = self.aggregator.aggregate(features)
        if signal.direction == "BULLISH":
            self._pending_side = "BUY"
            self._pending_signal = signal
            self._pending_features = features
            self._pending_current_price = close
        # BEARISH: no short-selling on spot - skip. NEUTRAL: nothing to do.

    def _enter(self, time: Any, open_price: float) -> dict[str, Any] | None:
        side = self._pending_side
        signal = self._pending_signal
        features = self._pending_features
        current_price = self._pending_current_price
        self._pending_side = self._pending_signal = self._pending_features = self._pending_current_price = None

        stop_loss = _structural_stop_loss(features, side, current_price)
        entry_price = _fill_price(open_price, side, self.fill_config.slippage_bps)
        size = self.risk.calculate_size(entry_price, stop_loss)
        if size <= 0:
            return None

        take_profit = self.risk.take_profit_price(entry_price, stop_loss, side)
        entry_fee = entry_price * size * self.fill_config.taker_fee_pct
        self.position = Position(
            side=side, entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            size=size, entry_time=time, entry_fee=entry_fee, confluence=signal.confluence_score,
        )
        return {
            "type": "entry", "side": side, "entry_price": entry_price,
            "stop_loss": stop_loss, "take_profit": take_profit, "size": size,
        }

    def _check_exit(self, time: Any, high: float, low: float) -> dict[str, Any] | None:
        pos = self.position
        exit_price_raw = None
        reason = None

        if low <= pos.stop_loss:
            exit_price_raw, reason = pos.stop_loss, "SL"
        elif high >= pos.take_profit:
            exit_price_raw, reason = pos.take_profit, "TP"

        if not reason:
            return None

        exit_price = _fill_price(exit_price_raw, "SELL", self.fill_config.slippage_bps)
        exit_fee = exit_price * pos.size * self.fill_config.taker_fee_pct
        gross = (exit_price - pos.entry_price) * pos.size
        pnl = gross - pos.entry_fee - exit_fee
        risk_amount = abs(pos.entry_price - pos.stop_loss) * pos.size
        r_multiple = pnl / risk_amount if risk_amount else 0.0

        self.risk.current_balance += pnl
        trade = Trade(
            side=pos.side, entry_time=pos.entry_time, exit_time=time,
            entry_price=pos.entry_price, exit_price=exit_price, size=pos.size,
            pnl=pnl, r_multiple=r_multiple, confluence=pos.confluence, exit_reason=reason,
        )
        self.trades.append(trade)
        self.equity_curve.append({"time": _iso(time), "balance": self.risk.current_balance})
        self.position = None
        return {"type": "exit", "reason": reason, "exit_price": exit_price, "pnl": pnl, "r_multiple": r_multiple}


def _iso(time: Any) -> str:
    return time.isoformat() if hasattr(time, "isoformat") else str(time)


def run_backtest(
    df: pd.DataFrame,
    aggregator: SMCSignalAggregator,
    risk: RiskManager,
    fill_config: FillConfig | None = None,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int = 200,
) -> BacktestResult:
    """Drives FillEngine over a static historical DataFrame."""
    initial_balance = risk.current_balance
    engine = FillEngine(aggregator, risk, fill_config, lookback_bars)
    n = len(df)

    for i, row in enumerate(df.itertuples()):
        engine.on_candle(row.Index, float(row.open), float(row.high), float(row.low), float(row.close))
        if on_progress and (i + 1) % progress_every == 0:
            on_progress({"candles_processed": i + 1, "candles_total": n, "balance": risk.current_balance})

    if on_progress:
        on_progress({"candles_processed": n, "candles_total": n, "balance": risk.current_balance})

    equity_curve = [{"time": df.index[0].isoformat(), "balance": initial_balance}] + engine.equity_curve
    buy_hold_curve = _buy_hold_curve(df, initial_balance)

    return BacktestResult(
        trades=engine.trades,
        equity_curve=equity_curve,
        buy_hold_curve=buy_hold_curve,
        final_balance=risk.current_balance,
        initial_balance=initial_balance,
    )


def _buy_hold_curve(df: pd.DataFrame, initial_balance: float) -> list[dict[str, Any]]:
    """Baseline equity curve for a naive buy-and-hold of the same initial balance,
    entering at the first close and marking to each subsequent close - for the
    dashboard's SMC-strategy-vs-buy&hold comparison."""
    if df.empty:
        return []
    entry_price = float(df.iloc[0]["close"])
    units = initial_balance / entry_price
    return [
        {"time": ts.isoformat(), "balance": float(row["close"]) * units}
        for ts, row in df.iloc[:: max(1, len(df) // 500)].iterrows()
    ]
