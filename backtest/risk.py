"""
risk.py
Position sizing and take-profit calculation for backtest/paper/live. Ported from the
original Crypto-Trading-Bot's risk.py::RiskManager, restoring what tradingagents-kr
had swapped out (a percent-of-price min-SL-distance, needed there because it sizes
stocks from $5 to $500) back to the original's absolute-dollar min_sl_distance - this
project is BTC/USD only, where an absolute-dollar floor makes sense again and doesn't
need to scale across tickers.

TODO: min_sl_distance's original value ($50) was tuned for BTC trading in the
multi-thousand-dollar range on 5m bars. Kept as the literal default here so the port
is faithful, but it goes back on the backtest re-validation checklist along with
rr_ratio - see the kickoff notes ("don't assume it's automatically good").

Capital-cap handling (skip_if_capital_capped): on unleveraged Spot, a tight
stop-loss distance can imply an "ideal" risk-pct-of-equity position size that costs
more than the account holds. The historical behavior (skip_if_capital_capped=False,
still the default) silently downsizes to whatever is affordable and takes the trade
anyway - which means the position's actual dollar risk is far larger than risk_pct
intended, while still paying full taker fees on the near-full-notional size. An
audit of the 2025-04-01..2025-12-31 tuning run found this hit 63.5% of trades on the
best-performing combo and accounted for the large majority of the realized loss.
skip_if_capital_capped=True is the cheaper structural fix: when the cap would bind,
skip the trade entirely rather than force a capital-capped, fee-heavy position -
see docs/tuning-log.md for the before/after comparison.
"""
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        risk_pct: float = 0.01,
        rr_ratio: float = 1.5,
        min_sl_distance: float = 50.0,
        skip_if_capital_capped: bool = False,
    ):
        """
        risk_pct: fraction of current balance to risk per trade (default 1%).
        rr_ratio: take-profit distance as a multiple of stop-loss distance (default
        1.5R) - restored from the original crypto bot; goes back on the re-validation
        checklist rather than being assumed correct.
        min_sl_distance: minimum stop-loss distance in absolute dollars. A tighter SL
        than this is treated as noise and the trade is rejected (size 0).
        skip_if_capital_capped: when the risk-sized position would cost more than the
        account can afford, reject the trade (size 0) instead of silently downsizing
        to the affordable amount. Default False preserves the historical (forced,
        fee-heavy) behavior; see the module docstring.
        """
        self.current_balance = initial_balance
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.min_sl_distance = min_sl_distance
        self.skip_if_capital_capped = skip_if_capital_capped

        # Counters for reporting how often the capital cap binds, regardless of
        # whether it's skipping the trade or (historical default) forcing it through
        # undersized. sizing_attempts is the denominator - every call that got past
        # the entry_price/min_sl_distance rejects and reached the cap comparison.
        self.sizing_attempts: int = 0
        self.capital_capped_trades: int = 0
        self.capital_capped_skipped: int = 0

    def calculate_size(self, entry_price: float, stop_loss: float) -> float:
        """Size (in BTC) that risks risk_pct of current_balance given entry/stop_loss.
        Returns 0.0 (reject) if entry_price is non-positive or the SL distance is
        below min_sl_distance. Also returns 0.0 (reject) if the ideal risk-sized
        position is more than the account can afford and skip_if_capital_capped is
        True - see the module docstring."""
        if entry_price <= 0:
            return 0.0

        risk_amount = self.current_balance * self.risk_pct
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0 or risk_per_unit < self.min_sl_distance:
            return 0.0

        ideal_qty = risk_amount / risk_per_unit

        # Spot purchasing-power cap: leave a 2% buffer for fees so a maxed-out size
        # doesn't get rejected by the exchange for insufficient balance.
        max_affordable_qty = (self.current_balance * 0.98) / entry_price

        self.sizing_attempts += 1
        if ideal_qty > max_affordable_qty:
            self.capital_capped_trades += 1
            if self.skip_if_capital_capped:
                self.capital_capped_skipped += 1
                logger.debug(
                    "skip_if_capital_capped: rejecting trade, ideal_qty=%.6f exceeds "
                    "max_affordable_qty=%.6f (entry_price=%.2f, balance=%.2f)",
                    ideal_qty, max_affordable_qty, entry_price, self.current_balance,
                )
                return 0.0
            logger.debug(
                "capital cap bound: forcing undersized trade, ideal_qty=%.6f -> "
                "%.6f (entry_price=%.2f, balance=%.2f)",
                ideal_qty, max_affordable_qty, entry_price, self.current_balance,
            )
            return max_affordable_qty

        return ideal_qty

    def take_profit_price(self, entry_price: float, stop_loss: float, side: str) -> float:
        """Take-profit price at rr_ratio times the stop-loss distance from entry."""
        risk_per_unit = abs(entry_price - stop_loss)
        if side.upper() in ("BUY", "LONG"):
            return entry_price + risk_per_unit * self.rr_ratio
        return entry_price - risk_per_unit * self.rr_ratio

    def apply_result(self, result: str) -> float:
        """Apply a WIN/LOSS result to current_balance and return the realized PnL."""
        risk_amount = self.current_balance * self.risk_pct
        pnl = risk_amount * self.rr_ratio if result == "WIN" else -risk_amount
        self.current_balance += pnl
        return round(pnl, 2)
