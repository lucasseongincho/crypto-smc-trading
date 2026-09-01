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
"""


class RiskManager:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        risk_pct: float = 0.01,
        rr_ratio: float = 1.5,
        min_sl_distance: float = 50.0,
    ):
        """
        risk_pct: fraction of current balance to risk per trade (default 1%).
        rr_ratio: take-profit distance as a multiple of stop-loss distance (default
        1.5R) - restored from the original crypto bot; goes back on the re-validation
        checklist rather than being assumed correct.
        min_sl_distance: minimum stop-loss distance in absolute dollars. A tighter SL
        than this is treated as noise and the trade is rejected (size 0).
        """
        self.current_balance = initial_balance
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.min_sl_distance = min_sl_distance

    def calculate_size(self, entry_price: float, stop_loss: float) -> float:
        """Size (in BTC) that risks risk_pct of current_balance given entry/stop_loss.
        Returns 0.0 (reject) if entry_price is non-positive or the SL distance is
        below min_sl_distance."""
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
        return min(ideal_qty, max_affordable_qty)

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
