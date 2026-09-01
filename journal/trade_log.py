"""
trade_log.py
Shared trade-record schema for the SMC strategy (and any future strategy source) plus
CSV persistence. Ported from tradingagents-kr's journal/{schema,store,adapters}.py,
consolidated into one module per this project's structure.

The original crypto bot's writer (live_main.py) and reader (pages/2_journal.py) each
invented their own CSV columns independently and drifted apart (Date/Pair/Side/Entry/
Exit/Result/PnL/Balance vs Entry_Date/Exit_Date/Side/P/L_USD). TradeLogEntry is the
single schema both the writer and reader import, so that kind of drift is structurally
not possible here - append_entries() and read_entries() both derive their columns from
the same model.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from backtest.runner import Trade
from signals.smc_aggregator import SMCSignal

TRADE_JOURNAL_FILENAME = "trade_journal.csv"

_DIRECTION_TO_DECISION = {"BULLISH": "BUY", "BEARISH": "SELL", "NEUTRAL": "HOLD"}


class TradeLogEntry(BaseModel):
    source: Literal["smc", "ai_agent", "combined"]
    ticker: str
    timestamp: datetime
    decision: Literal["BUY", "SELL", "HOLD"]
    confidence_or_confluence: float
    reasoning: str
    position_sizing: dict
    outcome: dict | None = None  # filled in once the trade closes (fill price, PnL, exit reason)


def smc_signal_to_entry(
    signal: SMCSignal,
    ticker: str,
    timestamp: datetime,
    position_sizing: dict,
) -> TradeLogEntry:
    return TradeLogEntry(
        source="smc",
        ticker=ticker,
        timestamp=timestamp,
        decision=_DIRECTION_TO_DECISION[signal.direction],
        confidence_or_confluence=float(signal.confluence_score),
        reasoning=signal.to_anchor_text(),
        position_sizing=position_sizing,
        outcome=None,
    )


def trade_to_entry(trade: Trade, ticker: str) -> TradeLogEntry:
    """A closed backtest/paper Trade, expressed as a TradeLogEntry with outcome
    already filled in - for logging completed simulated trades alongside live ones."""
    return TradeLogEntry(
        source="smc",
        ticker=ticker,
        timestamp=trade.entry_time,
        decision="BUY" if trade.side == "BUY" else "SELL",
        confidence_or_confluence=float(trade.confluence),
        reasoning=f"Simulated fill, exited on {trade.exit_reason}",
        position_sizing={"size": trade.size, "entry_price": trade.entry_price},
        outcome={
            "exit_time": trade.exit_time.isoformat() if hasattr(trade.exit_time, "isoformat") else str(trade.exit_time),
            "exit_price": trade.exit_price,
            "pnl": trade.pnl,
            "r_multiple": trade.r_multiple,
            "exit_reason": trade.exit_reason,
        },
    )


def _entry_to_row(entry: TradeLogEntry) -> dict:
    row = entry.model_dump(mode="json")
    row["position_sizing"] = json.dumps(entry.position_sizing, ensure_ascii=False)
    row["outcome"] = json.dumps(entry.outcome, ensure_ascii=False) if entry.outcome is not None else ""
    return row


def _row_to_entry(row: dict) -> TradeLogEntry:
    data = dict(row)
    ps_raw = data.get("position_sizing")
    data["position_sizing"] = json.loads(ps_raw) if isinstance(ps_raw, str) and ps_raw else {}
    outcome_raw = data.get("outcome")
    data["outcome"] = json.loads(outcome_raw) if isinstance(outcome_raw, str) and outcome_raw else None
    return TradeLogEntry.model_validate(data)


def append_entries(entries: list[TradeLogEntry], path: Path) -> None:
    """Append entries to path, creating it with a header if it doesn't exist yet."""
    if not entries:
        return
    rows = [_entry_to_row(e) for e in entries]
    df = pd.DataFrame(rows)
    file_exists = path.exists()
    df.to_csv(path, mode="a" if file_exists else "w", header=not file_exists, index=False)


def read_entries(path: Path) -> list[TradeLogEntry]:
    """Empty list if path doesn't exist yet. Round-trips exactly what append_entries()
    wrote, since both derive their columns from TradeLogEntry."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [_row_to_entry(row.to_dict()) for _, row in df.iterrows()]
