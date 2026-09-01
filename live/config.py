"""
config.py
API key handling for real trading. Kraken keys are only needed once live trading is
actually enabled (paper trading uses backtest/runner.py's fill simulator against
Kraken's public WS feed instead, so it needs no key). Everything here loads from
.env, which is gitignored - never commit real keys.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class KrakenCredentials:
    api_key: str
    api_secret: str


@dataclass(frozen=True)
class TelegramCredentials:
    bot_token: str
    chat_id: str


def load_kraken_credentials() -> KrakenCredentials:
    """Raises RuntimeError with a clear message if KRAKEN_API_KEY/KRAKEN_API_SECRET
    aren't set - callers should only invoke this once they're actually about to place
    a real order, not at import time (so paper/backtest runs don't need a key)."""
    api_key = os.getenv("KRAKEN_API_KEY", "").strip()
    api_secret = os.getenv("KRAKEN_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError(
            "KRAKEN_API_KEY / KRAKEN_API_SECRET are not set in .env. "
            "Generate an API key+secret pair from your Kraken account "
            "(Settings -> API) with Query Funds + Create/Cancel Orders permissions."
        )
    return KrakenCredentials(api_key=api_key, api_secret=api_secret)


def load_telegram_credentials() -> TelegramCredentials | None:
    """None (not an error) if unset - Telegram notifications are optional."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return None
    return TelegramCredentials(bot_token=bot_token, chat_id=chat_id)
