"""
notifications.py
Telegram push notifications - ported from the original Crypto-Trading-Bot's
notifications.py. Kept as-is per the kickoff decision: already built, free,
near-instant, and its inline-button support pairs well with a future
trade-confirmation flow.
"""
import requests

from live.config import load_telegram_credentials

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram_notification(message: str, prefix: str = "") -> bool:
    """Best-effort send - returns False (and prints, doesn't raise) if credentials
    are missing or the request fails, so a notification outage never takes down the
    trading loop that's calling it."""
    creds = load_telegram_credentials()
    if creds is None:
        print("Telegram credentials not configured - notification not sent.")
        return False

    url = TELEGRAM_API_BASE.format(token=creds.bot_token)
    payload = {
        "chat_id": creds.chat_id,
        "text": f"{prefix}{message}",
        "parse_mode": "Markdown",
    }

    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Telegram notification: {e}")
        return False
