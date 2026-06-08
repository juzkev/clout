"""Telegram notification bot (stub).

Sends research summaries and trade alerts via a Telegram Bot.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env.
Degrades silently if tokens are not configured.
"""

import logging

from config import settings

logger = logging.getLogger(__name__)


class TelegramBot:
    """Thin wrapper around the Telegram Bot API for trade notifications."""

    def __init__(self):
        self.token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self._configured = bool(self.token and self.chat_id)
        if not self._configured:
            logger.info("TelegramBot: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — notifications disabled")

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a text message. Returns True on success, False if unconfigured or error.

        Intended implementation: POST to
        https://api.telegram.org/bot{token}/sendMessage
        with chat_id and text payload.
        """
        if not self._configured:
            return False
        raise NotImplementedError

    def send_trade_summary(self, llm_result: dict) -> bool:
        """Format and send a trade-idea summary from llm_client output."""
        raise NotImplementedError
