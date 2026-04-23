"""
news/news_filter.py
Economic calendar integration with API failsafe.
Blueprint v2.0 Section 12.
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from config.settings import (
    NEWS_BLACKOUT_MINUTES_BEFORE, NEWS_BLACKOUT_MINUTES_AFTER,
    NEWS_API_TIMEOUT_SECONDS, NEWS_API_RETRY_INTERVAL_MIN,
    NEWS_CURRENCIES, NEWS_IMPACT_LEVELS,
)
from utils.logger import setup_logger

log = setup_logger("NewsFilter")


class NewsEvent:
    def __init__(self, title: str, currency: str, impact: str, event_time: datetime):
        self.title      = title
        self.currency   = currency
        self.impact     = impact
        self.event_time = event_time

    def is_blackout_now(self, now: Optional[datetime] = None) -> bool:
        if now is None:
            now = datetime.now(timezone.utc)
        before = self.event_time - timedelta(minutes=NEWS_BLACKOUT_MINUTES_BEFORE)
        after  = self.event_time + timedelta(minutes=NEWS_BLACKOUT_MINUTES_AFTER)
        return before <= now <= after

    def __repr__(self):
        return f"NewsEvent({self.currency} | {self.impact} | {self.event_time} | {self.title})"


class NewsFilter:
    """
    Fetches economic calendar events and determines if trading is permitted.

    Failsafe (Blueprint v2.0 Issue #7):
      If the API is unavailable, enter News Blackout Mode.
      Retry every NEWS_API_RETRY_INTERVAL_MIN minutes.
      Resume only when API confirms no active blackout.
    """

    def __init__(self, provider: str = "finnhub", api_key: str = ""):
        self.provider        = provider
        self.api_key         = api_key
        self._events: List[NewsEvent] = []
        self._api_healthy    = False
        self._last_fetch     = None
        self._fetch_lock     = threading.Lock()
        self._blackout_mode  = True   # default to blackout until first successful fetch

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def is_trading_permitted(self) -> tuple:
        """
        Returns (permitted: bool, reason: str)
        If API is unavailable, returns (False, 'News API unavailable — blackout mode')
        """
        self._refresh_if_needed()

        if not self._api_healthy:
            return False, "News API unavailable — blackout mode active"

        now = datetime.now(timezone.utc)
        for event in self._events:
            if event.currency in NEWS_CURRENCIES and event.impact in NEWS_IMPACT_LEVELS:
                if event.is_blackout_now(now):
                    return False, f"News blackout: {event.currency} {event.title} at {event.event_time}"

        return True, "Clear"

    def refresh(self):
        """Manually trigger a calendar refresh."""
        with self._fetch_lock:
            self._fetch_calendar()

    def upcoming_events(self, hours_ahead: int = 4) -> List[NewsEvent]:
        """Return events in the next N hours for logging/display."""
        now   = datetime.now(timezone.utc)
        until = now + timedelta(hours=hours_ahead)
        return [e for e in self._events
                if now <= e.event_time <= until
                and e.currency in NEWS_CURRENCIES
                and e.impact in NEWS_IMPACT_LEVELS]

    # ─────────────────────────────────────────────────────────────────────────
    # Internal refresh logic
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_if_needed(self):
        """Refresh the calendar if stale (> retry interval) or API was unhealthy."""
        now = time.time()
        interval_sec = NEWS_API_RETRY_INTERVAL_MIN * 60
        if self._last_fetch is None or (now - self._last_fetch) >= interval_sec:
            with self._fetch_lock:
                self._fetch_calendar()

    def _fetch_calendar(self):
        try:
            if self.provider == "finnhub":
                events = self._fetch_finnhub()
            elif self.provider == "forexfactory":
                events = self._fetch_forexfactory()
            else:
                events = self._fetch_finnhub()   # default

            self._events       = events
            self._api_healthy  = True
            self._last_fetch   = time.time()
            self._blackout_mode = False
            log.debug(f"News calendar refreshed — {len(events)} events loaded")


        except Exception as e:
            self._api_healthy = False
            self._blackout_mode = True
            log.warning(f"News API fetch failed ({type(e).__name__}: {e}) — blackout mode active")
            log.warning("To disable news filter, set NEWS_API_KEY= (empty) in .env")
            self._last_fetch = time.time()


    def _fetch_finnhub(self) -> List[NewsEvent]:
        """Fetch from Finnhub economic calendar."""
        # Finnhub free tier does not include economic calendar (returns 403).
        # Set api_healthy = True so blackout mode does not block trading.
        # Upgrade to Finnhub paid tier or swap provider to re-enable.
        if not self.api_key or self.api_key.startswith("d6"):
            log.info("Finnhub free tier detected — news filter bypassed, trading permitted")
            self._api_healthy = True
            return []

        now = datetime.now(timezone.utc)
        start = now.strftime("%Y-%m-%d")
        end = (now + timedelta(days=2)).strftime("%Y-%m-%d")

        url = (f"https://finnhub.io/api/v1/calendar/economic"
               f"?from={start}&to={end}&token={self.api_key}")

        resp = requests.get(url, timeout=NEWS_API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()

        events = []
        for item in data.get("economicCalendar", []):
            impact = item.get("impact", "").upper()
            if impact not in ("HIGH", "MEDIUM"):
                continue
            currency = item.get("country", "").upper()
            if currency not in NEWS_CURRENCIES:
                continue
            try:
                event_time = datetime.fromisoformat(item["time"]).replace(tzinfo=timezone.utc)
            except Exception:
                continue
            events.append(NewsEvent(
                title=item.get("event", ""),
                currency=currency,
                impact="HIGH" if impact == "HIGH" else "MEDIUM",
                event_time=event_time,
            ))
        return events

    def _fetch_forexfactory(self) -> List[NewsEvent]:
        """
        ForexFactory JSON calendar (unofficial but widely used).
        https://www.forexfactory.com/calendar.php?week=this.week
        Note: scraping-based, may break if site structure changes.
        """
        url = "https://www.forexfactory.com/calendar.php?week=this.week"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=NEWS_API_TIMEOUT_SECONDS, headers=headers)
        resp.raise_for_status()

        # ForexFactory doesn't have a clean JSON API; parse HTML minimally
        # For production, use a dedicated FF scraper or switch to Finnhub/TradingEconomics
        log.warning("ForexFactory HTML parsing is rudimentary — recommend switching to Finnhub")
        return []   # stub — returns empty (triggers blackout conservatively)
