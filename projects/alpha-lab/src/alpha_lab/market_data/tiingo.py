"""Header-authenticated, paced Tiingo EOD client with bounded retries."""

import re
import time
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime

import httpx


class TiingoError(RuntimeError):
    """Safe-to-display error without credentials or provider response bodies."""


def ticker_symbol(value: str) -> str:
    value = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,31}", value):
        raise ValueError("Invalid ticker symbol")
    return value


class TiingoClient:
    def __init__(self, api_key: str, *, requests_per_minute: float = 30,
                 retries: int = 3, transport=None, proxy: str | None = None,
                 sleep=time.sleep, clock=time.monotonic):
        if not api_key or not api_key.strip():
            raise ValueError("Set TIINGO_API_KEY in your environment or local .env")
        if not 0 < requests_per_minute <= 600 or not 0 <= retries <= 5:
            raise ValueError("Invalid request rate or retry count")
        self._http = httpx.Client(
            base_url="https://api.tiingo.com", timeout=60, follow_redirects=False,
            headers={"Authorization": "Token " + api_key.strip(), "Accept": "application/json"},
            transport=transport, proxy=proxy,
        )
        self._interval = 60 / requests_per_minute
        self._retries = retries
        self._sleep, self._clock = sleep, clock
        self._last_request = None

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _get(self, path: str, params=None):
        for attempt in range(self._retries + 1):
            if self._last_request is not None:
                self._sleep(max(0, self._interval - (self._clock() - self._last_request)))
            self._last_request = self._clock()
            try:
                response = self._http.get(path, params=params)
            except httpx.TransportError:
                if attempt == self._retries:
                    raise TiingoError("Tiingo connection failed after retries") from None
                self._sleep(min(60, 2 ** attempt))
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == self._retries:
                    raise TiingoError(f"Tiingo HTTP {response.status_code}; retry later")
                delay = 2 ** attempt
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        try:
                            delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                        except (ValueError, TypeError, OverflowError):
                            pass
                self._sleep(max(0, min(60, delay)))
                continue
            if response.status_code != 200:
                hint = "check your API key and EOD entitlement" if response.status_code in (401, 403) else "check the symbol and account limits"
                raise TiingoError(f"Tiingo HTTP {response.status_code}; {hint}")
            try:
                return response.json()
            except ValueError:
                raise TiingoError("Tiingo returned invalid JSON") from None

    def metadata(self, ticker: str) -> dict:
        result = self._get(f"/tiingo/daily/{ticker_symbol(ticker)}")
        if not isinstance(result, dict) or "ticker" not in result:
            raise TiingoError("Invalid Tiingo metadata response")
        return result

    def prices(self, ticker: str, start: date, end: date) -> list[dict]:
        if start > end:
            raise ValueError("start_date must be on or before end_date")
        result = self._get(f"/tiingo/daily/{ticker_symbol(ticker)}/prices", {
            "startDate": start.isoformat(), "endDate": end.isoformat(),
            "format": "json", "resampleFreq": "daily",
        })
        if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
            raise TiingoError("Invalid Tiingo prices response")
        return result
