"""
Financial Modeling Prep API client with rate limiting and disk caching.
Uses stable endpoints compatible with newer FMP account tiers.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import requests

from config import (
    CACHE_DIR,
    CACHE_TTL_HOURS,
    FMP_API_KEY,
    FMP_REQUESTS_PER_MINUTE,
    FMP_RETRY_ATTEMPTS,
    FMP_RETRY_BACKOFF,
)

FMP_STABLE_URL = "https://financialmodelingprep.com/stable"


class FMPClient:
    """Thin wrapper around FMP stable API with rate limiting and disk cache."""

    def __init__(self, api_key=None, use_cache=True):
        self._api_key = api_key or FMP_API_KEY
        self._base_url = FMP_STABLE_URL
        self._min_interval = 60.0 / FMP_REQUESTS_PER_MINUTE
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._use_cache = use_cache
        self._cache_dir = Path(CACHE_DIR)
        self._cache_dir.mkdir(exist_ok=True)
        self._batch_quote_supported = None

    def _rate_limit(self):
        """Sleep if needed to respect the rate limit."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _cache_path(self, endpoint, params):
        """Deterministic cache file path based on endpoint and params."""
        key = endpoint + "::" + json.dumps(params, sort_keys=True)
        digest = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{digest}.json"

    def _get_cached(self, cache_file):
        """Return cached JSON if fresh enough, else None."""
        if not self._use_cache or not cache_file.exists():
            return None
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours > CACHE_TTL_HOURS:
            return None
        with open(cache_file, "r") as f:
            return json.load(f)

    def _save_cache(self, cache_file, data):
        """Write JSON response to disk cache."""
        with open(cache_file, "w") as f:
            json.dump(data, f)

    @staticmethod
    def _unwrap(data):
        """Unwrap stable response envelope: {'value': [...], 'Count': n}."""
        if isinstance(data, dict) and "value" in data:
            return data["value"]
        return data

    def get(self, endpoint, params=None):
        """
        GET request with cache check -> rate limit -> retry logic.
        Returns parsed JSON payload.
        Raises RuntimeError on persistent failure.
        """
        params = params or {}
        params["apikey"] = self._api_key

        cache_file = self._cache_path(endpoint, params)
        cached = self._get_cached(cache_file)
        if cached is not None:
            return cached

        url = f"{self._base_url}/{endpoint}"

        for attempt in range(1, FMP_RETRY_ATTEMPTS + 1):
            self._rate_limit()
            self._last_request_time = time.monotonic()
            try:
                response = self._session.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                if attempt == FMP_RETRY_ATTEMPTS:
                    raise RuntimeError(f"FMP request failed after {attempt} attempts: {e}")
                time.sleep(FMP_RETRY_BACKOFF ** attempt)
                continue

            if response.status_code == 200:
                data = response.json()
                self._save_cache(cache_file, data)
                return data

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == FMP_RETRY_ATTEMPTS:
                    raise RuntimeError(
                        f"FMP returned {response.status_code} after {attempt} attempts"
                    )
                time.sleep(FMP_RETRY_BACKOFF ** attempt)
                continue

            body = response.text[:400]
            raise RuntimeError(f"FMP error {response.status_code}: {body}")

    def stock_screener(
        self,
        market_cap_more_than=None,
        country="US",
        limit=1000,
        page=None,
        exchange=None,
        is_etf=False,
        is_fund=False,
    ):
        """
        GET /company-screener.
        Some plans restrict this endpoint; caller should catch RuntimeError and fallback.
        """
        params = {"country": country, "limit": limit}
        if market_cap_more_than is not None:
            params["marketCapMoreThan"] = market_cap_more_than
        if page is not None:
            params["page"] = page
        if exchange:
            params["exchange"] = exchange
        if is_etf is not None:
            params["isEtf"] = str(bool(is_etf)).lower()
        if is_fund is not None:
            params["isFund"] = str(bool(is_fund)).lower()

        data = self.get("company-screener", params)
        return self._unwrap(data)

    def income_statement(self, ticker, period="quarter", limit=4):
        """GET /income-statement?symbol=TICKER."""
        data = self.get(
            "income-statement",
            {"symbol": ticker, "period": period, "limit": limit},
        )
        return self._unwrap(data)

    def company_profile(self, ticker):
        """GET /profile?symbol=TICKER."""
        data = self.get("profile", {"symbol": ticker})
        payload = self._unwrap(data)
        if isinstance(payload, list) and payload:
            return payload[0]
        if isinstance(payload, dict):
            return payload
        return None

    def quote(self, ticker):
        """GET /quote?symbol=TICKER."""
        data = self.get("quote", {"symbol": ticker})
        payload = self._unwrap(data)
        if isinstance(payload, list) and payload:
            return payload[0]
        if isinstance(payload, dict):
            return payload
        return None

    def can_batch_quote(self):
        """Detect whether multi-symbol quote calls are available on this plan."""
        if self._batch_quote_supported is not None:
            return self._batch_quote_supported
        try:
            data = self.get("quote", {"symbol": "AAPL,MSFT"})
            payload = self._unwrap(data)
            self._batch_quote_supported = bool(payload)
            return self._batch_quote_supported
        except RuntimeError as e:
            text = str(e).lower()
            if "premium query parameter" in text or "restricted endpoint" in text:
                self._batch_quote_supported = False
                return False
            raise

    def quote_batch(self, tickers: Iterable[str], chunk_size=100):
        """
        Fetch quotes for multiple tickers.
        Uses multi-symbol calls when available, otherwise falls back to one-symbol calls.
        """
        symbols = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if not symbols:
            return []

        out = []
        supports_batch = self.can_batch_quote()

        if supports_batch:
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i : i + chunk_size]
                joined = ",".join(chunk)
                try:
                    data = self.get("quote", {"symbol": joined})
                    payload = self._unwrap(data)
                except RuntimeError as e:
                    text = str(e).lower()
                    if "premium query parameter" in text or "restricted endpoint" in text:
                        self._batch_quote_supported = False
                        supports_batch = False
                        break
                    raise

                if isinstance(payload, list):
                    out.extend(payload)
                elif isinstance(payload, dict):
                    out.append(payload)

        if not supports_batch:
            for symbol in symbols:
                try:
                    quote = self.quote(symbol)
                except RuntimeError:
                    continue
                if quote:
                    out.append(quote)

        return out

    def shares_float(self, ticker):
        """GET /shares-float?symbol=TICKER — float shares and outstanding."""
        data = self.get("shares-float", {"symbol": ticker})
        payload = self._unwrap(data)
        if isinstance(payload, list) and payload:
            return payload[0]
        if isinstance(payload, dict):
            return payload
        return None

    # ── Historical data methods (for backtesting) ──────────────────

    def historical_price(self, ticker, date_from, date_to):
        """GET /historical-price-eod/light — daily close prices in a date range."""
        data = self.get(
            "historical-price-eod/light",
            {"symbol": ticker, "from": date_from, "to": date_to},
        )
        return self._unwrap(data) or []

    def historical_market_cap(self, ticker, date_from, date_to):
        """GET /historical-market-capitalization — daily market cap in a date range."""
        data = self.get(
            "historical-market-capitalization",
            {"symbol": ticker, "from": date_from, "to": date_to},
        )
        return self._unwrap(data) or []

    def enterprise_values(self, ticker, period="quarter", limit=20):
        """GET /enterprise-values — includes historical numberOfShares."""
        data = self.get(
            "enterprise-values",
            {"symbol": ticker, "period": period, "limit": limit},
        )
        return self._unwrap(data) or []

    def income_statement_full(self, ticker, period="quarter", limit=40):
        """Fetch extended income statement history for backtesting."""
        return self.income_statement(ticker, period=period, limit=limit)

    def delisted_companies(self, limit=100, page=0):
        """GET /delisted-companies with optional paging (plan-dependent)."""
        params = {"limit": limit}
        if page is not None:
            params["page"] = page
        data = self.get("delisted-companies", params)
        payload = self._unwrap(data)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        return []

    def clear_cache(self):
        """Delete all cached files."""
        count = 0
        for f in self._cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count
