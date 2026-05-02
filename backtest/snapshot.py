"""
As-of snapshot provider: fetch point-in-time data for a given date.
Ensures no look-ahead bias by only using data available before t0.
"""

from datetime import date, datetime, timedelta

import pandas as pd
from tqdm import tqdm

from config import MIN_FALR, MIN_MARKET_CAP, MIN_MONTHLY_VOLUME, PROFITABILITY_QUARTERS
from data_sources import normalize_sector_name
from fmp_client import FMPClient
from backtest.events import reconstruct_sp500_at

_US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "NYQ", "NMS", "NGM", "NCM", "ASE"}


class AsOfSnapshot:
    """
    Point-in-time data provider for a specific date.
    All data returned is strictly as-of t0 with no future information leaks.
    """

    def __init__(self, client: FMPClient, t0, current_sp500_df, all_events_df):
        """
        client: FMPClient instance
        t0: date or ISO string, the as-of date (prediction timestamp)
        current_sp500_df: today's S&P 500 DataFrame (for backward reconstruction)
        all_events_df: full events DataFrame from load_events()
        """
        self.client = client
        self.t0 = t0 if isinstance(t0, date) else datetime.fromisoformat(t0).date()
        self.t0_str = self.t0.isoformat()
        self._current_sp500 = current_sp500_df
        self._all_events = all_events_df
        self._sp500_members = None

    def sp500_members_at_t0(self):
        """Reconstruct S&P 500 membership at t0."""
        if self._sp500_members is None:
            self._sp500_members = reconstruct_sp500_at(
                self.t0, self._current_sp500, self._all_events
            )
        return self._sp500_members

    def get_market_cap_at_t0(self, ticker):
        """
        Get market cap for a ticker at t0 using historical-market-capitalization.
        Returns float or None.
        """
        date_from = (self.t0 - timedelta(days=7)).isoformat()
        date_to = self.t0_str
        try:
            data = self.client.historical_market_cap(ticker, date_from, date_to)
            if not data:
                return None
            for row in sorted(data, key=lambda r: r.get("date", ""), reverse=True):
                row_date = str(row.get("date", ""))
                if row_date and row_date <= self.t0_str:
                    cap = row.get("marketCap")
                    return float(cap) if cap is not None else None
            return None
        except Exception:
            return None

    def get_price_at_t0(self, ticker):
        """Get closing price at t0."""
        date_from = (self.t0 - timedelta(days=7)).isoformat()
        date_to = self.t0_str
        try:
            data = self.client.historical_price(ticker, date_from, date_to)
            if not data:
                return None
            for row in sorted(data, key=lambda r: r.get("date", ""), reverse=True):
                row_date = str(row.get("date", ""))
                if row_date and row_date <= self.t0_str:
                    price = row.get("price")
                    return float(price) if price is not None else None
            return None
        except Exception:
            return None

    def get_volume_at_t0(self, ticker):
        """Get average daily volume around t0 (30-day lookback)."""
        date_from = (self.t0 - timedelta(days=45)).isoformat()
        date_to = self.t0_str
        try:
            data = self.client.historical_price(ticker, date_from, date_to)
            if not data:
                return None
            volumes = []
            for row in data:
                row_date = str(row.get("date", ""))
                vol = row.get("volume")
                if row_date and row_date <= self.t0_str and vol is not None:
                    try:
                        volumes.append(float(vol))
                    except (TypeError, ValueError):
                        continue
            if not volumes:
                return None
            return sum(volumes) / len(volumes)
        except Exception:
            return None

    def check_profitability_at_t0(self, ticker):
        """
        PIT profitability check: only use statements with acceptedDate <= t0.
        Returns (q1_net_income, ttm_net_income, ttm_revenue, ttm_net_margin) or None.
        """
        try:
            statements = self.client.income_statement_full(
                ticker, period="quarter", limit=20
            )
            if not statements:
                return None

            pit_statements = []
            for s in statements:
                accepted = str(s.get("acceptedDate", ""))
                if accepted and accepted[:10] <= self.t0_str:
                    pit_statements.append(s)

            if len(pit_statements) < PROFITABILITY_QUARTERS:
                return None

            recent = pit_statements[:PROFITABILITY_QUARTERS]
            q1_ni_raw = recent[0].get("netIncome")
            if q1_ni_raw is None:
                return None
            q1_ni = float(q1_ni_raw)

            ttm_ni = sum(float(q.get("netIncome", 0) or 0) for q in recent)
            revenues = [float(q.get("revenue", 0) or 0) for q in recent]
            ttm_rev = sum(revenues)
            ttm_margin = (ttm_ni / ttm_rev) if ttm_rev > 0 else None

            if q1_ni <= 0 or ttm_ni <= 0:
                return None

            return q1_ni, ttm_ni, ttm_rev, ttm_margin
        except Exception:
            return None

    def build_candidate_universe(self, tickers, show_progress=False, metadata=None):
        """
        Build candidate DataFrame for tickers as-of t0.
        Applies market cap and domicile checks and excludes S&P 500 members at t0.
        """
        sp500_at_t0 = self.sp500_members_at_t0()
        metadata = metadata or {}
        results = []
        iterator = (
            tqdm(tickers, desc=f"Building universe at {self.t0_str}")
            if show_progress
            else tickers
        )

        for ticker in iterator:
            symbol = str(ticker).upper().strip()
            if not symbol or symbol in sp500_at_t0:
                continue

            market_cap = self.get_market_cap_at_t0(symbol)
            if market_cap is None or market_cap < MIN_MARKET_CAP:
                continue

            meta = metadata.get(symbol, {})
            profile = None
            try:
                profile = self.client.company_profile(symbol)
            except Exception:
                profile = None
            profile = profile or {}

            exchange = (
                profile.get("exchangeShortName")
                or profile.get("exchange")
                or meta.get("exchange")
            )
            country = profile.get("country") or meta.get("country")
            if not country and exchange and str(exchange).strip().upper() in _US_EXCHANGES:
                country = "US"

            if country not in ("United States", "US", "USA"):
                continue

            results.append(
                {
                    "symbol": symbol,
                    "companyName": profile.get("companyName")
                    or meta.get("companyName")
                    or symbol,
                    "marketCap": market_cap,
                    "sector": normalize_sector_name(
                        profile.get("sector") or meta.get("sector")
                    ),
                    "industry": profile.get("industry") or meta.get("industry"),
                    "country": country,
                    "exchange": exchange,
                }
            )

        return pd.DataFrame(results)

    def apply_profitability(self, candidates, show_progress=False):
        """
        Apply PIT profitability filter to candidates.
        Returns filtered DataFrame with profitability columns.
        """
        results = []
        iterator = candidates.iterrows()
        if show_progress:
            iterator = tqdm(
                iterator, total=len(candidates), desc="PIT profitability check"
            )

        for _, row in iterator:
            prof = self.check_profitability_at_t0(row["symbol"])
            if prof is None:
                continue
            q1_ni, ttm_ni, ttm_rev, ttm_margin = prof
            result = row.to_dict()
            result["q1_net_income"] = q1_ni
            result["ttm_net_income"] = ttm_ni
            result["ttm_revenue"] = ttm_rev
            result["ttm_net_margin"] = ttm_margin
            results.append(result)

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).reset_index(drop=True)

    def apply_liquidity(self, candidates, show_progress=False):
        """
        Apply liquidity filter using historical volume data at t0.
        Uses market cap as conservative proxy denominator for FALR.
        """
        results = []
        iterator = candidates.iterrows()
        if show_progress:
            iterator = tqdm(
                iterator, total=len(candidates), desc="PIT liquidity check"
            )

        for _, row in iterator:
            ticker = row["symbol"]
            avg_volume = self.get_volume_at_t0(ticker)
            if avg_volume is None or avg_volume < MIN_MONTHLY_VOLUME:
                continue

            price = self.get_price_at_t0(ticker)
            market_cap = row.get("marketCap", 0)
            if not price or not market_cap or market_cap <= 0:
                continue

            annual_dollar_volume = avg_volume * price * 252
            falr = annual_dollar_volume / market_cap
            if falr < MIN_FALR:
                continue

            result = row.to_dict()
            result["avg_daily_volume"] = avg_volume
            result["falr"] = round(falr, 4)
            results.append(result)

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).reset_index(drop=True)

    def get_confidence_tier(self, candidates):
        """
        Assess data confidence for this snapshot.
        A = full PIT financial and liquidity fields
        B = PIT profitability only
        C = major gaps
        """
        if candidates.empty:
            return "C"

        has_profitability = (
            "ttm_net_income" in candidates.columns
            and candidates["ttm_net_income"].notna().any()
        )
        has_liquidity = (
            "falr" in candidates.columns and candidates["falr"].notna().any()
        )

        if has_profitability and has_liquidity:
            return "A"
        if has_profitability:
            return "B"
        return "C"
