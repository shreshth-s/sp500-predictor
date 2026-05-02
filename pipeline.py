"""
Pipeline orchestrator — wires filters, scoring, and monitoring together.
"""

import os
from datetime import datetime

import pandas as pd

from config import MIN_MARKET_CAP, OUTPUT_DIR, TOP_N_CANDIDATES
from data_sources import get_large_cap_universe, normalize_sector_name
from fmp_client import FMPClient
from filters import (
    apply_data_quality_filter,
    apply_liquidity_filter,
    apply_profitability_filter,
    exclude_current_sp500,
    get_eligible_universe,
)
from scoring import score_candidates
from monitor import get_bottom_n_sp500, check_removal_candidates


class Pipeline:
    """Orchestrates the full filter -> score -> rank workflow."""

    def __init__(self, client: FMPClient):
        self.client = client

    def run(self, skip_profitability=False, top_n=TOP_N_CANDIDATES) -> pd.DataFrame:
        """
        Full pipeline execution:
        1. Hard filters -> eligible universe
        2. Score and rank candidates
        3. Return top N
        """
        print("=" * 60)
        print("  S&P 500 SHADOW COMMITTEE - Inclusion Predictor")
        print("=" * 60)
        print()

        candidates = get_eligible_universe(
            client=self.client,
            skip_profitability=skip_profitability
        )

        if candidates.empty:
            print("\nNo eligible candidates found.")
            return candidates

        ranked = score_candidates(candidates, client=self.client)

        print(f"\nTop {min(top_n, len(ranked))} predicted candidates:")
        print("-" * 40)

        return ranked.head(top_n)

    @staticmethod
    def _profile_to_candidate_row(profile: dict) -> dict:
        """Normalize FMP profile dict into candidate row schema."""
        market_cap = profile.get("mktCap", profile.get("marketCap"))
        try:
            market_cap = float(market_cap) if market_cap is not None else None
        except (TypeError, ValueError):
            market_cap = None

        return {
            "symbol": str(profile.get("symbol", "")).upper(),
            "companyName": profile.get("companyName") or profile.get("symbol"),
            "marketCap": market_cap,
            "sector": normalize_sector_name(profile.get("sector")),
            "industry": profile.get("industry"),
            "country": profile.get("country"),
            "exchange": profile.get("exchangeShortName") or profile.get("exchange"),
            "is_etf": profile.get("isEtf"),
            "is_fund": profile.get("isFund"),
            "is_adr": profile.get("isAdr"),
            "is_actively_trading": profile.get("isActivelyTrading"),
        }

    def score_single_ticker(self, ticker: str, skip_profitability=False) -> dict:
        """
        Evaluate one ticker through hard filters and scoring.
        If eligible, returns full score and rank among all eligible candidates.
        """
        symbol = ticker.strip().upper()
        universe = get_large_cap_universe(self.client)

        row_df = pd.DataFrame()
        if not universe.empty and "symbol" in universe.columns:
            row_df = universe[universe["symbol"].astype(str).str.upper() == symbol].head(1).copy()

        if row_df.empty:
            profile = self.client.company_profile(symbol)
            if profile:
                row_df = pd.DataFrame([self._profile_to_candidate_row(profile)])

        if row_df.empty:
            return {
                "symbol": symbol,
                "found": False,
                "eligible": False,
                "reason": "Ticker not found in current data sources.",
            }

        row = row_df.iloc[0]
        hard_filters = {
            "market_cap": (pd.notna(row.get("marketCap")) and float(row.get("marketCap")) >= MIN_MARKET_CAP),
            "domicile": (str(row.get("country", "")).strip() in {"United States", "US", "USA"}),
            "data_quality": not apply_data_quality_filter(row_df).empty,
            "not_in_sp500": not exclude_current_sp500(row_df).empty,
        }

        if skip_profitability:
            hard_filters["profitability"] = True
            prof_df = row_df.copy()
            prof_df["q1_net_income"] = None
            prof_df["ttm_net_income"] = None
            prof_df["ttm_revenue"] = None
            prof_df["ttm_net_margin"] = None
        else:
            prof_df = apply_profitability_filter(row_df, client=self.client)
            hard_filters["profitability"] = not prof_df.empty

        # Liquidity & float check
        if hard_filters["profitability"]:
            liq_df = apply_liquidity_filter(prof_df, client=self.client)
            hard_filters["liquidity"] = not liq_df.empty
            if not liq_df.empty:
                prof_df = liq_df
        else:
            hard_filters["liquidity"] = False

        eligible = all(hard_filters.values())
        out = {
            "symbol": symbol,
            "found": True,
            "eligible": eligible,
            "hard_filters": hard_filters,
            "candidate_row": prof_df if hard_filters["profitability"] else row_df,
        }

        if not eligible:
            return out

        candidates = get_eligible_universe(
            client=self.client,
            skip_profitability=skip_profitability,
        )
        ranked = score_candidates(candidates, client=self.client)
        hit = ranked[ranked["symbol"].astype(str).str.upper() == symbol]

        if hit.empty:
            out["eligible"] = False
            out["reason"] = "Ticker passed quick checks but was not found in ranked universe."
            return out

        rank = int(hit.index[0]) + 1
        out["rank"] = rank
        out["universe_size"] = len(ranked)
        out["score_row"] = hit.reset_index(drop=True)
        return out

    def run_removal_scenario(self, removed_ticker: str, top_n=TOP_N_CANDIDATES) -> pd.DataFrame:
        """
        Simulate a specific removal and predict the replacement.
        """
        print("=" * 60)
        print(f"  REMOVAL SCENARIO: {removed_ticker}")
        print("=" * 60)
        print()

        try:
            profile = self.client.company_profile(removed_ticker)
            if profile and isinstance(profile, dict):
                print(f"  Removed: {profile.get('companyName', removed_ticker)}")
                print(f"  Sector:  {profile.get('sector', 'Unknown')}")
                cap = profile.get("mktCap") or profile.get("marketCap", 0)
                if cap:
                    print(f"  Mkt Cap: ${cap/1e9:.1f}B")
                print()
        except Exception:
            pass

        result = self.run(top_n=top_n)

        if not result.empty:
            top = result.iloc[0]
            print(f"\n>>> Top prediction: {top['symbol']} "
                  f"(Score: {top['total_score']:.1f})")

        return result

    def show_watchlist(self, bottom_n=10, top_n=TOP_N_CANDIDATES) -> dict:
        """
        Quick watchlist view:
        - Bottom N S&P 500 members at risk of removal
        - Top N candidates for inclusion
        """
        print("=" * 60)
        print("  S&P 500 WATCHLIST")
        print("=" * 60)
        print()

        print(f"--- Bottom {bottom_n} S&P 500 by Market Cap (Removal Risk) ---")
        bottom = get_bottom_n_sp500(self.client, n=bottom_n)

        print()
        print("--- At-Risk Members (Market Cap < $10B) ---")
        at_risk = check_removal_candidates(self.client)

        if at_risk.empty:
            print("  No S&P 500 members currently below $10B threshold.")

        print()
        print(f"--- Top {top_n} Inclusion Candidates ---")
        candidates = self.run(skip_profitability=True, top_n=top_n)

        return {
            "bottom": bottom,
            "at_risk": at_risk,
            "candidates": candidates,
        }

    def export_results(self, df: pd.DataFrame, label: str = "results"):
        """Save results to the output directory as CSV."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.csv"
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"\nResults exported to: {path}")
        return path
