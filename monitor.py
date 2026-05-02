"""
Phase 3 event monitoring: track removal catalysts and trigger predictions.
"""

import pandas as pd

from config import REMOVAL_RISK_CAP
from data_sources import get_index_market_data, get_sp500_tickers, normalize_sector_name


def get_bottom_n_sp500(client, n: int = 10) -> pd.DataFrame:
    """
    Fetch S&P 500 constituents and return the N smallest by market cap.
    These are the companies most at risk of being removed.
    """
    sp500 = get_sp500_tickers()
    print("  Fetching market data for S&P 500 members via FMP quotes (can be slow on free tier)...")
    market_data = get_index_market_data(client, sp500, sector_col="GICS Sector")
    if market_data.empty:
        return pd.DataFrame()

    market_data = market_data.dropna(subset=["marketCap"])
    market_data = market_data.sort_values("marketCap", ascending=True).head(n)
    return market_data.reset_index(drop=True)


def check_removal_candidates(client, cap_threshold: float = REMOVAL_RISK_CAP) -> pd.DataFrame:
    """
    Identify S&P 500 members at risk of removal:
    - Market cap below the given threshold (default $10B)
    """
    sp500 = get_sp500_tickers()
    market_data = get_index_market_data(client, sp500, sector_col="GICS Sector")
    if market_data.empty:
        return pd.DataFrame()

    market_data = market_data.dropna(subset=["marketCap"])
    at_risk = market_data[market_data["marketCap"] < cap_threshold].copy()
    at_risk = at_risk.sort_values("marketCap", ascending=True)
    return at_risk.reset_index(drop=True)


def get_sp500_sector_weights(client) -> pd.DataFrame:
    """
    Compute current S&P 500 sector weights by market cap where available.
    Returns a DataFrame with sector, count, market_cap, and weight_pct columns.
    """
    sp500 = get_sp500_tickers()
    market_data = get_index_market_data(client, sp500, sector_col="GICS Sector")
    if market_data.empty:
        return pd.DataFrame()

    market_data["sector"] = market_data["sector"].map(normalize_sector_name)
    market_data = market_data.dropna(subset=["sector"])
    market_data["marketCap"] = pd.to_numeric(market_data["marketCap"], errors="coerce")
    market_data = market_data.dropna(subset=["marketCap"])

    grouped = (
        market_data.groupby("sector")
        .agg(
            count=("symbol", "count"),
            market_cap=("marketCap", "sum"),
        )
        .reset_index()
    )
    total_cap = grouped["market_cap"].sum()
    grouped["weight_pct"] = (grouped["market_cap"] / total_cap * 100).round(2) if total_cap else 0.0
    return grouped.sort_values("weight_pct", ascending=False).reset_index(drop=True)
