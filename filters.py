"""
Phase 1 hard filters: S&P published eligibility criteria.
Companies that fail these cannot be added regardless of other qualities.
"""

import re

import pandas as pd
from tqdm import tqdm

from config import MIN_FALR, MIN_MARKET_CAP, MIN_MONTHLY_VOLUME, PROFITABILITY_QUARTERS
from data_sources import get_large_cap_universe, get_sp500_tickers

_COMMON_STOCK_TICKER_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")
_MAJOR_EXCHANGES = {
    "NYSE",
    "NASDAQ",
    "AMEX",
    "NYQ",
    "NMS",
    "NGM",
    "NCM",
    "ASE",
}


def apply_market_cap_filter(candidates: pd.DataFrame) -> pd.DataFrame:
    """Filter A (Size): Market cap must exceed the threshold."""
    if "marketCap" not in candidates.columns:
        return candidates
    df = candidates[candidates["marketCap"].notna()].copy()
    df = df[df["marketCap"] >= MIN_MARKET_CAP]
    return df.reset_index(drop=True)


def apply_domicile_filter(candidates: pd.DataFrame) -> pd.DataFrame:
    """Filter B (Domicile): Must be US-headquartered."""
    if "country" not in candidates.columns:
        return candidates
    us_variants = {"United States", "US", "USA"}
    df = candidates[candidates["country"].isin(us_variants)]
    return df.reset_index(drop=True)


def exclude_current_sp500(candidates: pd.DataFrame) -> pd.DataFrame:
    """Remove companies already in the S&P 500."""
    sp500 = get_sp500_tickers()
    sp500_tickers = set(sp500["Symbol"].astype(str).str.upper().tolist())
    return candidates[~candidates["symbol"].isin(sp500_tickers)].reset_index(drop=True)


def apply_data_quality_filter(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Remove likely non-common-stock or inactive artifacts.
    Keeps only tradable common-stock-like rows when metadata is available.
    """
    if candidates.empty:
        return candidates

    df = candidates.copy()

    if "symbol" in df.columns:
        symbols = df["symbol"].astype(str).str.upper().str.strip()
        df = df[symbols.map(lambda s: bool(_COMMON_STOCK_TICKER_RE.match(s)))]

    if "is_actively_trading" in df.columns:
        active = df["is_actively_trading"]
        df = df[active.isna() | (active == True)]  # noqa: E712

    if "is_etf" in df.columns:
        etf = df["is_etf"]
        df = df[etf.isna() | (etf == False)]  # noqa: E712

    if "is_fund" in df.columns:
        fund = df["is_fund"]
        df = df[fund.isna() | (fund == False)]  # noqa: E712

    if "exchange" in df.columns:
        exch = df["exchange"].astype(str).str.upper().str.strip()
        df = df[exch.isin(_MAJOR_EXCHANGES)]

    return df.reset_index(drop=True)


def _extract_recent_quarter_values(income_rows, key, count):
    """Extract most recent numeric values from statement rows."""
    out = []
    if not isinstance(income_rows, list):
        return out

    for row in income_rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
        if len(out) >= count:
            break
    return out


def apply_profitability_filter(candidates: pd.DataFrame, client) -> pd.DataFrame:
    """
    Filter C (Profitability): FMP GAAP net income checks.
      - Most recent quarter net income > 0
      - Trailing twelve months (sum of last 4 quarters) net income > 0
    """
    results = []

    for _, row in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="Checking GAAP profitability",
    ):
        ticker = row["symbol"]
        try:
            income_rows = client.income_statement(
                ticker,
                period="quarter",
                limit=PROFITABILITY_QUARTERS,
            )
        except Exception:
            continue

        net_income_quarters = _extract_recent_quarter_values(
            income_rows, "netIncome", PROFITABILITY_QUARTERS
        )
        if len(net_income_quarters) < PROFITABILITY_QUARTERS:
            continue

        q1_net_income = net_income_quarters[0]
        ttm_net_income = sum(net_income_quarters)
        if q1_net_income <= 0 or ttm_net_income <= 0:
            continue

        revenue_quarters = _extract_recent_quarter_values(
            income_rows, "revenue", PROFITABILITY_QUARTERS
        )
        ttm_revenue = sum(revenue_quarters) if len(revenue_quarters) == PROFITABILITY_QUARTERS else None
        ttm_net_margin = (ttm_net_income / ttm_revenue) if ttm_revenue and ttm_revenue > 0 else None

        result = row.to_dict()
        result["q1_net_income"] = q1_net_income
        result["ttm_net_income"] = ttm_net_income
        result["ttm_revenue"] = ttm_revenue
        result["ttm_net_margin"] = ttm_net_margin
        results.append(result)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).reset_index(drop=True)


def apply_liquidity_filter(candidates: pd.DataFrame, client) -> pd.DataFrame:
    """
    Filter D (Liquidity & Float):
      - Float-Adjusted Liquidity Ratio (FALR) >= 0.75
        FALR = annual dollar volume traded / float-adjusted market cap
      - Minimum monthly trading volume >= 250,000 shares
    """
    results = []

    for _, row in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="Checking liquidity & float",
    ):
        ticker = row["symbol"]
        try:
            float_data = client.shares_float(ticker)
            if not float_data:
                continue

            float_shares = float_data.get("floatShares")
            outstanding = float_data.get("outstandingShares")
            if not float_shares or not outstanding or float_shares <= 0 or outstanding <= 0:
                continue

            # Get average daily volume from profile
            profile = client.company_profile(ticker)
            if not profile:
                continue

            avg_volume = profile.get("averageVolume")
            price = profile.get("price")
            if not avg_volume or not price or avg_volume <= 0 or price <= 0:
                continue

            # Monthly volume check (~21 trading days/month, use avg daily)
            if avg_volume < MIN_MONTHLY_VOLUME:
                continue

            # FALR = annual dollar volume traded / float-adjusted market cap
            # Annual trading days ~252
            annual_dollar_volume = avg_volume * price * 252
            float_adjusted_market_cap = float_shares * price
            falr = annual_dollar_volume / float_adjusted_market_cap

            if falr < MIN_FALR:
                continue

            result = row.to_dict()
            result["float_shares"] = float_shares
            result["avg_daily_volume"] = avg_volume
            result["falr"] = round(falr, 4)
            results.append(result)

        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).reset_index(drop=True)


def get_eligible_universe(client, skip_profitability=False) -> pd.DataFrame:
    """
    Master function: runs all hard filters in sequence.
    Returns a DataFrame of companies eligible for S&P 500 inclusion.
    """
    print("Building candidate universe...")
    df = get_large_cap_universe(client)
    if df.empty:
        print("No market data retrieved.")
        return df
    print(f"  Retrieved data for {len(df)} companies.")

    print("Applying market cap filter (>${:.1f}B)...".format(MIN_MARKET_CAP / 1e9))
    df = apply_market_cap_filter(df)
    print(f"  {len(df)} companies above threshold.")

    print("Applying domicile filter (US only)...")
    df = apply_domicile_filter(df)
    print(f"  {len(df)} US-based companies.")

    print("Applying data quality filter (active common stocks on major exchanges)...")
    df = apply_data_quality_filter(df)
    print(f"  {len(df)} candidates remain after quality checks.")

    print("Excluding current S&P 500 members...")
    df = exclude_current_sp500(df)
    print(f"  {len(df)} candidates remain after exclusion.")

    if skip_profitability:
        print("  Skipping profitability filter (--skip-profitability).")
        df["q1_net_income"] = None
        df["ttm_net_income"] = None
        df["ttm_revenue"] = None
        df["ttm_net_margin"] = None

        print("Checking liquidity & float (FALR >= {}, monthly vol >= {:,})...".format(
            MIN_FALR, MIN_MONTHLY_VOLUME
        ))
        df = apply_liquidity_filter(df, client=client)
        print(f"  {len(df)} candidates pass the liquidity filter.")
        return df

    print("Checking GAAP profitability via FMP...")
    df = apply_profitability_filter(df, client=client)
    print(f"  {len(df)} candidates pass the profitability filter.")

    if df.empty:
        return df

    print("Checking liquidity & float (FALR >= {}, monthly vol >= {:,})...".format(
        MIN_FALR, MIN_MONTHLY_VOLUME
    ))
    df = apply_liquidity_filter(df, client=client)
    print(f"  {len(df)} candidates pass the liquidity filter.")

    return df
