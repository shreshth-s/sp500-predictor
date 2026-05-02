"""
Phase 2 soft scoring: rank eligible candidates by committee-like preferences.
Each scoring function returns a Series indexed by the candidate DataFrame index.
All scores are in [0, weight] range. Total score is the sum.
"""

import pandas as pd

from config import (
    MARKET_CAP_WEIGHT,
    MIDCAP_400_PREMIUM,
    PROFITABILITY_MARGIN_WEIGHT,
    SECTOR_GAP_WEIGHT,
    SENTIMENT_WEIGHT,
)
from data_sources import (
    get_sp400_tickers,
    get_sp500_tickers,
    get_sp500_sector_market_weights,
    get_us_market_proxy_sector_weights,
    normalize_sector_name,
)


def compute_sector_gap_scores(candidates: pd.DataFrame, client) -> pd.Series:
    """
    Compare S&P 500 sector weights vs a broad US market proxy.
    Underrepresented sectors in the S&P 500 receive higher scores.
    """
    if "sector" not in candidates.columns or candidates.empty:
        return pd.Series(0.0, index=candidates.index)

    sp500_weights = get_sp500_sector_market_weights(client)
    if sp500_weights.empty:
        # Fallback mode: count-based weights on both sides to stay apples-to-apples.
        sp500 = get_sp500_tickers()
        sp500_weights = (
            sp500["GICS Sector"]
            .map(normalize_sector_name)
            .value_counts(normalize=True)
        )
        market_weights = (
            candidates["sector"]
            .map(normalize_sector_name)
            .value_counts(normalize=True)
        )
    else:
        market_weights = get_us_market_proxy_sector_weights(client)
        if market_weights.empty:
            fallback = candidates.copy()
            fallback["sector"] = fallback["sector"].map(normalize_sector_name)
            fallback["marketCap"] = pd.to_numeric(fallback.get("marketCap"), errors="coerce")
            fallback = fallback.dropna(subset=["sector", "marketCap"])
            fallback = fallback[fallback["marketCap"] > 0]
            if not fallback.empty:
                market_weights = fallback.groupby("sector")["marketCap"].sum()
                market_weights = market_weights / market_weights.sum()
            else:
                market_weights = pd.Series(dtype=float)

    all_sectors = set(sp500_weights.index) | set(market_weights.index)
    positive_gaps = {}
    for sector in all_sectors:
        market_weight = float(market_weights.get(sector, 0.0))
        sp500_weight = float(sp500_weights.get(sector, 0.0))
        positive_gaps[sector] = max(market_weight - sp500_weight, 0.0)

    max_gap = max(positive_gaps.values()) if positive_gaps else 0.0
    if max_gap <= 0:
        return pd.Series(0.0, index=candidates.index)

    candidate_sector = candidates["sector"].map(normalize_sector_name)
    scores = candidate_sector.map(
        lambda s: (positive_gaps.get(s, 0.0) / max_gap) * SECTOR_GAP_WEIGHT
    )
    return scores.fillna(0.0)


def compute_midcap_premium(candidates: pd.DataFrame) -> pd.Series:
    """
    Flat bonus for companies currently in the S&P MidCap 400.
    These are historically favored for promotion to the S&P 500.
    """
    try:
        sp400 = get_sp400_tickers()
        sp400_tickers = set(sp400["Symbol"].astype(str).str.upper().tolist())
        return candidates["symbol"].map(
            lambda t: MIDCAP_400_PREMIUM if str(t).upper() in sp400_tickers else 0.0
        )
    except Exception:
        print("  Warning: Could not fetch S&P MidCap 400 data. Skipping premium.")
        return pd.Series(0.0, index=candidates.index)


def compute_market_cap_score(candidates: pd.DataFrame) -> pd.Series:
    """Min-max normalized market cap score. Larger cap => higher score."""
    if "marketCap" not in candidates.columns or candidates.empty:
        return pd.Series(0.0, index=candidates.index)

    caps = pd.to_numeric(candidates["marketCap"], errors="coerce")
    min_cap = caps.min()
    max_cap = caps.max()

    if pd.isna(min_cap) or pd.isna(max_cap):
        return pd.Series(0.0, index=candidates.index)
    if max_cap == min_cap:
        return pd.Series(MARKET_CAP_WEIGHT / 2, index=candidates.index)

    normalized = (caps - min_cap) / (max_cap - min_cap)
    return normalized.fillna(0.0) * MARKET_CAP_WEIGHT


def compute_profitability_score(candidates: pd.DataFrame) -> pd.Series:
    """
    Profitability quality score.
    Preferred input is TTM net margin; falls back to TTM net income.
    """
    if candidates.empty:
        return pd.Series(0.0, index=candidates.index)

    if "ttm_net_margin" in candidates.columns and not candidates["ttm_net_margin"].isna().all():
        series = pd.to_numeric(candidates["ttm_net_margin"], errors="coerce")
    elif "ttm_net_income" in candidates.columns and not candidates["ttm_net_income"].isna().all():
        series = pd.to_numeric(candidates["ttm_net_income"], errors="coerce")
    else:
        return pd.Series(0.0, index=candidates.index)

    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(0.0, index=candidates.index)
    if max_val == min_val:
        return pd.Series(PROFITABILITY_MARGIN_WEIGHT / 2, index=candidates.index)

    normalized = (series - min_val) / (max_val - min_val)
    return normalized.fillna(0.0) * PROFITABILITY_MARGIN_WEIGHT


def score_candidates(candidates: pd.DataFrame, client, use_sentiment=True) -> pd.DataFrame:
    """
    Master scoring function. Adds individual score columns and a total_score.
    Returns candidates sorted by total_score descending.
    """
    if candidates.empty:
        return candidates

    print("Scoring candidates...")
    candidates = candidates.copy()

    print("  Computing sector gap scores...")
    candidates["sector_gap_score"] = compute_sector_gap_scores(candidates, client=client)

    print("  Computing MidCap 400 premium...")
    candidates["midcap_premium"] = compute_midcap_premium(candidates)

    print("  Computing market cap score...")
    candidates["market_cap_score"] = compute_market_cap_score(candidates)

    print("  Computing profitability score...")
    candidates["profitability_score"] = compute_profitability_score(candidates)

    if use_sentiment:
        print("  Computing news sentiment score...")
        from sentiment import compute_sentiment_scores
        candidates["sentiment_score"] = compute_sentiment_scores(candidates)

    score_cols = [
        "sector_gap_score",
        "midcap_premium",
        "market_cap_score",
        "profitability_score",
    ]
    if use_sentiment and "sentiment_score" in candidates.columns:
        score_cols.append("sentiment_score")
    candidates["total_score"] = candidates[score_cols].sum(axis=1)

    return candidates.sort_values("total_score", ascending=False).reset_index(drop=True)
