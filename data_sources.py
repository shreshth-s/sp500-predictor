"""
Data sources for index constituents and market data.
Uses Wikipedia for constituent lists and FMP for market/fundamental data.
"""

import json
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    CACHE_DIR,
    CACHE_TTL_HOURS,
    FMP_SCREENER_LIMIT,
    MARKET_PROXY_MIN_CAP,
    MIN_MARKET_CAP,
)

_CACHE_DIR = Path(CACHE_DIR)
_CACHE_DIR.mkdir(exist_ok=True)

WIKI_HEADERS = {"User-Agent": "SP500Predictor/1.0"}

SECTOR_ALIASES = {
    "Basic Materials": "Materials",
    "Communication Services": "Communication Services",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Energy": "Energy",
    "Financial Services": "Financials",
    "Financials": "Financials",
    "Health Care": "Health Care",
    "Healthcare": "Health Care",
    "Industrials": "Industrials",
    "Information Technology": "Information Technology",
    "Materials": "Materials",
    "Real Estate": "Real Estate",
    "Technology": "Information Technology",
    "Telecommunications Services": "Communication Services",
    "Utilities": "Utilities",
}


def normalize_sector_name(value):
    """Normalize sector taxonomy differences between data sources."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return SECTOR_ALIASES.get(text, text)


def _read_cache(name):
    """Read a named cache file if fresh enough."""
    path = _CACHE_DIR / f"{name}.json"
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > CACHE_TTL_HOURS:
        return None
    with open(path, "r") as f:
        return json.load(f)


def _write_cache(name, data):
    """Write data to a named cache file."""
    path = _CACHE_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f)


def _fetch_wikipedia_constituents(cache_name, url):
    """Fetch one constituents table from Wikipedia with caching."""
    cached = _read_cache(cache_name)
    if cached is not None:
        return pd.DataFrame(cached)

    response = requests.get(url, headers=WIKI_HEADERS, timeout=20)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    df = tables[0]
    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)

    _write_cache(cache_name, df.to_dict(orient="records"))
    return df


def get_sp500_tickers() -> pd.DataFrame:
    """Fetch current S&P 500 constituents from Wikipedia."""
    return _fetch_wikipedia_constituents(
        "sp500_constituents",
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    )


def get_sp400_tickers() -> pd.DataFrame:
    """Fetch current S&P MidCap 400 constituents from Wikipedia."""
    return _fetch_wikipedia_constituents(
        "sp400_constituents",
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    )


def get_sp600_tickers() -> pd.DataFrame:
    """Fetch current S&P SmallCap 600 constituents from Wikipedia."""
    return _fetch_wikipedia_constituents(
        "sp600_constituents",
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    )


def _normalize_screener_df(rows) -> pd.DataFrame:
    """Normalize FMP screener rows into a consistent schema."""
    if not rows:
        return pd.DataFrame()

    normalized = []
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        market_cap = row.get("marketCap")
        try:
            market_cap = float(market_cap) if market_cap is not None else None
        except (TypeError, ValueError):
            market_cap = None
        normalized.append(
            {
                "symbol": symbol,
                "companyName": row.get("companyName") or row.get("company") or symbol,
                "marketCap": market_cap,
                "sector": normalize_sector_name(row.get("sector")),
                "industry": row.get("industry"),
                "country": row.get("country"),
                "exchange": row.get("exchangeShortName") or row.get("exchange"),
                "is_etf": row.get("isEtf"),
                "is_fund": row.get("isFund"),
                "is_adr": row.get("isAdr"),
                "is_actively_trading": row.get("isActivelyTrading"),
            }
        )
    return pd.DataFrame(normalized)


def _normalize_profile_df(rows) -> pd.DataFrame:
    """Normalize FMP profile rows into a consistent schema."""
    if not rows:
        return pd.DataFrame()

    normalized = []
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        market_cap = row.get("mktCap", row.get("marketCap"))
        try:
            market_cap = float(market_cap) if market_cap is not None else None
        except (TypeError, ValueError):
            market_cap = None
        normalized.append(
            {
                "symbol": symbol,
                "companyName": row.get("companyName") or symbol,
                "marketCap": market_cap,
                "sector": normalize_sector_name(row.get("sector")),
                "industry": row.get("industry"),
                "country": row.get("country"),
                "exchange": row.get("exchangeShortName") or row.get("exchange"),
                "is_etf": row.get("isEtf"),
                "is_fund": row.get("isFund"),
                "is_adr": row.get("isAdr"),
                "is_actively_trading": row.get("isActivelyTrading"),
            }
        )
    return pd.DataFrame(normalized)


def _looks_like_restricted_endpoint(error: Exception) -> bool:
    """Detect plan restriction failures from FMP response text."""
    text = str(error).lower()
    return "restricted endpoint" in text or "not available under your current subscription" in text


def _fetch_screener_rows(
    client,
    market_cap_more_than,
    country="US",
    page_size=1000,
    max_pages=10,
):
    """
    Fetch stock screener rows with pagination and symbol-level deduping.
    Stops early when a page adds no new symbols or final page is shorter.
    """
    out = []
    seen = set()

    for page in range(max_pages):
        rows = client.stock_screener(
            market_cap_more_than=market_cap_more_than,
            country=country,
            limit=page_size,
            page=page,
            is_etf=False,
            is_fund=False,
        )
        if not isinstance(rows, list) or not rows:
            break

        new_count = 0
        for row in rows:
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            out.append(row)
            new_count += 1

        if len(rows) < page_size or new_count == 0:
            break

    return out


def get_large_cap_universe(client) -> pd.DataFrame:
    """
    Build the baseline candidate universe using FMP stock screener:
    US companies above S&P published market cap threshold.
    """
    cache_name = f"fmp_large_cap_universe_{int(MIN_MARKET_CAP)}"
    cached = _read_cache(cache_name)
    if cached is not None:
        return pd.DataFrame(cached)

    try:
        rows = _fetch_screener_rows(
            client,
            market_cap_more_than=MIN_MARKET_CAP,
            country="US",
            page_size=min(FMP_SCREENER_LIMIT, 1000),
            max_pages=5,
        )
        df = _normalize_screener_df(rows)
        if not df.empty:
            df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
            _write_cache(cache_name, df.to_dict(orient="records"))
            return df
    except RuntimeError as e:
        if not _looks_like_restricted_endpoint(e):
            raise

    # Fallback for plans where company-screener is restricted:
    # build a candidate universe from S&P MidCap 400 profiles.
    print("  company-screener is restricted on this plan; falling back to S&P 400 profiles.")
    sp400 = get_sp400_tickers()
    tickers = (
        sp400["Symbol"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )

    profile_rows = []
    for ticker in tqdm(tickers, desc="Fetching profiles"):
        try:
            profile = client.company_profile(ticker)
        except RuntimeError:
            continue
        if profile:
            profile_rows.append(profile)

    df = _normalize_profile_df(profile_rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    _write_cache(cache_name, df.to_dict(orient="records"))
    return df


def get_us_market_proxy_sector_weights(client, min_market_cap=MARKET_PROXY_MIN_CAP) -> pd.Series:
    """
    Approximate broad US market sector weights from FMP screener output.
    Returns a sector -> weight Series where weights sum to 1.
    """
    cache_name = f"fmp_market_proxy_weights_{int(min_market_cap)}"
    cached = _read_cache(cache_name)
    if cached is not None:
        return pd.Series(cached)

    try:
        rows = _fetch_screener_rows(
            client,
            market_cap_more_than=min_market_cap,
            country="US",
            page_size=min(FMP_SCREENER_LIMIT, 1000),
            max_pages=15,
        )
    except RuntimeError as e:
        if _looks_like_restricted_endpoint(e):
            return pd.Series(dtype=float)
        raise

    df = _normalize_screener_df(rows)
    if df.empty:
        return pd.Series(dtype=float)

    df = df.dropna(subset=["sector", "marketCap"]).copy()
    df = df[df["marketCap"] > 0]
    if df.empty:
        return pd.Series(dtype=float)

    weights = df.groupby("sector")["marketCap"].sum()
    weights = weights / weights.sum()
    _write_cache(cache_name, weights.to_dict())
    return weights


def get_sp500_sector_market_weights(client) -> pd.Series:
    """
    Compute S&P 500 sector weights by market cap.
    Returns a sector -> weight Series where weights sum to 1.
    """
    cache_name = "sp500_sector_market_weights"
    cached = _read_cache(cache_name)
    if cached is not None:
        return pd.Series(cached)

    if hasattr(client, "can_batch_quote") and not client.can_batch_quote():
        return pd.Series(dtype=float)

    sp500 = get_sp500_tickers()
    quotes = get_index_market_data(client, sp500, sector_col="GICS Sector")
    if quotes.empty:
        return pd.Series(dtype=float)

    quotes = quotes.dropna(subset=["sector", "marketCap"]).copy()
    quotes = quotes[quotes["marketCap"] > 0]
    if quotes.empty:
        return pd.Series(dtype=float)

    weights = quotes.groupby("sector")["marketCap"].sum()
    weights = weights / weights.sum()
    _write_cache(cache_name, weights.to_dict())
    return weights


def get_index_market_data(client, constituents: pd.DataFrame, sector_col=None) -> pd.DataFrame:
    """
    Fetch current market-cap data for an index constituent list using FMP quote batch.
    """
    if constituents.empty or "Symbol" not in constituents.columns:
        return pd.DataFrame()

    tickers = (
        constituents["Symbol"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .str.upper()
        .unique()
        .tolist()
    )
    if not tickers:
        return pd.DataFrame()

    quotes = client.quote_batch(tickers)
    if not quotes:
        return pd.DataFrame()

    rows = []
    for quote in quotes:
        symbol = str(quote.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        rows.append(
            {
                "symbol": symbol,
                "companyName": quote.get("name") or quote.get("companyName") or symbol,
                "marketCap": quote.get("marketCap"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if sector_col and sector_col in constituents.columns:
        sector_map = {
            str(sym).strip().replace(".", "-").upper(): normalize_sector_name(sec)
            for sym, sec in zip(constituents["Symbol"], constituents[sector_col])
        }
        df["sector"] = df["symbol"].map(sector_map)

    df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
    return df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
