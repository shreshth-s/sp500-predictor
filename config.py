"""
Configuration for the S&P 500 Inclusion Predictor.
All tunable constants, thresholds, and scoring weights live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
FMP_API_KEY = os.getenv("FMP_API_KEY", "REPLACE_WITH_YOUR_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# --- Hard Filter Thresholds ---
MIN_MARKET_CAP = 22.7e9  # $22.7 Billion
REMOVAL_RISK_CAP = 10e9  # $10 Billion removal watch threshold

# --- Soft Scoring Weights (tunable) ---
SECTOR_GAP_WEIGHT = 20.0        # Max points from sector underrepresentation
MIDCAP_400_PREMIUM = 15.0       # Flat bonus for MidCap 400 membership
MARKET_CAP_WEIGHT = 10.0        # Scaled: larger cap = higher score
PROFITABILITY_MARGIN_WEIGHT = 15.0  # Higher TTM margin = higher score
SENTIMENT_WEIGHT = 10.0             # News sentiment via FinBERT

# --- Market Proxy / Universe Build ---
MARKET_PROXY_MIN_CAP = 2e9      # Proxy for broad US market sector weights
FMP_SCREENER_LIMIT = 10000      # Request as much as possible in one call
PROFITABILITY_QUARTERS = 4      # Required GAAP quarters for S&P rule

# --- Liquidity & Float ---
MIN_FALR = 0.75                 # Float-Adjusted Liquidity Ratio minimum
MIN_MONTHLY_VOLUME = 250_000    # Minimum monthly trading volume (shares)

# --- Rate Limiting ---
FMP_REQUESTS_PER_MINUTE = 250   # Paid tier (300 limit, conservative)
FMP_RETRY_ATTEMPTS = 3
FMP_RETRY_BACKOFF = 2.0         # Exponential backoff base (seconds)

# --- Cache ---
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_TTL_HOURS = 24

# --- Output ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
TOP_N_CANDIDATES = 10
