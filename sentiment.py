"""
News sentiment scoring using FinBERT.
Fetches recent headlines via Google News RSS and scores them
with the ProsusAI/finbert model.
"""

import warnings
from xml.etree import ElementTree

import pandas as pd
import requests
from tqdm import tqdm

from config import SENTIMENT_WEIGHT

# Lazy-loaded singleton so the model only downloads/loads once per session
_finbert_pipeline = None


def _get_finbert():
    """Load FinBERT pipeline on first use."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import pipeline
            _finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1,  # CPU
                top_k=None,
            )
    return _finbert_pipeline


def fetch_headlines(ticker, company_name=None, limit=15):
    """
    Fetch recent news headlines for a ticker from Google News RSS.
    Returns a list of headline strings.
    """
    query = f"{ticker} stock"
    if company_name:
        # Use first word of company name to improve relevance
        short_name = company_name.split()[0] if company_name else ""
        if short_name and short_name.upper() != ticker:
            query = f"{short_name} {ticker} stock"

    url = (
        f"https://news.google.com/rss/search"
        f"?q={query}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        items = root.findall(".//item")
        headlines = []
        for item in items[:limit]:
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                headlines.append(title_el.text.strip())
        return headlines
    except Exception:
        return []


def score_headlines(headlines):
    """
    Run FinBERT on a list of headlines.
    Returns a float in [-1, 1]: -1 = very negative, +1 = very positive.
    """
    if not headlines:
        return 0.0

    nlp = _get_finbert()
    # FinBERT has a 512-token limit; headlines are short so batch is fine
    results = nlp(headlines, batch_size=16, truncation=True)

    total = 0.0
    for result in results:
        # result is a list of dicts with label/score for each class
        scores_by_label = {r["label"]: r["score"] for r in result}
        pos = scores_by_label.get("positive", 0.0)
        neg = scores_by_label.get("negative", 0.0)
        # net sentiment: positive - negative (neutral contributes 0)
        total += pos - neg

    return total / len(results)


def compute_sentiment_scores(candidates: pd.DataFrame) -> pd.Series:
    """
    Compute a sentiment score for each candidate ticker.
    Returns a Series in [0, SENTIMENT_WEIGHT] range.
    """
    if candidates.empty:
        return pd.Series(0.0, index=candidates.index)

    raw_scores = []
    for _, row in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="  Fetching news & scoring sentiment",
    ):
        ticker = row.get("symbol", "")
        name = row.get("companyName", "")
        headlines = fetch_headlines(ticker, company_name=name)
        score = score_headlines(headlines)
        raw_scores.append(score)

    raw = pd.Series(raw_scores, index=candidates.index)

    # Normalize from [-1, 1] to [0, SENTIMENT_WEIGHT]
    # Shift to [0, 2] then scale to [0, weight]
    normalized = ((raw + 1) / 2) * SENTIMENT_WEIGHT
    return normalized.fillna(SENTIMENT_WEIGHT / 2)
