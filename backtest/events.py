"""
Event loader: parse historical S&P 500 additions/removals from Wikipedia.
Each event represents a committee decision that we can replay.
"""

import json
import re
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from config import CACHE_DIR, CACHE_TTL_HOURS

_CACHE_DIR = Path(CACHE_DIR)
_CACHE_DIR.mkdir(exist_ok=True)

WIKI_HEADERS = {"User-Agent": "SP500Predictor/1.0"}
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _parse_date(text):
    """Parse a Wikipedia date string like 'March 23, 2026' into a date."""
    text = text.strip()
    # Remove citation references like [6]
    text = re.sub(r"\[.*?\]", "", text).strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _trading_days_before(date, n=1):
    """
    Approximate n trading days before a given date.
    Skips weekends. Does not account for holidays (acceptable for MVP).
    """
    d = date
    skipped = 0
    while skipped < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            skipped += 1
    return d


def load_events(min_year=2016) -> pd.DataFrame:
    """
    Load S&P 500 addition/removal events from Wikipedia.

    Returns DataFrame with columns:
        effective_date, t0, added_ticker, added_name,
        removed_ticker, removed_name, reason, confidence

    min_year: only return events from this year onward
              (FMP income data goes back ~10 years)
    """
    cache_path = _CACHE_DIR / "sp500_changes_raw.json"
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < CACHE_TTL_HOURS * 7:  # Cache changes table for a week
            with open(cache_path, "r") as f:
                rows = json.load(f)
            return _build_event_df(rows, min_year)

    resp = requests.get(WIKI_URL, headers=WIKI_HEADERS, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))

    # The changes table is the second table (index 1)
    if len(tables) < 2:
        raise RuntimeError("Could not find S&P 500 changes table on Wikipedia")

    changes = tables[1]
    # Flatten multi-level columns
    changes.columns = [
        "effective_date", "added_ticker", "added_name",
        "removed_ticker", "removed_name", "reason",
    ]

    rows = changes.to_dict(orient="records")
    with open(cache_path, "w") as f:
        json.dump(rows, f)

    return _build_event_df(rows, min_year)


def _build_event_df(rows, min_year):
    """Convert raw Wikipedia rows into structured event DataFrame."""
    events = []
    for row in rows:
        date = _parse_date(str(row.get("effective_date", "")))
        if date is None or date.year < min_year:
            continue

        added = str(row.get("added_ticker", "")).strip()
        removed = str(row.get("removed_ticker", "")).strip()
        if not added:
            continue

        # Clean ticker (Wikipedia uses dots, FMP uses dashes)
        added = added.replace(".", "-")
        removed = removed.replace(".", "-") if removed else ""

        # t0 = 1 trading day before the effective date
        t0 = _trading_days_before(date, n=1)

        # Determine confidence tier based on reason
        reason = str(row.get("reason", ""))
        confidence = _classify_confidence(reason)

        events.append({
            "effective_date": date.isoformat(),
            "t0": t0.isoformat(),
            "added_ticker": added,
            "added_name": str(row.get("added_name", "")),
            "removed_ticker": removed,
            "removed_name": str(row.get("removed_name", "")),
            "reason": re.sub(r"\[.*?\]", "", reason).strip(),
            "confidence": confidence,
        })

    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values("effective_date", ascending=True).reset_index(drop=True)
    return df


def _classify_confidence(reason):
    """
    Assign confidence tier based on how predictable the event type is.
    A = highly predictable (M&A, market cap changes)
    B = moderately predictable (restructuring, spin-off)
    C = hard to predict (committee discretion, IPO fast-track)
    """
    reason_lower = reason.lower()
    if any(kw in reason_lower for kw in ["acqui", "merger", "bought", "purchased"]):
        return "A"
    if any(kw in reason_lower for kw in ["market cap", "capitalization"]):
        return "A"
    if any(kw in reason_lower for kw in ["spin", "split", "restructur"]):
        return "B"
    if any(kw in reason_lower for kw in ["ipo", "spac", "direct listing"]):
        return "C"
    return "B"


def reconstruct_sp500_at(target_date, current_members_df, changes_df):
    """
    Reconstruct S&P 500 membership at a historical date by walking
    the changes table backward from the current list.

    current_members_df: today's S&P 500 (from Wikipedia, with 'Symbol' column)
    changes_df: output of load_events() (with effective_date, added_ticker, removed_ticker)

    Returns a set of ticker symbols that were in the S&P 500 at target_date.
    """
    if isinstance(target_date, str):
        target_date = datetime.fromisoformat(target_date).date()

    members = set(current_members_df["Symbol"].astype(str).str.upper().tolist())

    # Sort changes newest-first so we can undo them
    changes = changes_df.copy()
    changes["_date"] = pd.to_datetime(changes["effective_date"]).dt.date
    changes = changes.sort_values("_date", ascending=False)

    for _, row in changes.iterrows():
        event_date = row["_date"]
        if event_date <= target_date:
            break

        added = str(row.get("added_ticker", "")).strip().upper()
        removed = str(row.get("removed_ticker", "")).strip().upper()

        # Undo this change: remove the added ticker, re-add the removed one
        if added:
            members.discard(added)
        if removed:
            members.add(removed)

    return members
