"""
Walk-forward backtest runner.
Replays historical S&P 500 events through the prediction pipeline
using point-in-time data.
"""

import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from config import OUTPUT_DIR, TOP_N_CANDIDATES
from data_sources import get_sp400_tickers, get_sp500_tickers
from fmp_client import FMPClient
from scoring import score_candidates
from backtest.events import load_events
from backtest.snapshot import AsOfSnapshot


def _get_candidate_tickers_for_era(t0_date, all_events_df):
    """
    Build a set of tickers to screen for a historical date.
    Uses current S&P 400 + any tickers that appear in the events table.
    This isn't perfect (survivorship issue) but is the best we can do
    without a full historical ticker database.
    """
    tickers = set()

    # Current S&P 400 members (these are the prime promotion candidates)
    try:
        sp400 = get_sp400_tickers()
        tickers.update(sp400["Symbol"].astype(str).str.upper().tolist())
    except Exception:
        pass

    # Add tickers from events table (companies that were actually added/removed)
    # This helps catch companies that were in the S&P 400 at the time
    # but may have since moved out
    for _, row in all_events_df.iterrows():
        added = str(row.get("added_ticker", "")).strip().upper()
        removed = str(row.get("removed_ticker", "")).strip().upper()
        if added and added != "NAN":
            tickers.add(added)
        if removed and removed != "NAN":
            tickers.add(removed)

    return sorted(tickers)


def run_backtest(
    client: FMPClient,
    min_year=2020,
    max_events=None,
    top_n=TOP_N_CANDIDATES,
    skip_liquidity=False,
    show_progress=True,
):
    """
    Run a walk-forward backtest over historical S&P 500 events.

    For each event:
    1. Build a PIT snapshot at t0
    2. Filter and score candidates using only data available at t0
    3. Record whether the actual addition was predicted

    Returns a list of result dicts, one per event.
    """
    full_events = load_events(min_year=min_year)
    if full_events.empty:
        print("No events found.")
        return []

    backtest_events = full_events.copy()
    if max_events:
        backtest_events = backtest_events.tail(max_events).reset_index(drop=True)

    current_sp500 = get_sp500_tickers()

    results = []
    events_iter = backtest_events.iterrows()
    if show_progress:
        events_iter = tqdm(
            list(events_iter),
            desc="Backtesting events",
        )

    for _, event in events_iter:
        t0 = event["t0"]
        added_ticker = str(event["added_ticker"]).strip().upper()
        effective_date = event["effective_date"]

        if show_progress:
            events_iter.set_postfix_str(f"{effective_date} +{added_ticker}")

        if not added_ticker or added_ticker == "NAN":
            continue

        candidate_tickers = _get_candidate_tickers_for_era(t0, full_events)
        snapshot = AsOfSnapshot(
            client=client,
            t0=t0,
            current_sp500_df=current_sp500,
            all_events_df=full_events,
        )

        result = {
            "effective_date": effective_date,
            "t0": t0,
            "added_ticker": added_ticker,
            "added_name": event.get("added_name", ""),
            "removed_ticker": str(event.get("removed_ticker", "")),
            "reason": event.get("reason", ""),
            "event_confidence": event.get("confidence", "B"),
        }

        try:
            # Build universe as-of t0
            candidates = snapshot.build_candidate_universe(candidate_tickers)
            result["universe_size"] = len(candidates)

            if candidates.empty:
                result["prediction_rank"] = None
                result["hit_top_n"] = False
                result["data_confidence"] = "C"
                results.append(result)
                continue

            # Apply profitability filter
            candidates = snapshot.apply_profitability(candidates)
            result["post_profitability"] = len(candidates)

            if candidates.empty:
                result["prediction_rank"] = None
                result["hit_top_n"] = False
                result["data_confidence"] = "C"
                results.append(result)
                continue

            # Apply liquidity filter (optional — slower)
            if not skip_liquidity:
                candidates = snapshot.apply_liquidity(candidates)
                result["post_liquidity"] = len(candidates)
            else:
                result["post_liquidity"] = len(candidates)

            if candidates.empty:
                result["prediction_rank"] = None
                result["hit_top_n"] = False
                result["data_confidence"] = "C"
                results.append(result)
                continue

            # Score candidates using the same scoring logic as live
            ranked = score_candidates(candidates, client=client, use_sentiment=False)

            # Check if the actual addition is in our ranked list
            match = ranked[ranked["symbol"].str.upper() == added_ticker]
            if not match.empty:
                rank = int(match.index[0]) + 1
                result["prediction_rank"] = rank
                result["hit_top_n"] = rank <= top_n
                result["predicted_score"] = float(match.iloc[0].get("total_score", 0))
            else:
                result["prediction_rank"] = None
                result["hit_top_n"] = False

            result["data_confidence"] = snapshot.get_confidence_tier(ranked)
            result["top_predicted"] = ranked.iloc[0]["symbol"] if not ranked.empty else None
            result["candidates_ranked"] = len(ranked)

        except Exception as e:
            result["prediction_rank"] = None
            result["hit_top_n"] = False
            result["data_confidence"] = "C"
            result["error"] = str(e)[:200]

        results.append(result)

    return results


def export_backtest_results(results, label="backtest"):
    """Save backtest results to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.csv"
    path = os.path.join(OUTPUT_DIR, filename)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"\nBacktest results exported to: {path}")
    return path
