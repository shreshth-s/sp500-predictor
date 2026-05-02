"""
Returns backtester: compute hypothetical portfolio returns from
buying top-N predicted candidates before S&P 500 inclusion events.

Answers the question: "If you bought our top 10 picks before each
announcement, what returns would you have earned?"
"""

from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from fmp_client import FMPClient


def _get_price_at(client, ticker, target_date, lookback_days=7):
    """Get closing price on or near a target date."""
    if isinstance(target_date, str):
        target_date = datetime.fromisoformat(target_date).date()

    date_from = (target_date - timedelta(days=lookback_days)).isoformat()
    date_to = target_date.isoformat()

    try:
        prices = client.historical_price(ticker, date_from, date_to)
        if not prices:
            return None
        # Sort by date descending, take most recent on or before target
        for p in sorted(prices, key=lambda x: x.get("date", ""), reverse=True):
            d = p.get("date", "")
            if d <= date_to:
                close = p.get("close")
                if close and close > 0:
                    return float(close)
        return None
    except Exception:
        return None


def _get_price_after(client, ticker, target_date, days_after, lookforward_days=10):
    """Get closing price N trading days after a target date."""
    if isinstance(target_date, str):
        target_date = datetime.fromisoformat(target_date).date()

    # Approximate calendar days for N trading days
    cal_days = int(days_after * 1.5) + 3
    date_from = (target_date + timedelta(days=1)).isoformat()
    date_to = (target_date + timedelta(days=cal_days + lookforward_days)).isoformat()

    try:
        prices = client.historical_price(ticker, date_from, date_to)
        if not prices:
            return None
        # Sort by date ascending, pick the Nth trading day
        sorted_prices = sorted(prices, key=lambda x: x.get("date", ""))
        idx = min(days_after - 1, len(sorted_prices) - 1)
        if idx < 0:
            return None
        close = sorted_prices[idx].get("close")
        return float(close) if close and close > 0 else None
    except Exception:
        return None


def compute_event_returns(
    client: FMPClient,
    backtest_results: list,
    horizons=(1, 5, 20),
    top_n=10,
    show_progress=True,
):
    """
    For each backtest event, compute returns for:
    1. The actually-added ticker (the "inclusion pop")
    2. Our top-1 predicted ticker
    3. Equal-weight portfolio of all top-N predicted tickers

    Args:
        client: FMPClient instance
        backtest_results: output from run_backtest() — must include
            'top_predictions' field (list of tickers in ranked order)
        horizons: tuple of trading-day horizons (e.g., 1, 5, 20)
        top_n: number of top predictions to include in portfolio

    Returns:
        list of dicts with return data per event
    """
    returns = []

    events_iter = enumerate(backtest_results)
    if show_progress:
        events_iter = tqdm(
            list(events_iter),
            desc="Computing returns",
        )

    for i, event in events_iter:
        t0 = event.get("t0")
        effective_date = event.get("effective_date")
        added_ticker = event.get("added_ticker", "")
        top_preds = event.get("top_predictions", [])

        if not t0:
            continue

        row = {
            "effective_date": effective_date,
            "t0": t0,
            "added_ticker": added_ticker,
            "hit_top_n": event.get("hit_top_n", False),
            "prediction_rank": event.get("prediction_rank"),
        }

        # Get buy price for the actual addition
        actual_buy = _get_price_at(client, added_ticker, t0)
        row["actual_buy_price"] = actual_buy

        # Returns for the actually-added stock at each horizon
        for h in horizons:
            if actual_buy and actual_buy > 0:
                sell = _get_price_after(client, added_ticker, t0, h)
                if sell:
                    row[f"actual_{h}d_return"] = round((sell - actual_buy) / actual_buy * 100, 2)
                else:
                    row[f"actual_{h}d_return"] = None
            else:
                row[f"actual_{h}d_return"] = None

        # Top-1 prediction returns
        top1 = top_preds[0] if top_preds else event.get("top_predicted")
        row["top1_predicted"] = top1
        if top1:
            top1_buy = _get_price_at(client, top1, t0)
            for h in horizons:
                if top1_buy and top1_buy > 0:
                    sell = _get_price_after(client, top1, t0, h)
                    if sell:
                        row[f"top1_{h}d_return"] = round((sell - top1_buy) / top1_buy * 100, 2)
                    else:
                        row[f"top1_{h}d_return"] = None
                else:
                    row[f"top1_{h}d_return"] = None

        # Equal-weight top-N portfolio returns
        portfolio_tickers = top_preds[:top_n] if top_preds else []
        row["portfolio_size"] = len(portfolio_tickers)

        for h in horizons:
            ticker_returns = []
            for ticker in portfolio_tickers:
                buy = _get_price_at(client, ticker, t0)
                if buy and buy > 0:
                    sell = _get_price_after(client, ticker, t0, h)
                    if sell:
                        ticker_returns.append((sell - buy) / buy * 100)

            if ticker_returns:
                row[f"portfolio_{h}d_return"] = round(
                    sum(ticker_returns) / len(ticker_returns), 2
                )
                row[f"portfolio_{h}d_tickers_priced"] = len(ticker_returns)
            else:
                row[f"portfolio_{h}d_return"] = None
                row[f"portfolio_{h}d_tickers_priced"] = 0

        returns.append(row)

    return returns


def summarize_returns(returns_data, horizons=(1, 5, 20)):
    """
    Compute aggregate return statistics across all events.

    Returns a dict with mean, median, win rate for each strategy/horizon.
    """
    df = pd.DataFrame(returns_data)
    if df.empty:
        return {}

    summary = {"total_events": len(df)}

    for strategy in ["actual", "top1", "portfolio"]:
        for h in horizons:
            col = f"{strategy}_{h}d_return"
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue

            key = f"{strategy}_{h}d"
            summary[f"{key}_mean"] = round(float(series.mean()), 2)
            summary[f"{key}_median"] = round(float(series.median()), 2)
            summary[f"{key}_win_rate"] = round(
                float((series > 0).sum() / len(series) * 100), 1
            )
            summary[f"{key}_max"] = round(float(series.max()), 2)
            summary[f"{key}_min"] = round(float(series.min()), 2)
            summary[f"{key}_count"] = int(len(series))

    return summary
