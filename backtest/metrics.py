"""
Backtest evaluation metrics and reporting.
"""

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def compute_metrics(results, top_n=10):
    """
    Compute accuracy metrics from backtest results.

    Returns a dict with:
    - total_events: number of events evaluated
    - events_with_rank: events where the added ticker was in our universe
    - hit_rate_top_n: % of events where actual addition was in our top N
    - hit_rate_top_1: % where we predicted the exact addition
    - mean_rank: average rank of the actual addition (when found)
    - median_rank: median rank
    - by_confidence: metrics broken down by confidence tier
    """
    df = pd.DataFrame(results)
    if df.empty:
        return {"total_events": 0}

    total = len(df)
    has_rank = df["prediction_rank"].notna()
    ranked_df = df[has_rank]

    metrics = {
        "total_events": total,
        "events_with_rank": int(has_rank.sum()),
        "events_not_found": int((~has_rank).sum()),
    }

    if ranked_df.empty:
        metrics["hit_rate_top_n"] = 0.0
        metrics["hit_rate_top_1"] = 0.0
        metrics["mean_rank"] = None
        metrics["median_rank"] = None
    else:
        ranks = ranked_df["prediction_rank"].astype(float)
        hits_top_n = (ranks <= top_n).sum()
        hits_top_1 = (ranks == 1).sum()

        metrics["hit_rate_top_n"] = round(hits_top_n / total * 100, 1)
        metrics["hit_rate_top_1"] = round(hits_top_1 / total * 100, 1)
        metrics["hit_rate_top_n_of_ranked"] = round(hits_top_n / len(ranked_df) * 100, 1)
        metrics["mean_rank"] = round(float(ranks.mean()), 1)
        metrics["median_rank"] = round(float(ranks.median()), 1)
        metrics["top_n"] = top_n

    # Breakdown by confidence tier
    by_confidence = {}
    for tier in ["A", "B", "C"]:
        tier_mask = df["event_confidence"] == tier
        if not tier_mask.any():
            continue
        tier_df = df[tier_mask]
        tier_ranked = tier_df[tier_df["prediction_rank"].notna()]
        tier_total = len(tier_df)

        tier_metrics = {"total": tier_total, "ranked": len(tier_ranked)}
        if tier_ranked.empty:
            tier_metrics["hit_rate_top_n"] = 0.0
            tier_metrics["mean_rank"] = None
        else:
            tier_ranks = tier_ranked["prediction_rank"].astype(float)
            tier_metrics["hit_rate_top_n"] = round(
                (tier_ranks <= top_n).sum() / tier_total * 100, 1
            )
            tier_metrics["mean_rank"] = round(float(tier_ranks.mean()), 1)

        by_confidence[tier] = tier_metrics

    metrics["by_confidence"] = by_confidence

    # Breakdown by data confidence
    by_data_conf = {}
    if "data_confidence" in df.columns:
        for tier in ["A", "B", "C"]:
            tier_mask = df["data_confidence"] == tier
            if tier_mask.any():
                by_data_conf[tier] = int(tier_mask.sum())
    metrics["by_data_confidence"] = by_data_conf

    return metrics


def print_report(results, metrics, top_n=10):
    """Print a formatted backtest report."""
    df = pd.DataFrame(results)

    console.print()
    console.print("=" * 60)
    console.print("  BACKTEST REPORT", style="bold")
    console.print("=" * 60)
    console.print()

    # Summary metrics
    summary = Table(title="Summary Metrics", show_lines=True)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")

    summary.add_row("Total Events", str(metrics["total_events"]))
    summary.add_row("Events With Prediction", str(metrics["events_with_rank"]))
    summary.add_row("Events Not Found", str(metrics["events_not_found"]))

    if metrics.get("hit_rate_top_n") is not None:
        summary.add_row(f"Hit Rate (Top {top_n})", f"{metrics['hit_rate_top_n']}%")
        summary.add_row("Hit Rate (Top 1)", f"{metrics['hit_rate_top_1']}%")
        if metrics.get("hit_rate_top_n_of_ranked") is not None:
            summary.add_row(
                f"Hit Rate Top {top_n} (of ranked)",
                f"{metrics['hit_rate_top_n_of_ranked']}%",
            )
        if metrics.get("mean_rank") is not None:
            summary.add_row("Mean Rank", str(metrics["mean_rank"]))
            summary.add_row("Median Rank", str(metrics["median_rank"]))

    console.print(summary)

    # Confidence tier breakdown
    by_conf = metrics.get("by_confidence", {})
    if by_conf:
        conf_table = Table(title="By Event Confidence Tier", show_lines=True)
        conf_table.add_column("Tier", style="bold")
        conf_table.add_column("Events", justify="right")
        conf_table.add_column("Ranked", justify="right")
        conf_table.add_column(f"Hit Rate (Top {top_n})", justify="right")
        conf_table.add_column("Mean Rank", justify="right")

        for tier in ["A", "B", "C"]:
            if tier not in by_conf:
                continue
            t = by_conf[tier]
            conf_table.add_row(
                tier,
                str(t["total"]),
                str(t["ranked"]),
                f"{t['hit_rate_top_n']}%",
                str(t["mean_rank"] or "-"),
            )
        console.print(conf_table)

    # Data confidence breakdown
    by_data = metrics.get("by_data_confidence", {})
    if by_data:
        console.print(f"\nData Confidence: " + ", ".join(
            f"Tier {k}: {v}" for k, v in sorted(by_data.items())
        ))

    # Event-level detail table
    if not df.empty:
        detail = Table(title="Event Details", show_lines=True)
        detail.add_column("#", style="dim", width=4)
        detail.add_column("Date", width=12)
        detail.add_column("Added", style="bold", width=8)
        detail.add_column("Rank", justify="right", width=6)
        detail.add_column(f"Top {top_n}?", justify="center", width=8)
        detail.add_column("Top Predicted", width=12)
        detail.add_column("Conf", width=5)
        detail.add_column("Universe", justify="right", width=8)

        for i, row in df.iterrows():
            rank = row.get("prediction_rank")
            rank_str = str(int(rank)) if pd.notna(rank) else "-"
            hit = row.get("hit_top_n", False)
            hit_str = "YES" if hit else "no"
            hit_style = "bold green" if hit else "dim"

            detail.add_row(
                str(i + 1),
                str(row.get("effective_date", ""))[:10],
                str(row.get("added_ticker", "")),
                rank_str,
                f"[{hit_style}]{hit_str}[/{hit_style}]",
                str(row.get("top_predicted", "-")),
                str(row.get("event_confidence", "")),
                str(row.get("candidates_ranked", "-")),
            )

        console.print(detail)
