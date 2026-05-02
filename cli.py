"""
CLI entry point for the S&P 500 Inclusion Predictor.

Usage:
    python cli.py run [--top N] [--export] [--skip-profitability] [--no-cache]
    python cli.py watchlist
    python cli.py simulate TICKER [--top N]
    python cli.py score TICKER [--skip-profitability] [--no-cache]
    python cli.py backtest [--min-year YEAR] [--max-events N] [--top N] [--skip-liquidity] [--export]
    python cli.py clear-cache
"""

import argparse
import sys

from rich.console import Console
from rich.table import Table

from config import FMP_API_KEY, TOP_N_CANDIDATES
from fmp_client import FMPClient
from pipeline import Pipeline
from backtest.runner import run_backtest, export_backtest_results
from backtest.metrics import compute_metrics, print_report


console = Console()


def format_table(df, title="Results"):
    """Render a DataFrame as a rich table."""
    if df.empty:
        console.print("[yellow]No results to display.[/yellow]")
        return

    # Pick the most useful columns to display
    display_cols = []
    preferred = [
        "symbol", "companyName", "sector", "marketCap",
        "sector_gap_score", "midcap_premium", "market_cap_score",
        "profitability_score", "sentiment_score", "total_score",
    ]
    for col in preferred:
        if col in df.columns:
            display_cols.append(col)

    # Fallback: if none of the preferred columns exist, show first 8
    if not display_cols:
        display_cols = list(df.columns[:8])

    table = Table(title=title, show_lines=True)

    table.add_column("#", style="dim", width=4)
    for col in display_cols:
        if "score" in col.lower() or "premium" in col.lower():
            table.add_column(col, justify="right", style="cyan")
        elif col == "marketCap":
            table.add_column("Market Cap", justify="right", style="green")
        elif col == "symbol":
            table.add_column("Ticker", style="bold white")
        elif col == "companyName":
            table.add_column("Company", max_width=30)
        elif col == "sector":
            table.add_column("Sector", style="magenta")
        else:
            table.add_column(col)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        values = [str(i)]
        for col in display_cols:
            val = row[col]
            if col == "marketCap" and isinstance(val, (int, float)):
                values.append(f"${val/1e9:.1f}B")
            elif isinstance(val, float):
                values.append(f"{val:.2f}")
            elif val is None:
                values.append("-")
            else:
                values.append(str(val))
        table.add_row(*values)

    console.print()
    console.print(table)


def cmd_run(args):
    """Run the full prediction pipeline."""
    client = FMPClient(use_cache=not args.no_cache)
    pipe = Pipeline(client)

    results = pipe.run(
        skip_profitability=args.skip_profitability,
        top_n=args.top,
    )

    format_table(results, title="S&P 500 Inclusion Predictions")

    if args.export and not results.empty:
        pipe.export_results(results, label="predictions")


def cmd_watchlist(args):
    """Show removal risks and top candidates."""
    client = FMPClient()
    pipe = Pipeline(client)

    data = pipe.show_watchlist()

    format_table(data["bottom"], title="S&P 500 — Smallest Members (Removal Risk)")
    if not data["at_risk"].empty:
        format_table(data["at_risk"], title="S&P 500 — Below $10B Threshold")
    format_table(data["candidates"], title="Top Inclusion Candidates")


def cmd_simulate(args):
    """Simulate a removal and predict replacement."""
    client = FMPClient()
    pipe = Pipeline(client)

    results = pipe.run_removal_scenario(
        removed_ticker=args.ticker.upper(),
        top_n=args.top,
    )

    format_table(results, title=f"Predicted Replacements for {args.ticker.upper()}")

    if args.export and not results.empty:
        pipe.export_results(results, label=f"simulate_{args.ticker.upper()}")


def cmd_score(args):
    """Evaluate one ticker through hard filters and scoring."""
    client = FMPClient(use_cache=not args.no_cache)
    pipe = Pipeline(client)

    result = pipe.score_single_ticker(
        ticker=args.ticker,
        skip_profitability=args.skip_profitability,
    )

    if not result.get("found"):
        console.print(f"[red]{result.get('symbol')}[/red]: {result.get('reason')}")
        return

    hf = result.get("hard_filters", {})
    console.print(f"\n[bold]Ticker:[/bold] {result.get('symbol')}")
    console.print(f"  market_cap:     {'PASS' if hf.get('market_cap') else 'FAIL'}")
    console.print(f"  domicile:       {'PASS' if hf.get('domicile') else 'FAIL'}")
    if "data_quality" in hf:
        console.print(f"  data_quality:   {'PASS' if hf.get('data_quality') else 'FAIL'}")
    console.print(f"  not_in_sp500:   {'PASS' if hf.get('not_in_sp500') else 'FAIL'}")
    console.print(f"  profitability:  {'PASS' if hf.get('profitability') else 'FAIL'}")

    if not result.get("eligible"):
        reason = result.get("reason")
        if reason:
            console.print(f"\n[yellow]{reason}[/yellow]")
        row = result.get("candidate_row")
        if row is not None and not row.empty:
            format_table(row, title=f"{args.ticker.upper()} Snapshot")
        return

    score_row = result.get("score_row")
    rank = result.get("rank")
    size = result.get("universe_size")
    console.print(f"\n[green]Eligible[/green] | Rank: [bold]{rank}[/bold] / {size}")
    format_table(score_row, title=f"{args.ticker.upper()} Scoring Breakdown")


def cmd_backtest(args):
    """Run walk-forward backtest over historical S&P 500 events."""
    client = FMPClient(use_cache=not args.no_cache)

    console.print(f"\n[bold]Running backtest[/bold] (min_year={args.min_year}, "
                  f"max_events={args.max_events or 'all'}, top_n={args.top}, "
                  f"skip_liquidity={args.skip_liquidity})\n")

    results = run_backtest(
        client=client,
        min_year=args.min_year,
        max_events=args.max_events,
        top_n=args.top,
        skip_liquidity=args.skip_liquidity,
        show_progress=True,
    )

    if not results:
        console.print("[yellow]No backtest results.[/yellow]")
        return

    metrics = compute_metrics(results, top_n=args.top)
    print_report(results, metrics, top_n=args.top)

    if args.export:
        export_backtest_results(results, label="backtest")


def cmd_clear_cache(args):
    """Clear all cached API responses."""
    client = FMPClient()
    count = client.clear_cache()
    console.print(f"[green]Cleared {count} cached files.[/green]")


def main():
    parser = argparse.ArgumentParser(
        prog="sp500-predictor",
        description="Shadow Committee: S&P 500 Inclusion Predictor",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run ---
    run_p = subparsers.add_parser("run", help="Run the full prediction pipeline")
    run_p.add_argument("--top", type=int, default=TOP_N_CANDIDATES,
                       help="Number of top candidates to show")
    run_p.add_argument("--skip-profitability", action="store_true",
                       help="Skip profitability API calls (faster, less accurate)")
    run_p.add_argument("--export", action="store_true",
                       help="Save results to output/ directory")
    run_p.add_argument("--no-cache", action="store_true",
                       help="Ignore cached API responses")

    # --- watchlist ---
    subparsers.add_parser("watchlist", help="Show removal risks + top candidates")

    # --- simulate ---
    sim_p = subparsers.add_parser("simulate",
                                   help="Simulate a removal and predict replacement")
    sim_p.add_argument("ticker", type=str,
                       help="Ticker of the S&P 500 company being removed")
    sim_p.add_argument("--top", type=int, default=5,
                       help="Number of replacement candidates to show")
    sim_p.add_argument("--export", action="store_true",
                       help="Save results to output/ directory")

    # --- score ---
    score_p = subparsers.add_parser(
        "score",
        help="Score a single ticker and show rank/filter diagnostics",
    )
    score_p.add_argument("ticker", type=str, help="Ticker symbol to evaluate")
    score_p.add_argument(
        "--skip-profitability",
        action="store_true",
        help="Skip profitability check in both eligibility and ranking",
    )
    score_p.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached API responses",
    )

    # --- backtest ---
    bt_p = subparsers.add_parser("backtest",
                                  help="Run walk-forward backtest on historical events")
    bt_p.add_argument("--min-year", type=int, default=2020,
                      help="Only include events from this year onward (default: 2020)")
    bt_p.add_argument("--max-events", type=int, default=None,
                      help="Limit to N most recent events (default: all)")
    bt_p.add_argument("--top", type=int, default=TOP_N_CANDIDATES,
                      help="Top-N threshold for hit rate (default: 10)")
    bt_p.add_argument("--skip-liquidity", action="store_true",
                      help="Skip PIT liquidity filter (faster)")
    bt_p.add_argument("--export", action="store_true",
                      help="Save results CSV to output/ directory")
    bt_p.add_argument("--no-cache", action="store_true",
                      help="Ignore cached API responses")

    # --- clear-cache ---
    subparsers.add_parser("clear-cache", help="Delete all cached API responses")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Check API key
    placeholder_keys = {"REPLACE_WITH_YOUR_KEY", "your_api_key_here", ""}
    if args.command != "clear-cache" and FMP_API_KEY in placeholder_keys:
        console.print(
            "[red]Error:[/red] No API key configured.\n"
            "Set your FMP API key in the .env file:\n"
            "  FMP_API_KEY=your_key_here\n\n"
            "Get a free key at: https://financialmodelingprep.com/"
        )
        sys.exit(1)

    commands = {
        "run": cmd_run,
        "watchlist": cmd_watchlist,
        "simulate": cmd_simulate,
        "score": cmd_score,
        "backtest": cmd_backtest,
        "clear-cache": cmd_clear_cache,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
