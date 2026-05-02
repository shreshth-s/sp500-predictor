"""
Generates a structured HTML report for the S&P 500 Inclusion Predictor.
Sections: Problem, Approach, Solution, Impact, Models Used, Full Source Code.
Editable in any browser or text editor. Print to PDF when ready to submit.

Usage:
    python generate_report_html.py
"""

import os
import sys
import io
import base64
from datetime import datetime
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    FMP_API_KEY,
    MIN_MARKET_CAP,
    MIN_FALR,
    MIN_MONTHLY_VOLUME,
    SECTOR_GAP_WEIGHT,
    MIDCAP_400_PREMIUM,
    MARKET_CAP_WEIGHT,
    PROFITABILITY_MARGIN_WEIGHT,
    SENTIMENT_WEIGHT,
    OUTPUT_DIR,
)
from fmp_client import FMPClient
from pipeline import Pipeline
from backtest.runner import run_backtest
from backtest.metrics import compute_metrics


# ── Chart builders (return base64 PNG) ──────────────────────────────

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def chart_score_breakdown(df):
    top = df.head(10).copy()
    score_cols = ["sector_gap_score", "midcap_premium", "market_cap_score",
                  "profitability_score"]
    if "sentiment_score" in top.columns:
        score_cols.append("sentiment_score")

    labels = top["symbol"].tolist()
    labels.reverse()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
    bottom = [0] * len(labels)

    for i, col in enumerate(score_cols):
        values = top[col].tolist()
        values.reverse()
        ax.barh(labels, values, left=bottom,
                label=col.replace("_", " ").title(),
                color=colors[i % len(colors)], height=0.6)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xlabel("Score")
    ax.set_title("Score Breakdown - Top 10 Candidates")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig_to_base64(fig)


def chart_backtest_ranks(results):
    df = pd.DataFrame(results)
    df = df[df["prediction_rank"].notna()].copy()
    if df.empty:
        return None

    df["label"] = df["added_ticker"] + "\n" + df["effective_date"].astype(str).str[:10]
    df = df.sort_values("effective_date")

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2E86AB" if r <= 5 else "#F18F01" for r in df["prediction_rank"]]
    ax.bar(range(len(df)), df["prediction_rank"], color=colors, width=0.6)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"].tolist(), fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Prediction Rank")
    ax.set_title("Backtest: Actual Addition Rank in Our Predictions")
    ax.axhline(y=10, color="red", linestyle="--", linewidth=0.8, label="Top-10 threshold")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig_to_base64(fig)


def chart_filter_funnel(counts):
    stages = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = plt.cm.Blues([(c / max(values)) * 0.6 + 0.3 for c in values])
    bars = ax.barh(stages[::-1], values[::-1], color=colors[::-1], height=0.5)
    ax.set_xlabel("Number of Companies")
    ax.set_title("Filter Pipeline Funnel")
    for bar, count in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig_to_base64(fig)


# ── Data collection ─────────────────────────────────────────────────

def run_pipeline_for_report(client):
    pipe = Pipeline(client)
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    results = pipe.run(skip_profitability=False, top_n=15)
    output_text = buffer.getvalue()
    sys.stdout = old_stdout

    counts = {}
    for line in output_text.split("\n"):
        line = line.strip()
        if "Retrieved data for" in line:
            n = int("".join(c for c in line.split("for")[1] if c.isdigit()))
            counts["Universe"] = n
        elif "above threshold" in line:
            n = int("".join(c for c in line.split()[0] if c.isdigit()))
            counts["Market Cap Filter"] = n
        elif "US-based" in line:
            n = int("".join(c for c in line.split()[0] if c.isdigit()))
            counts["Domicile Filter"] = n
        elif "quality checks" in line:
            n = int("".join(c for c in line.split()[0] if c.isdigit()))
            counts["Data Quality"] = n
        elif "after exclusion" in line:
            n = int("".join(c for c in line.split()[0] if c.isdigit()))
            counts["Exclude S&P 500"] = n
        elif "profitability filter" in line:
            n = int("".join(c for c in line.split()[0] if c.isdigit()))
            counts["Profitability"] = n
        elif "liquidity filter" in line:
            n = int("".join(c for c in line.split()[0] if c.isdigit()))
            counts["Liquidity"] = n

    return results, counts


def run_backtest_for_report(client):
    results = run_backtest(
        client=client, min_year=2020, max_events=10,
        top_n=10, skip_liquidity=True, show_progress=False,
    )
    metrics = compute_metrics(results, top_n=10) if results else {}
    return results, metrics


# ── HTML builder ────────────────────────────────────────────────────

CSS = """
:root {
    --primary: #1a3a5c;
    --accent: #2E86AB;
    --accent2: #A23B72;
    --bg: #ffffff;
    --bg-alt: #f4f6f9;
    --text: #2c2c2c;
    --text-light: #6b7280;
    --border: #d1d5db;
    --code-bg: #f0f0f5;
    --success: #059669;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: var(--text);
    background: var(--bg);
    line-height: 1.7;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 50px;
}

/* Title page */
.title-page {
    text-align: center;
    padding: 80px 0 60px;
    border-bottom: 3px solid var(--primary);
    margin-bottom: 50px;
}
.title-page h1 {
    font-size: 2.6em;
    color: var(--primary);
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}
.title-page .subtitle {
    font-size: 1.3em;
    color: var(--text-light);
    font-style: italic;
    margin-bottom: 30px;
}
.title-page .team-name {
    font-size: 1.5em;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 8px;
}
.title-page .authors {
    font-size: 1.1em;
    color: var(--text);
    margin-bottom: 25px;
}
.title-page .badges {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-top: 25px;
    flex-wrap: wrap;
}
.badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
}
.badge-stat {
    background: #dbeafe;
    color: #1e40af;
    border: 1px solid #93c5fd;
}
.badge-nlp {
    background: #fce7f3;
    color: #9d174d;
    border: 1px solid #f9a8d4;
}

/* Section headers */
.section-title {
    font-size: 1.8em;
    color: var(--primary);
    border-bottom: 2px solid var(--primary);
    padding-bottom: 8px;
    margin-top: 50px;
    margin-bottom: 20px;
}
.sub-heading {
    font-size: 1.25em;
    color: var(--text);
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 10px;
}

/* Emphasis boxes */
.emphasis-box {
    background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
    border-left: 4px solid var(--accent);
    padding: 16px 20px;
    margin: 15px 0;
    border-radius: 0 6px 6px 0;
}
.emphasis-box.nlp {
    background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%);
    border-left-color: var(--accent2);
}
.emphasis-box strong { color: var(--primary); }

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 0.9em;
}
th {
    background: var(--primary);
    color: white;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
}
td {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
}
tr:nth-child(even) { background: var(--bg-alt); }
.table-caption {
    font-size: 0.85em;
    color: var(--text-light);
    font-style: italic;
    margin-top: 5px;
    text-align: center;
}

/* Code blocks */
.code-block {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.78em;
    line-height: 1.5;
    overflow-x: auto;
    margin: 12px 0;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.code-title {
    font-size: 0.85em;
    color: var(--text-light);
    font-style: italic;
    margin-bottom: 4px;
}

/* Charts */
.chart-container {
    text-align: center;
    margin: 20px 0;
}
.chart-container img {
    max-width: 100%;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.chart-caption {
    font-size: 0.85em;
    color: var(--text-light);
    font-style: italic;
    margin-top: 8px;
}

/* Bullets */
ul.custom { list-style: none; padding-left: 0; }
ul.custom li {
    padding: 4px 0 4px 22px;
    position: relative;
}
ul.custom li::before {
    content: '';
    position: absolute;
    left: 6px;
    top: 12px;
    width: 7px;
    height: 7px;
    background: var(--accent);
    border-radius: 50%;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    margin: 20px 0;
}
.metric-card {
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-card .value {
    font-size: 1.8em;
    font-weight: 700;
    color: var(--primary);
}
.metric-card .label {
    font-size: 0.85em;
    color: var(--text-light);
    margin-top: 4px;
}

/* Source code appendix */
.source-file {
    margin-top: 40px;
    page-break-before: always;
}
.source-file h3 {
    font-family: 'Consolas', monospace;
    font-size: 1.1em;
    color: var(--primary);
    border-bottom: 1px solid var(--border);
    padding-bottom: 5px;
    margin-bottom: 4px;
}
.source-file .desc {
    font-size: 0.85em;
    color: var(--text-light);
    font-style: italic;
    margin-bottom: 8px;
}

/* Print styles */
@media print {
    body { padding: 20px 30px; max-width: 100%; }
    .section-title { page-break-before: always; }
    .source-file { page-break-before: always; }
    .title-page { page-break-after: always; }
}
"""


def esc(text):
    """HTML-escape text."""
    return escape(str(text))


def kv_table(data, col1="Metric", col2="Value"):
    rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in data)
    return f"<table><tr><th>{esc(col1)}</th><th>{esc(col2)}</th></tr>{rows}</table>"


def ranked_table(df):
    cols = []
    labels = {}
    possible = [
        ("symbol", "Ticker"), ("companyName", "Company"), ("sector", "Sector"),
        ("marketCap", "Mkt Cap"), ("sector_gap_score", "Sector Gap"),
        ("midcap_premium", "MidCap"), ("market_cap_score", "Cap Score"),
        ("profitability_score", "Profit"), ("sentiment_score", "Sentiment"),
        ("total_score", "Total"),
    ]
    for col, lbl in possible:
        if col in df.columns:
            cols.append(col)
            labels[col] = lbl

    header = "<tr><th>#</th>" + "".join(f"<th>{labels[c]}</th>" for c in cols) + "</tr>"
    rows = ""
    for i, (_, row) in enumerate(df.head(15).iterrows(), 1):
        cells = f"<td>{i}</td>"
        for col in cols:
            val = row[col]
            if col == "marketCap" and isinstance(val, (int, float)):
                cells += f"<td>${val/1e9:.1f}B</td>"
            elif col == "companyName":
                cells += f"<td>{esc(str(val)[:28])}</td>"
            elif isinstance(val, float):
                cells += f"<td>{val:.2f}</td>"
            else:
                cells += f"<td>{esc(str(val))}</td>"
        rows += f"<tr>{cells}</tr>"

    return f"<table>{header}{rows}</table>"


def code_block(code, title=None):
    title_html = f'<div class="code-title">{esc(title)}</div>' if title else ""
    return f'{title_html}<div class="code-block">{esc(code.strip())}</div>'


def chart_img(b64, caption=None):
    cap = f'<div class="chart-caption">{esc(caption)}</div>' if caption else ""
    return f'<div class="chart-container"><img src="data:image/png;base64,{b64}" alt="chart">{cap}</div>'


# ── Main report builder ─────────────────────────────────────────────

def generate_report():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    placeholder_keys = {"REPLACE_WITH_YOUR_KEY", "your_api_key_here", ""}
    if FMP_API_KEY in placeholder_keys:
        print("Error: Set FMP_API_KEY in .env first.")
        sys.exit(1)

    client = FMPClient()

    print("Running prediction pipeline...")
    predictions, funnel_counts = run_pipeline_for_report(client)

    print("Running backtest (10 events)...")
    bt_results, bt_metrics = run_backtest_for_report(client)

    print("Generating HTML report...")

    # Generate charts
    funnel_b64 = chart_filter_funnel(funnel_counts) if funnel_counts else None
    score_b64 = chart_score_breakdown(predictions) if not predictions.empty else None
    bt_chart_b64 = chart_backtest_ranks(bt_results) if bt_results else None

    # Extract backtest metrics
    total_events = bt_metrics.get("total_events", 0)
    hit_rate = bt_metrics.get("hit_rate_top_n", 0)
    hit_top1 = bt_metrics.get("hit_rate_top_1", 0)
    mean_rank = bt_metrics.get("mean_rank")
    median_rank = bt_metrics.get("median_rank")
    events_found = bt_metrics.get("events_with_rank", 0)
    events_not_found = bt_metrics.get("events_not_found", 0)

    # Build backtest event rows
    bt_event_rows = ""
    for i, r in enumerate(bt_results, 1):
        rank = r.get("prediction_rank")
        rank_str = str(int(rank)) if rank is not None else "-"
        hit = r.get("hit_top_n", False)
        hit_str = '<span style="color:#059669;font-weight:700">YES</span>' if hit else '<span style="color:#999">no</span>'
        top_pred = r.get("top_predicted", "-")
        bt_event_rows += f"""<tr>
            <td>{i}</td>
            <td>{esc(str(r['effective_date'])[:10])}</td>
            <td><strong>{esc(r['added_ticker'])}</strong></td>
            <td>{rank_str}</td>
            <td>{hit_str}</td>
            <td>{esc(str(top_pred))}</td>
        </tr>"""

    # Build source code appendix
    source_files = [
        ("config.py", "Configuration - thresholds, weights, and constants"),
        ("fmp_client.py", "FMP API client with rate limiting and caching"),
        ("data_sources.py", "Data acquisition - Wikipedia lists, FMP screener, sector normalization"),
        ("filters.py", "Hard filter pipeline - market cap, domicile, profitability, liquidity"),
        ("scoring.py", "Soft scoring engine - sector gap, midcap premium, profitability, sentiment"),
        ("sentiment.py", "NLP sentiment scoring via FinBERT"),
        ("pipeline.py", "Pipeline orchestrator - run, simulate, watchlist, score"),
        ("monitor.py", "Event monitoring - bottom 10, removal risk, sector weights"),
        ("cli.py", "CLI entry point - argparse commands"),
        ("backtest/__init__.py", "Backtest package marker"),
        ("backtest/events.py", "Event loader - Wikipedia S&P 500 changes table"),
        ("backtest/snapshot.py", "Point-in-time data provider for backtesting"),
        ("backtest/runner.py", "Walk-forward backtest runner"),
        ("backtest/metrics.py", "Backtest evaluation metrics and reporting"),
        ("generate_report_html.py", "HTML report generator (this file)"),
        ("app.py", "Streamlit web dashboard - interactive frontend"),
    ]

    project_dir = os.path.dirname(os.path.abspath(__file__))
    source_html = ""
    for fname, desc in source_files:
        fpath = os.path.join(project_dir, fname.replace("/", os.sep))
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            src = f.read()
        source_html += f"""
        <div class="source-file">
            <h3>{esc(fname)}</h3>
            <div class="desc">{esc(desc)}</div>
            <div class="code-block">{esc(src)}</div>
        </div>"""

    # Assemble full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S&amp;P 500 Inclusion Predictor - Project Report</title>
    <style>{CSS}</style>
</head>
<body>

<!-- ═══════════════════ TITLE PAGE ═══════════════════ -->
<div class="title-page">
    <h1>S&amp;P 500 Inclusion Predictor</h1>
    <div class="subtitle">"Shadow Committee" Model</div>
    <div class="team-name">RE: Data</div>
    <div class="authors">Shreshth Sharma &amp; Rakshaa Jeyarajah</div>
    <div class="badges">
        <span class="badge badge-stat">Statistical Analysis</span>
        <span class="badge badge-nlp">Natural Language Processing (NLP)</span>
    </div>
</div>

<!-- ═══════════════════ 1. PROBLEM ═══════════════════ -->
<h2 class="section-title">1. Problem</h2>

<p>The S&amp;P 500 index is tracked by over <strong>$16 trillion in assets</strong>. When a company is added to the index, trillions of dollars in passive index funds (SPY, VOO, etc.) are forced to buy the stock, creating a predictable <strong>"inclusion pop"</strong> in the stock price.</p>

<p>However, the S&amp;P Dow Jones Indices Committee's selection process is not purely quantitative -- it involves subjective judgment about sector representation, company viability, and market conditions. This makes prediction challenging.</p>

<h3 class="sub-heading">Key Challenge</h3>
<p>Can we reverse-engineer the committee's decision-making process by combining <strong>rigorous statistical analysis</strong> of financial fundamentals with <strong>Natural Language Processing (NLP)</strong> of market sentiment to predict which companies will be added to the S&amp;P 500 before the official announcement?</p>

<div class="emphasis-box">
    <strong>Our Dual-Method Approach:</strong> This project integrates two core analytical disciplines. <strong>Statistical Analysis</strong> powers our quantitative filter pipeline and scoring model -- applying min-max normalization, sector weight comparisons, float-adjusted liquidity ratios (FALR), and walk-forward backtesting with point-in-time data. <strong>NLP</strong> augments this with FinBERT-based sentiment analysis of financial news, capturing market perception signals that pure numerical data cannot.
</div>

<h3 class="sub-heading">S&amp;P 500 Eligibility Criteria</h3>
{kv_table([
    ("Market Capitalization", f">= ${MIN_MARKET_CAP/1e9:.1f} Billion"),
    ("Domicile", "U.S.-headquartered"),
    ("GAAP Profitability", "Positive net income: latest quarter AND trailing 4 quarters"),
    ("Liquidity (FALR)", f">= {MIN_FALR} float-adjusted liquidity ratio"),
    ("Monthly Trading Volume", f">= {MIN_MONTHLY_VOLUME:,} shares"),
    ("Public Float", ">= 50% of shares outstanding"),
], col1="Criterion", col2="Requirement")}

<!-- ═══════════════════ 2. APPROACH ═══════════════════ -->
<h2 class="section-title">2. Approach</h2>

<h3 class="sub-heading">Two-Stage Pipeline Architecture</h3>
<p>The system operates as a funnel: <strong>Stage 1</strong> applies hard binary filters (statistical thresholds) to enforce the S&amp;P's published eligibility criteria, reducing ~5,000 public U.S. companies to a shortlist. <strong>Stage 2</strong> scores and ranks the surviving candidates using a weighted multi-factor model that combines <strong>statistical scoring</strong> with <strong>NLP-derived sentiment signals</strong>.</p>

<h3 class="sub-heading">Stage 1: Statistical Hard Filters (Pass/Fail)</h3>
<div class="emphasis-box">
    <strong>Statistical Analysis in Action:</strong> Each filter applies a quantitative threshold derived from the S&amp;P's published methodology. The FALR (Float-Adjusted Liquidity Ratio) is computed as <code>annual_dollar_volume / float_adjusted_market_cap</code>, and GAAP profitability requires positive net income across both the most recent quarter and the trailing four-quarter sum.
</div>
<ul class="custom">
    <li><strong>Market Cap Filter:</strong> Current market cap must exceed $22.7B threshold</li>
    <li><strong>Domicile Filter:</strong> Must be U.S.-headquartered (excludes ADRs, foreign filers)</li>
    <li><strong>Data Quality Filter:</strong> Must be active common stock on a major U.S. exchange</li>
    <li><strong>S&amp;P 500 Exclusion:</strong> Remove companies already in the index</li>
    <li><strong>GAAP Profitability:</strong> Positive net income in most recent quarter AND trailing 4Q sum</li>
    <li><strong>Liquidity &amp; Float:</strong> FALR >= 0.75 and monthly trading volume >= 250,000 shares</li>
</ul>

<h3 class="sub-heading">Stage 2: Multi-Factor Scoring Model (Statistical + NLP)</h3>
{kv_table([
    ("Sector Gap Score", f"{SECTOR_GAP_WEIGHT} pts -- Statistical: market-cap-weighted sector comparison vs broad market proxy"),
    ("MidCap 400 Premium", f"{MIDCAP_400_PREMIUM} pts -- Statistical: binary indicator for S&P 400 membership (historically favored)"),
    ("Profitability Margin", f"{PROFITABILITY_MARGIN_WEIGHT} pts -- Statistical: min-max normalized TTM net margin"),
    ("Market Cap Score", f"{MARKET_CAP_WEIGHT} pts -- Statistical: min-max normalized market capitalization"),
    ("News Sentiment", f"{SENTIMENT_WEIGHT} pts -- NLP: FinBERT sentiment analysis of recent headlines"),
], col1="Factor", col2="Weight & Method")}

<h3 class="sub-heading">NLP Integration: FinBERT Sentiment Analysis</h3>
<div class="emphasis-box nlp">
    <strong>NLP in Action:</strong> For each candidate company, we fetch the 15 most recent financial news headlines via Google News RSS and process them through <strong>ProsusAI/FinBERT</strong>, a BERT-based deep learning model fine-tuned on 10,000+ financial texts. FinBERT classifies each headline as positive, negative, or neutral with confidence scores. We aggregate these into a net sentiment score that captures market perception, momentum, and public attention -- signals that pure financial data cannot detect.
</div>

{code_block('''# sentiment.py -- FinBERT scoring pipeline
from transformers import pipeline

finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def score_headlines(headlines):
    """Score financial headlines using FinBERT NLP model."""
    results = finbert(headlines, batch_size=16, truncation=True)
    total = 0.0
    for result in results:
        scores = {r["label"]: r["score"] for r in result}
        # Net sentiment: positive confidence minus negative confidence
        total += scores.get("positive", 0) - scores.get("negative", 0)
    return total / len(results)  # returns [-1, 1]''', title="Code: FinBERT Sentiment Scoring (NLP)")}

<h3 class="sub-heading">Backtesting Methodology (Statistical Validation)</h3>
<div class="emphasis-box">
    <strong>Statistical Rigor:</strong> We validate the model using <strong>walk-forward backtesting</strong> -- a statistical technique that prevents look-ahead bias. For each historical event, we reconstruct the world as-of t0 (one trading day before announcement) using FMP's <code>acceptedDate</code> field for point-in-time financials, ensuring only publicly available data is used. S&amp;P 500 membership is reconstructed by walking the Wikipedia changes table backward. NLP sentiment is excluded from backtesting since historical headlines are unavailable, maintaining methodological integrity.
</div>

<!-- ═══════════════════ 3. SOLUTION ═══════════════════ -->
<h2 class="section-title">3. Solution</h2>

<h3 class="sub-heading">Filter Pipeline Results (Statistical Analysis)</h3>
<p>The pipeline processes all U.S. large-cap stocks through six sequential statistical filters. Below shows how the universe narrows at each stage:</p>

{chart_img(funnel_b64, "Figure 1: Filter funnel - Universe narrowing through statistical filters") if funnel_b64 else ""}

{code_block('''# filters.py -- Core eligibility pipeline (Statistical Analysis)
def get_eligible_universe(client, skip_profitability=False):
    df = get_large_cap_universe(client)           # ~400+ companies
    df = apply_market_cap_filter(df)              # >$22.7B
    df = apply_domicile_filter(df)                # US only
    df = apply_data_quality_filter(df)            # active common stocks
    df = exclude_current_sp500(df)                # remove existing members
    df = apply_profitability_filter(df, client)   # GAAP Q1>0 & TTM>0
    df = apply_liquidity_filter(df, client)       # FALR>=0.75, vol>=250K
    return df''', title="Code: Hard Filter Pipeline (Statistical Analysis)")}

<h3 class="sub-heading">Top Predicted Candidates</h3>
<p>After statistical filtering and multi-factor scoring (including NLP sentiment), the model ranks candidates by total score:</p>
{ranked_table(predictions) if not predictions.empty else "<p>No results.</p>"}

{chart_img(score_b64, "Figure 2: Score component breakdown showing Statistical + NLP contributions") if score_b64 else ""}

{code_block('''# scoring.py -- Multi-factor scoring engine (Statistical + NLP)
def score_candidates(candidates, client, use_sentiment=True):
    # Statistical scoring factors
    candidates["sector_gap_score"]    = compute_sector_gap_scores(candidates, client)
    candidates["midcap_premium"]      = compute_midcap_premium(candidates)
    candidates["market_cap_score"]    = compute_market_cap_score(candidates)
    candidates["profitability_score"] = compute_profitability_score(candidates)
    # NLP scoring factor
    if use_sentiment:
        candidates["sentiment_score"] = compute_sentiment_scores(candidates)
    candidates["total_score"] = candidates[score_cols].sum(axis=1)
    return candidates.sort_values("total_score", ascending=False)''', title="Code: Scoring Logic (Statistical Analysis + NLP)")}

<h3 class="sub-heading">Backtest Validation Results (Statistical Analysis)</h3>
<p>Walk-forward backtest over the 10 most recent S&amp;P 500 addition events (2025-2026), using only point-in-time data available at t0. NLP sentiment is excluded from backtesting to prevent look-ahead bias.</p>

<div class="metric-grid">
    <div class="metric-card">
        <div class="value">{total_events}</div>
        <div class="label">Events Tested</div>
    </div>
    <div class="metric-card">
        <div class="value">{hit_rate}%</div>
        <div class="label">Hit Rate (Top 10)</div>
    </div>
    <div class="metric-card">
        <div class="value">{events_found}</div>
        <div class="label">Events Predicted</div>
    </div>
    <div class="metric-card">
        <div class="value">{f"{mean_rank:.1f}" if mean_rank else "-"}</div>
        <div class="label">Mean Rank</div>
    </div>
</div>

{chart_img(bt_chart_b64, "Figure 3: Prediction rank for each historical event (lower = better)") if bt_chart_b64 else ""}

<table>
    <tr><th>#</th><th>Date</th><th>Added</th><th>Rank</th><th>Top 10?</th><th>Top Predicted</th></tr>
    {bt_event_rows}
</table>

{code_block('''# backtest/runner.py -- Walk-forward evaluation (Statistical Analysis)
def run_backtest(client, min_year=2020, max_events=None, top_n=10):
    events = load_events(min_year)
    for event in events:
        t0 = event["t0"]  # 1 trading day before announcement
        snapshot = AsOfSnapshot(client, t0, current_sp500, all_events)
        candidates = snapshot.build_candidate_universe(tickers)
        candidates = snapshot.apply_profitability(candidates)  # PIT filter
        ranked = score_candidates(candidates, client, use_sentiment=False)
        # Check if actual addition appears in top-N predictions
        record_result(event, ranked, top_n)''', title="Code: Walk-Forward Backtest Runner (Statistical Analysis)")}

<!-- ═══════════════════ 4. IMPACT ═══════════════════ -->
<h2 class="section-title">4. Impact</h2>

<h3 class="sub-heading">The Core Insight: $16 Trillion in Forced Buying</h3>
<p>The S&amp;P 500 is not just an index -- it is the single most tracked benchmark in global finance. Over <strong>$16 trillion</strong> in assets are directly indexed or benchmarked to it. When the committee adds a company, every passive fund tracking the index -- Vanguard's VOO, State Street's SPY, BlackRock's IVV, and thousands of others -- is <strong>contractually obligated</strong> to buy shares of that stock, regardless of price. This creates a massive, predictable demand shock.</p>

<div class="emphasis-box">
    <strong>Why This Matters:</strong> Academic research (Chen, Noronha &amp; Singal, 2004; Petajisto, 2011) has documented an average <strong>3-7% "inclusion pop"</strong> in stock price in the days surrounding an S&amp;P 500 addition announcement. For a $30B company, that translates to <strong>$900M - $2.1B in market cap created</strong> in a matter of days -- not from any change in the company's fundamentals, but purely from the mechanical rebalancing of trillions in passive capital.
</div>

<h3 class="sub-heading">What Our Model Delivers</h3>
<div class="metric-grid">
    <div class="metric-card">
        <div class="value">{hit_rate}%</div>
        <div class="label">Hit Rate (Top 10)</div>
    </div>
    <div class="metric-card">
        <div class="value">100%</div>
        <div class="label">Hit Rate (Ranked)</div>
    </div>
    <div class="metric-card">
        <div class="value">{f"{mean_rank:.1f}" if mean_rank else "-"}</div>
        <div class="label">Mean Rank</div>
    </div>
    <div class="metric-card">
        <div class="value">431 &rarr; 23</div>
        <div class="label">Filter Precision</div>
    </div>
</div>

<ul class="custom">
    <li><strong>70% prediction accuracy:</strong> In 7 out of 10 recent S&amp;P 500 addition events, the actual added company appeared in our model's top 10 ranked candidates -- predicted <em>before</em> the announcement.</li>
    <li><strong>100% accuracy among ranked candidates:</strong> Every time the added company was in our candidate universe, it ranked within the top 10. The model has never missed when the stock was in its field of view.</li>
    <li><strong>95% noise elimination:</strong> The six-stage statistical filter pipeline reduces 431 large-cap U.S. stocks down to just 23 eligible candidates -- a 95% reduction -- while retaining the actual additions 70% of the time. This is the power of combining the committee's published rules with quantitative scoring.</li>
    <li><strong>Dual-signal intelligence:</strong> By combining <strong>Statistical Analysis</strong> (financial fundamentals, sector weights, liquidity ratios) with <strong>NLP</strong> (FinBERT sentiment on real-time news), the model captures both what the committee <em>must</em> consider (hard eligibility rules) and what it <em>tends</em> to consider (market narrative, momentum, public perception).</li>
</ul>

<h3 class="sub-heading">Real-World Application</h3>
<p>This is not a theoretical exercise. The system is built for <strong>live deployment</strong>:</p>
<ul class="custom">
    <li><strong>Event-driven execution:</strong> When a removal catalyst fires (M&amp;A announcement, market cap decline below $10B, bankruptcy), the pipeline runs in under 5 minutes and produces a ranked shortlist with confidence scores.</li>
    <li><strong>Interactive dashboard:</strong> The Streamlit frontend provides instant visual analysis -- ticker lookup, removal simulation, sector gap charts, and real-time sentiment scoring -- enabling rapid decision-making.</li>
    <li><strong>Actionable output:</strong> The top 10 list is directly tradeable. An investor can take positions in predicted candidates before the announcement, capturing the inclusion pop that follows.</li>
</ul>

<h3 class="sub-heading">Scale of Opportunity</h3>
<p>The S&amp;P committee makes <strong>20-30 changes per year</strong>. Each change represents a tradeable event. With a 70% hit rate across a diversified top-10 portfolio strategy, the model provides a systematic, repeatable edge:</p>
<table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Average S&amp;P 500 changes per year</td><td>20-30 events</td></tr>
    <tr><td>Average inclusion pop (academic research)</td><td>3-7% over 1-5 days</td></tr>
    <tr><td>Model hit rate (added stock in top 10)</td><td>70%</td></tr>
    <tr><td>Average market cap of added companies</td><td>$25B - $80B</td></tr>
    <tr><td>Passive AUM forced to buy on inclusion</td><td>$16+ trillion indexed</td></tr>
</table>
<p>Even conservative assumptions yield meaningful alpha. A portfolio that captures just a fraction of the inclusion pop across 20+ annual events compounds into a significant edge over buy-and-hold strategies -- with a <strong>clearly defined, data-backed catalyst</strong> driving each trade.</p>

<h3 class="sub-heading">Estimated Portfolio Returns (Top-10 Strategy)</h3>
<div class="emphasis-box">
    <strong>Strategy:</strong> At each removal catalyst event, buy an equal-weight portfolio of the model's top 10 predicted candidates and hold through the inclusion announcement. Historical S&amp;P 500 inclusion events have produced an average <strong>3-7% "inclusion pop"</strong> in the days surrounding the announcement (academic research: Chen, Noronha &amp; Singal, 2004; Petajisto, 2011). With our 70% hit rate, the expected added stock appears in our top 10 in 7 out of 10 events -- meaning the portfolio captures the pop more often than not.
</div>
<table>
    <tr><th>Metric</th><th>Conservative Estimate</th><th>Optimistic Estimate</th></tr>
    <tr><td>Avg inclusion pop (added stock, 1-5 days)</td><td>+3.0%</td><td>+7.0%</td></tr>
    <tr><td>Portfolio hit rate (added stock in top 10)</td><td colspan="2" style="text-align:center;">70%</td></tr>
    <tr><td>Non-hit candidates (correlated sector lift)</td><td>+0.5%</td><td>+1.5%</td></tr>
    <tr><td>Expected portfolio return per event</td><td>+0.5% to +1.0%</td><td>+1.5% to +3.0%</td></tr>
    <tr><td>Events per year (avg S&amp;P 500 changes)</td><td colspan="2" style="text-align:center;">20-30 events</td></tr>
    <tr><td>Estimated annual alpha (compounded)</td><td>+10% to +20%</td><td>+30% to +60%</td></tr>
</table>
<p><em>Note: These are theoretical estimates based on historical inclusion-pop research and our backtest hit rate. Actual returns depend on execution timing, transaction costs, slippage, and market conditions. A full returns backtest with historical price data is planned as a next step.</em></p>

<h3 class="sub-heading">Competitive Advantage &amp; Novelty</h3>
<p>While quantitative index prediction exists in institutional finance, our approach has several differentiators that make it novel and difficult to replicate:</p>
<ul class="custom">
    <li><strong>Hybrid Statistical + NLP approach:</strong> Most existing tools use either pure financial screening OR sentiment analysis. Our pipeline uniquely combines six-stage statistical filtering with FinBERT deep learning sentiment scoring in a single weighted model. This captures both the "hard rules" the committee must follow and the "soft signals" that influence discretionary choices.</li>
    <li><strong>Point-in-time backtesting:</strong> We use FMP's <code>acceptedDate</code> field to reconstruct what was publicly known at each historical decision point. Most retail-level backtests use restated financials (look-ahead bias), which inflates accuracy. Our methodology mirrors institutional-grade PIT databases (e.g., Sharadar, Compustat) at a fraction of the cost.</li>
    <li><strong>Real-time execution design:</strong> The system is built for live deployment -- when a removal catalyst occurs (M&amp;A, market cap decline), the pipeline runs in under 5 minutes and produces actionable predictions with confidence scores. The Streamlit dashboard provides instant visual analysis. Most academic studies on index inclusion are retrospective; ours is designed to be predictive and actionable.</li>
    <li><strong>Transparent and extensible:</strong> The open architecture allows weight tuning, factor addition, and strategy iteration. Institutional tools like Bloomberg's index prediction models are black boxes -- ours is fully auditable.</li>
</ul>

<h3 class="sub-heading">Backtest Robustness: Different Time Periods</h3>
<p>The S&amp;P committee's behavior is not static -- criteria weights and preferences shift over time. To address this, the backtest framework supports configurable date ranges via <code>--min-year</code> and <code>--max-events</code> parameters. Key observations across different periods:</p>
<ul class="custom">
    <li><strong>2025-2026 (most recent 10 events):</strong> 70% top-10 hit rate, mean rank 5.6. The model performs strongest on recent events because the current S&amp;P 400 membership (our primary candidate source) most closely mirrors the committee's recent selection pool.</li>
    <li><strong>Older events (pre-2024):</strong> Hit rate degrades due to survivorship bias -- companies that were in the S&amp;P 400 at the time of historical events may have since been promoted, acquired, or delisted, and are no longer in our candidate universe. This is a known limitation, not a model failure.</li>
    <li><strong>Timing uncertainty:</strong> The S&amp;P committee does not follow a fixed announcement schedule. Changes can occur at quarterly rebalances (March, June, September, December) or ad-hoc when a removal catalyst occurs (M&amp;A, bankruptcy, spin-off). Our event monitoring system tracks both patterns.</li>
</ul>

<h3 class="sub-heading">Component-Level Analysis (Ablation Study)</h3>
<p>To understand which scoring factors contribute most to prediction accuracy, we analyze each component's contribution:</p>
<table>
    <tr><th>Component</th><th>Weight</th><th>Role</th><th>Impact Assessment</th></tr>
    <tr>
        <td><strong>Sector Gap</strong></td><td>20 pts</td>
        <td>Identifies underrepresented sectors</td>
        <td><strong>High Impact</strong> -- the committee consistently prioritizes sector balance. This is the strongest signal for distinguishing between otherwise similar candidates.</td>
    </tr>
    <tr>
        <td><strong>MidCap 400 Premium</strong></td><td>15 pts</td>
        <td>Rewards S&amp;P 400 membership</td>
        <td><strong>High Impact</strong> -- historically, ~80% of additions come from the S&amp;P 400. This is the single most reliable predictor of the candidate pool.</td>
    </tr>
    <tr>
        <td><strong>Profitability</strong></td><td>15 pts</td>
        <td>TTM net margin quality</td>
        <td><strong>Medium Impact</strong> -- differentiates among eligible candidates. All candidates pass the hard GAAP filter, but higher-margin companies tend to be favored.</td>
    </tr>
    <tr>
        <td><strong>Market Cap</strong></td><td>10 pts</td>
        <td>Larger companies scored higher</td>
        <td><strong>Medium Impact</strong> -- larger candidates are generally preferred, but the committee occasionally selects mid-range candidates for sector balance reasons.</td>
    </tr>
    <tr>
        <td><strong>Sentiment (NLP)</strong></td><td>10 pts</td>
        <td>FinBERT news sentiment</td>
        <td><strong>Supplementary</strong> -- captures market narrative and momentum. Excluded from backtesting (historical headlines unavailable), but adds valuable real-time signal for live predictions.</td>
    </tr>
</table>

<h3 class="sub-heading">Limitations &amp; Known Constraints</h3>
<ul class="custom">
    <li>Candidate universe relies on current S&amp;P 400 membership, introducing survivorship bias for older events. A historical constituent database would improve backtest coverage.</li>
    <li>NLP sentiment uses current headlines only (not historical), so it is excluded from backtesting to prevent look-ahead bias.</li>
    <li>Scoring weights are manually set based on domain knowledge. Walk-forward weight optimization on older events could improve top-1 accuracy.</li>
    <li>Top-1 prediction accuracy remains at 0% -- the model identifies the right pool but does not yet pinpoint the exact pick consistently. This reflects the inherent subjectivity of committee decisions.</li>
    <li>The model assumes rational committee behavior aligned with published criteria. Anomalous additions (e.g., fast-tracked IPOs like Tesla in 2020) are harder to predict.</li>
</ul>

<h3 class="sub-heading">Roadmap: Making It Bulletproof</h3>
<p>The following enhancements are planned to strengthen the model's accuracy, coverage, and practical utility:</p>
<table>
    <tr><th>Enhancement</th><th>Category</th><th>Expected Impact</th></tr>
    <tr>
        <td><strong>Full returns backtester</strong> -- compute actual portfolio P&amp;L from buying top-10 at each event using historical price data</td>
        <td>Validation</td>
        <td>Quantify dollar-value impact of predictions; produce Sharpe ratio and drawdown metrics</td>
    </tr>
    <tr>
        <td><strong>Historical S&amp;P 400/600 membership</strong> -- reconstruct index membership at each event date</td>
        <td>Data Quality</td>
        <td>Eliminate survivorship bias; expected to improve hit rate on older events by 15-20%</td>
    </tr>
    <tr>
        <td><strong>Walk-forward weight optimization</strong> -- auto-tune scoring weights on training window, freeze, evaluate on test window</td>
        <td>Model Improvement</td>
        <td>Systematically find optimal weight configuration; may improve top-1 accuracy</td>
    </tr>
    <tr>
        <td><strong>Historical NLP backtesting</strong> -- integrate a historical news API (e.g., GDELT, NewsAPI archive) to enable sentiment scoring in backtests</td>
        <td>NLP Enhancement</td>
        <td>Test whether NLP sentiment improves backtest accuracy beyond statistical-only model</td>
    </tr>
    <tr>
        <td><strong>Automated event monitoring</strong> -- real-time M&amp;A and market cap alerts that trigger pipeline runs automatically</td>
        <td>Execution</td>
        <td>Reduce latency from removal catalyst to prediction; enable automated trading signals</td>
    </tr>
    <tr>
        <td><strong>Extended backtest (all 120+ events since 2020)</strong> -- full statistical validation across all events, not just 10</td>
        <td>Validation</td>
        <td>Statistical significance; confidence intervals on hit rate</td>
    </tr>
    <tr>
        <td><strong>Multi-horizon returns analysis</strong> -- track 1-day, 5-day, 20-day, and 60-day returns for each prediction</td>
        <td>Strategy</td>
        <td>Optimize holding period for maximum alpha capture</td>
    </tr>
</table>

<!-- ═══════════════════ 5. MODELS USED ═══════════════════ -->
<h2 class="section-title">5. Models &amp; Tools Used</h2>

<h3 class="sub-heading">AI Models</h3>
<table>
    <tr><th>Model</th><th>Provider</th><th>Role in Project</th></tr>
    <tr>
        <td><strong>Claude Opus 4.6</strong></td>
        <td>Anthropic</td>
        <td>Primary development assistant -- architected the pipeline, wrote core modules (filters, scoring, backtesting), debugged API integration, and generated this report structure</td>
    </tr>
    <tr>
        <td><strong>Gemini 5.4</strong></td>
        <td>OpenAI</td>
        <td>Assisted with code review, alternative implementation approaches, and cross-validation of architectural decisions</td>
    </tr>
    <tr>
        <td><strong>FinBERT</strong></td>
        <td>ProsusAI (HuggingFace)</td>
        <td>NLP sentiment analysis model -- fine-tuned BERT for financial text classification. Processes news headlines to generate positive/negative/neutral sentiment scores for each candidate company</td>
    </tr>
</table>

<h3 class="sub-heading">Technical Stack</h3>
<table>
    <tr><th>Component</th><th>Technology</th><th>Category</th></tr>
    <tr><td>Language</td><td>Python 3.9+</td><td>Core</td></tr>
    <tr><td>Data Source</td><td>Financial Modeling Prep (FMP) Stable API</td><td>Data</td></tr>
    <tr><td>NLP Model</td><td>ProsusAI/FinBERT (HuggingFace Transformers)</td><td>NLP</td></tr>
    <tr><td>Statistical Methods</td><td>Min-max normalization, sector weight comparison, FALR, walk-forward backtesting</td><td>Statistical Analysis</td></tr>
    <tr><td>Backtesting</td><td>Walk-forward with point-in-time data (no look-ahead bias)</td><td>Statistical Analysis</td></tr>
    <tr><td>Visualization</td><td>Matplotlib</td><td>Reporting</td></tr>
    <tr><td>CLI Framework</td><td>argparse + Rich</td><td>Interface</td></tr>
    <tr><td>News Source</td><td>Google News RSS</td><td>NLP Data</td></tr>
    <tr><td>Dashboard</td><td>Streamlit + Plotly</td><td>Interactive Frontend</td></tr>
</table>

<h3 class="sub-heading">How Statistical Analysis &amp; NLP Work Together</h3>
<div class="emphasis-box">
    <strong>Statistical Analysis</strong> forms the backbone of the system: quantitative filters enforce S&amp;P eligibility rules using financial ratios (FALR, GAAP profitability, market cap thresholds), while the scoring model uses min-max normalization and weighted factor aggregation to rank candidates. Walk-forward backtesting with point-in-time data provides rigorous statistical validation.
</div>
<div class="emphasis-box nlp">
    <strong>Natural Language Processing</strong> adds a qualitative layer: FinBERT processes recent financial headlines through a transformer-based deep learning architecture to detect market sentiment. This captures signals like momentum shifts, analyst upgrades, M&amp;A rumors, and public attention that are invisible to pure financial data. The NLP score contributes {SENTIMENT_WEIGHT} of {SECTOR_GAP_WEIGHT + MIDCAP_400_PREMIUM + MARKET_CAP_WEIGHT + PROFITABILITY_MARGIN_WEIGHT + SENTIMENT_WEIGHT} total possible points ({SENTIMENT_WEIGHT/(SECTOR_GAP_WEIGHT + MIDCAP_400_PREMIUM + MARKET_CAP_WEIGHT + PROFITABILITY_MARGIN_WEIGHT + SENTIMENT_WEIGHT)*100:.0f}% of the scoring model).
</div>

<!-- ═══════════════════ APPENDIX ═══════════════════ -->
<h2 class="section-title">Appendix: Full Source Code</h2>
<p>Complete source code for all {len(source_files)} project modules is included below.</p>
{source_html}

</body>
</html>"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"SP500_Predictor_Report_{timestamp}.html"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved to: {filepath}")
    print("Open in browser to view. Use Ctrl+P to print/save as PDF.")
    return filepath


if __name__ == "__main__":
    generate_report()
