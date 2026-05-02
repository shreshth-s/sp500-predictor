"""
Streamlit dashboard for the S&P 500 Inclusion Predictor.
Run with: streamlit run app.py
"""

import os
import sys
import time
import threading
from io import StringIO
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    FMP_API_KEY,
    MIN_MARKET_CAP,
    REMOVAL_RISK_CAP,
    TOP_N_CANDIDATES,
    SECTOR_GAP_WEIGHT,
    MIDCAP_400_PREMIUM,
    MARKET_CAP_WEIGHT,
    PROFITABILITY_MARGIN_WEIGHT,
    SENTIMENT_WEIGHT,
)
from fmp_client import FMPClient
from data_sources import get_sp500_tickers, get_sp400_tickers, normalize_sector_name
from filters import get_eligible_universe
from scoring import score_candidates
from monitor import get_bottom_n_sp500, check_removal_candidates, get_sp500_sector_weights
from backtest.runner import run_backtest, export_backtest_results
from backtest.metrics import compute_metrics
from backtest.events import load_events

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="S&P 500 Shadow Committee",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.85;
        font-size: 1.05rem;
    }

    /* Metric cards */
    .metric-card {
        background: #e8edf3;
        border: 1px solid #c5cdd8;
        border-left: 4px solid #0f3460;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-card.green { border-left-color: #28a745; }
    .metric-card.red { border-left-color: #dc3545; }
    .metric-card.blue { border-left-color: #007bff; }
    .metric-card.gold { border-left-color: #ffc107; }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e !important;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #495057 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-pass { background: #d4edda; color: #155724; }
    .badge-fail { background: #f8d7da; color: #721c24; }

    /* Ticker highlight */
    .ticker-highlight {
        background: #e3f2fd;
        padding: 0.2em 0.5em;
        border-radius: 4px;
        font-weight: 700;
        font-family: monospace;
    }

    /* Pipeline step */
    .pipeline-step {
        background: #e8edf3;
        border: 1px solid #c5cdd8;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.3rem;
    }
    .pipeline-step h4 { margin: 0; color: #0f3460; }
    .pipeline-step p { margin: 0.3rem 0 0; font-size: 0.85rem; color: #495057; }

    /* Hide default streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metric card styling */
    div[data-testid="stMetric"] {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 0.8rem;
        border-left: 3px solid #0f3460;
    }
    div[data-testid="stMetric"] label {
        color: #495057 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #495057 !important;
    }

    /* Force dark text on all custom HTML elements */
    .metric-card .metric-value { color: #1a1a2e !important; }
    .metric-card .metric-label { color: #495057 !important; }

    /* Ensure buttons have readable text */
    .stButton > button {
        color: #1a1a2e !important;
    }
    .stButton > button[kind="primary"] {
        color: white !important;
        background-color: #0f3460 !important;
        border-color: #0f3460 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #16213e !important;
        border-color: #16213e !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ────────────────────────────────────────────────────────

def check_api_key():
    """Check if API key is configured."""
    placeholder_keys = {"REPLACE_WITH_YOUR_KEY", "your_api_key_here", ""}
    return FMP_API_KEY not in placeholder_keys


@st.cache_resource
def get_client(use_cache=True):
    """Get a cached FMP client instance."""
    return FMPClient(use_cache=use_cache)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_sp500_tickers():
    return get_sp500_tickers()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_sp400_tickers():
    return get_sp400_tickers()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_events(min_year=2016):
    return load_events(min_year=min_year)


def format_market_cap(val):
    """Format market cap as human-readable string."""
    if pd.isna(val) or val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.1f}B"
    if val >= 1e6:
        return f"${val/1e6:.0f}M"
    return f"${val:,.0f}"


def render_header():
    """Render the main dashboard header."""
    st.markdown("""
    <div class="main-header">
        <h1>S&P 500 Shadow Committee</h1>
        <p>Quantitative Inclusion Predictor &mdash; Statistical Analysis + NLP Pipeline</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, card_class=""):
    """Render a styled metric card."""
    return f"""
    <div class="metric-card {card_class}">
        <p class="metric-label">{label}</p>
        <p class="metric-value">{value}</p>
    </div>
    """


def render_pipeline_funnel(stages, title="Pipeline Filter Funnel"):
    """
    Render a fancy horizontal funnel chart.

    stages: list of (label, count) tuples, e.g.
        [("Universe", 431), ("Market Cap", 431), ...]

    Produces the "fancy suggested" style: horizontal bars with percentage
    circles, gradient coloring dark-to-light, value labels on bars.
    """
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    top_val = values[0] if values[0] > 0 else 1
    pcts = [v / top_val * 100 for v in values]

    # Color gradient: dark navy → teal → light cyan
    n = len(stages)
    colors = [
        f"rgb({int(10 + (100 - 10) * i / max(n-1, 1))}, "
        f"{int(40 + (180 - 40) * i / max(n-1, 1))}, "
        f"{int(80 + (190 - 80) * i / max(n-1, 1))})"
        for i in range(n)
    ]
    # Lighter version for background watermark bars
    bg_colors = [
        f"rgba({int(10 + (100 - 10) * i / max(n-1, 1))}, "
        f"{int(40 + (180 - 40) * i / max(n-1, 1))}, "
        f"{int(80 + (190 - 80) * i / max(n-1, 1))}, 0.15)"
        for i in range(n)
    ]

    fig = go.Figure()

    # Background bars (full width, faded) for context
    fig.add_trace(go.Bar(
        y=labels,
        x=[top_val] * n,
        orientation='h',
        marker=dict(color=bg_colors),
        hoverinfo='skip',
        showlegend=False,
    ))

    # Foreground bars (actual values)
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1),
        ),
        text=[f"<b>{v:,}</b>" for v in values],
        textposition='inside',
        textfont=dict(color='white', size=14),
        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        showlegend=False,
    ))

    # Add percentage annotations as circles on the left
    for i, (label, val, pct) in enumerate(zip(labels, values, pcts)):
        # Percentage circle
        fig.add_annotation(
            x=-top_val * 0.08,
            y=label,
            text=f"<b>{pct:.1f}%</b>",
            showarrow=False,
            font=dict(size=11, color='white'),
            bgcolor=colors[i],
            bordercolor=colors[i],
            borderwidth=1,
            borderpad=6,
            opacity=0.95,
        )
        # Value label on right side
        fig.add_annotation(
            x=top_val * 1.02,
            y=label,
            text=f"<b>{val:,}</b>",
            showarrow=False,
            font=dict(size=13, color=colors[i]),
            xanchor='left',
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1a1a2e')),
        barmode='overlay',
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-top_val * 0.18, top_val * 1.18],
        ),
        yaxis=dict(
            autorange='reversed',
            showgrid=False,
            tickfont=dict(size=13, color='#1a1a2e'),
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=max(55 * n + 80, 300),
        margin=dict(l=20, r=80, t=50, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render sidebar navigation and settings."""
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Select Page",
            [
                "Dashboard",
                "Live Predictions",
                "Ticker Lookup",
                "Watchlist",
                "Removal Simulator",
                "Backtest",
                "Sentiment Analysis",
                "Sector Analysis",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### Settings")

        top_n = st.slider("Top N Candidates", 5, 30, TOP_N_CANDIDATES)
        skip_prof = st.checkbox("Skip Profitability Filter", value=False,
                                help="Faster but less accurate — skips GAAP profitability API calls")
        no_cache = st.checkbox("Bypass Cache", value=False,
                               help="Force fresh API calls (slower)")

        st.markdown("---")
        st.markdown("### Model Weights")
        st.caption("Current scoring configuration:")
        weight_data = {
            "Component": ["Sector Gap", "MidCap 400", "Market Cap", "Profitability", "Sentiment (NLP)"],
            "Weight": [SECTOR_GAP_WEIGHT, MIDCAP_400_PREMIUM, MARKET_CAP_WEIGHT,
                       PROFITABILITY_MARGIN_WEIGHT, SENTIMENT_WEIGHT],
        }
        st.dataframe(pd.DataFrame(weight_data), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#888; font-size:0.8rem;'>"
            "RE: Data &mdash; Shadow Committee v1.0<br>"
            "Shreshth Sharma & Rakshaa Jeyarajah"
            "</div>",
            unsafe_allow_html=True,
        )

    return page, top_n, skip_prof, no_cache


# ─── Pages ───────────────────────────────────────────────────────────────────

def page_dashboard():
    """Main dashboard overview page."""
    render_header()

    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)

    sp500 = cached_sp500_tickers()
    sp400 = cached_sp400_tickers()
    events = cached_load_events(min_year=2020)

    with col1:
        st.metric("S&P 500 Members", len(sp500))
    with col2:
        st.metric("S&P 400 Members", len(sp400))
    with col3:
        st.metric("Historical Events (2020+)", len(events))
    with col4:
        st.metric("Min Market Cap", format_market_cap(MIN_MARKET_CAP))

    st.markdown("---")

    # Pipeline funnel with example numbers from a typical run
    st.subheader("Pipeline Filter Funnel")
    st.caption("Example from a typical pipeline run — each stage narrows the candidate pool")

    example_stages = [
        ("FMP Screener (US > $22.7B)", 431),
        ("Data Quality Filter", 392),
        ("Exclude Current S&P 500", 53),
        ("GAAP Profitability", 34),
        ("Liquidity & Float (FALR)", 23),
        ("Scored & Ranked", 23),
        ("Top 10 Predictions", 10),
    ]
    render_pipeline_funnel(example_stages, title="Candidate Selection Funnel")

    st.markdown("---")

    # Two-column layout: recent events + model info
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Recent S&P 500 Changes")
        if not events.empty:
            recent = events.tail(15).iloc[::-1].copy()
            recent = recent[["effective_date", "added_ticker", "added_name",
                            "removed_ticker", "removed_name", "reason"]].copy()
            recent.columns = ["Date", "Added", "Company Added",
                             "Removed", "Company Removed", "Reason"]
            st.dataframe(recent, hide_index=True, use_container_width=True, height=400)
        else:
            st.info("No event data available.")

    with right:
        st.subheader("Model Architecture")
        st.markdown("""
        **Hard Filters (Pass/Fail)**
        - Market Cap > $22.7B
        - US-domiciled company
        - GAAP Profitability (Q1 > 0, TTM > 0)
        - Float-Adjusted Liquidity Ratio >= 0.75
        - Not already in S&P 500

        **Soft Scoring (Weighted)**
        - Sector underrepresentation vs broad market
        - S&P MidCap 400 membership premium
        - Market capitalization (larger = higher)
        - TTM net margin quality
        - FinBERT news sentiment (live pipeline only)

        **Backtest Performance**
        - 70% Top-10 Hit Rate (10 events tested)
        - 100% Hit Rate among ranked candidates
        - Mean Rank: 5.6 / Median: 6.0
        """)


def page_live_predictions(top_n, skip_prof, no_cache):
    """Run the live prediction pipeline."""
    st.header("Live Predictions")
    st.caption("Run the full pipeline to generate current S&P 500 inclusion predictions.")

    use_sentiment = st.checkbox("Include NLP Sentiment Scoring", value=True,
                                help="Uses FinBERT on recent news headlines (adds ~2 min)")

    if st.button("Run Prediction Pipeline", type="primary", use_container_width=True):
        client = get_client(use_cache=not no_cache)

        from filters import (
            apply_market_cap_filter, apply_domicile_filter,
            apply_data_quality_filter, exclude_current_sp500,
            apply_profitability_filter, apply_liquidity_filter,
        )
        from data_sources import get_large_cap_universe

        with st.status("Running prediction pipeline...", expanded=True) as status:
            funnel_stages = []

            st.write("Fetching universe from FMP screener...")
            df = get_large_cap_universe(client)
            funnel_stages.append(("FMP Screener (US > $22.7B)", len(df)))

            st.write("Applying market cap filter...")
            df = apply_market_cap_filter(df)
            funnel_stages.append(("Market Cap Filter", len(df)))

            st.write("Applying domicile filter...")
            df = apply_domicile_filter(df)

            st.write("Applying data quality filter...")
            df = apply_data_quality_filter(df)
            funnel_stages.append(("Data Quality Filter", len(df)))

            st.write("Excluding current S&P 500...")
            df = exclude_current_sp500(df)
            funnel_stages.append(("Exclude S&P 500", len(df)))

            if not skip_prof:
                st.write("Checking GAAP profitability (this takes a while)...")
                df = apply_profitability_filter(df, client=client)
                funnel_stages.append(("GAAP Profitability", len(df)))

            if not df.empty:
                st.write("Checking liquidity & float...")
                df = apply_liquidity_filter(df, client=client)
                funnel_stages.append(("Liquidity & Float", len(df)))

            candidates = df

            if candidates.empty:
                st.error("No eligible candidates found. Check API key and connectivity.")
                st.session_state["funnel_stages"] = funnel_stages
                return

            st.write(f"Scoring {len(candidates)} candidates...")
            ranked = score_candidates(candidates, client=client, use_sentiment=use_sentiment)
            top = ranked.head(top_n)
            funnel_stages.append((f"Top {top_n} Predictions", len(top)))

            status.update(label="Pipeline complete!", state="complete")

        # Store in session state for persistence
        st.session_state["predictions"] = ranked
        st.session_state["top_predictions"] = top
        st.session_state["prediction_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["funnel_stages"] = funnel_stages

    # Display funnel even if pipeline returned empty
    if "funnel_stages" in st.session_state:
        render_pipeline_funnel(
            st.session_state["funnel_stages"],
            title="Live Pipeline Filter Funnel",
        )

    # Display results if available
    if "top_predictions" in st.session_state:
        top = st.session_state["top_predictions"]
        ranked = st.session_state["predictions"]

        st.success(f"Predictions generated at {st.session_state['prediction_time']}")

        # Top candidate spotlight
        if not top.empty:
            winner = top.iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Top Prediction", winner["symbol"],
                          delta=f"Score: {winner['total_score']:.1f}")
            with col2:
                st.metric("Company", str(winner.get("companyName", ""))[:30])
            with col3:
                st.metric("Market Cap", format_market_cap(winner.get("marketCap")))

        st.markdown("---")

        # Results table
        st.subheader(f"Top {len(top)} Candidates")
        display_cols = ["symbol", "companyName", "sector", "marketCap"]
        score_cols = ["sector_gap_score", "midcap_premium", "market_cap_score",
                      "profitability_score"]
        if "sentiment_score" in top.columns:
            score_cols.append("sentiment_score")
        score_cols.append("total_score")

        display = top[display_cols + score_cols].copy()
        display["marketCap"] = display["marketCap"].apply(format_market_cap)
        for col in score_cols:
            display[col] = display[col].round(2)
        display.index = range(1, len(display) + 1)
        display.index.name = "Rank"

        st.dataframe(display, use_container_width=True)

        # Score breakdown chart
        st.subheader("Score Breakdown")
        chart_data = top.head(10)[["symbol"] + score_cols[:-1]].copy()
        chart_melted = chart_data.melt(id_vars="symbol", var_name="Component", value_name="Score")
        fig = px.bar(
            chart_melted, x="symbol", y="Score", color="Component",
            title="Score Components by Candidate",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(xaxis_title="Ticker", yaxis_title="Score Points",
                          legend_title="Scoring Component")
        st.plotly_chart(fig, use_container_width=True)

        # Sector distribution
        if "sector" in top.columns:
            col1, col2 = st.columns(2)
            with col1:
                sector_counts = top["sector"].value_counts()
                fig_sector = px.pie(
                    values=sector_counts.values,
                    names=sector_counts.index,
                    title="Candidate Sector Distribution",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                st.plotly_chart(fig_sector, use_container_width=True)

            with col2:
                if "marketCap" in ranked.columns:
                    cap_data = top[["symbol", "marketCap"]].copy()
                    cap_data["marketCap_B"] = pd.to_numeric(cap_data["marketCap"], errors="coerce") / 1e9
                    fig_cap = px.bar(
                        cap_data, x="symbol", y="marketCap_B",
                        title="Market Capitalization ($B)",
                        color="marketCap_B",
                        color_continuous_scale="Blues",
                    )
                    fig_cap.update_layout(xaxis_title="Ticker", yaxis_title="Market Cap ($B)",
                                          showlegend=False)
                    st.plotly_chart(fig_cap, use_container_width=True)


def page_ticker_lookup(skip_prof, no_cache):
    """Score a single ticker."""
    st.header("Ticker Lookup")
    st.caption("Evaluate any ticker through the S&P 500 eligibility filters and scoring model.")

    ticker = st.text_input("Enter Ticker Symbol", placeholder="e.g., PLTR, CRWD, DASH",
                           max_chars=10)

    if ticker and st.button("Evaluate Ticker", type="primary"):
        ticker = ticker.strip().upper()
        client = get_client(use_cache=not no_cache)

        from data_sources import get_large_cap_universe
        from filters import (
            apply_data_quality_filter, exclude_current_sp500,
            apply_profitability_filter, apply_liquidity_filter,
        )
        from pipeline import Pipeline

        with st.status(f"Evaluating {ticker}...", expanded=True) as status:
            # Step 1: Find the ticker
            st.write("Looking up ticker...")
            universe = get_large_cap_universe(client)
            row_df = pd.DataFrame()
            if not universe.empty and "symbol" in universe.columns:
                row_df = universe[universe["symbol"].astype(str).str.upper() == ticker].head(1).copy()

            if row_df.empty:
                profile = client.company_profile(ticker)
                if profile:
                    row_df = pd.DataFrame([Pipeline._profile_to_candidate_row(profile)])

            if row_df.empty:
                status.update(label="Ticker not found", state="error")
                st.error(f"{ticker}: Ticker not found in current data sources.")
                return

            # Step 2: Run hard filters on just this one ticker
            st.write("Running hard filters...")
            row = row_df.iloc[0]
            hard_filters = {
                "market_cap": (pd.notna(row.get("marketCap")) and float(row.get("marketCap")) >= MIN_MARKET_CAP),
                "domicile": (str(row.get("country", "")).strip() in {"United States", "US", "USA"}),
                "data_quality": not apply_data_quality_filter(row_df).empty,
                "not_in_sp500": not exclude_current_sp500(row_df).empty,
            }

            if skip_prof:
                hard_filters["profitability"] = True
                prof_df = row_df.copy()
                prof_df["q1_net_income"] = None
                prof_df["ttm_net_income"] = None
                prof_df["ttm_revenue"] = None
                prof_df["ttm_net_margin"] = None
            else:
                st.write("Checking GAAP profitability...")
                prof_df = apply_profitability_filter(row_df, client=client)
                hard_filters["profitability"] = not prof_df.empty

            if hard_filters["profitability"]:
                st.write("Checking liquidity & float...")
                liq_df = apply_liquidity_filter(prof_df, client=client)
                hard_filters["liquidity"] = not liq_df.empty
                if not liq_df.empty:
                    prof_df = liq_df
            else:
                hard_filters["liquidity"] = False

            eligible = all(hard_filters.values())
            candidate_row = prof_df if hard_filters["profitability"] else row_df

            # Step 3: If eligible, score just this ticker (no full universe run)
            scored_row = None
            rank_info = None
            if eligible and not candidate_row.empty:
                st.write("Scoring candidate...")
                scored = score_candidates(candidate_row, client=client, use_sentiment=False)
                if not scored.empty:
                    scored_row = scored.iloc[0]

                # Check if we have cached predictions for rank context
                if "predictions" in st.session_state:
                    cached_ranked = st.session_state["predictions"]
                    hit = cached_ranked[cached_ranked["symbol"].astype(str).str.upper() == ticker]
                    if not hit.empty:
                        rank_info = (int(hit.index[0]) + 1, len(cached_ranked))

            status.update(label="Evaluation complete!", state="complete")

        # Store result
        st.session_state["ticker_result"] = {
            "ticker": ticker,
            "hard_filters": hard_filters,
            "eligible": eligible,
            "candidate_row": candidate_row,
            "scored_row": scored_row,
            "rank_info": rank_info,
        }

    # Display results
    if "ticker_result" not in st.session_state:
        return

    result = st.session_state["ticker_result"]
    ticker = result["ticker"]
    hf = result["hard_filters"]

    st.subheader(f"Eligibility Report: {ticker}")

    cols = st.columns(len(hf))
    for col, (filter_name, passed) in zip(cols, hf.items()):
        with col:
            label = filter_name.replace("_", " ").title()
            if passed:
                st.markdown(f"**{label}**")
                st.success("PASS")
            else:
                st.markdown(f"**{label}**")
                st.error("FAIL")

    if not result["eligible"]:
        st.warning("Not eligible: Did not pass all hard filters.")
        candidate_row = result["candidate_row"]
        if candidate_row is not None and not candidate_row.empty:
            st.subheader("Ticker Snapshot")
            st.dataframe(candidate_row, use_container_width=True)
        return

    # Eligible — show score
    rank_info = result["rank_info"]
    if rank_info:
        rank, universe_size = rank_info
        st.success(f"Eligible! Ranked **#{rank}** out of {universe_size} candidates "
                   f"(from last Live Predictions run).")
    else:
        st.success(f"Eligible! Run **Live Predictions** first to see rank among all candidates.")

    scored_row = result["scored_row"]
    if scored_row is not None:
        row = scored_row

        # Score gauges
        score_components = {
            "Sector Gap": ("sector_gap_score", SECTOR_GAP_WEIGHT),
            "MidCap 400": ("midcap_premium", MIDCAP_400_PREMIUM),
            "Market Cap": ("market_cap_score", MARKET_CAP_WEIGHT),
            "Profitability": ("profitability_score", PROFITABILITY_MARGIN_WEIGHT),
        }
        if "sentiment_score" in row.index:
            score_components["Sentiment"] = ("sentiment_score", SENTIMENT_WEIGHT)

        cols = st.columns(len(score_components) + 1)
        for col, (label, (col_name, max_val)) in zip(cols, score_components.items()):
            with col:
                val = float(row.get(col_name, 0))
                st.metric(label, f"{val:.1f}", delta=f"/ {max_val:.0f}")

        with cols[-1]:
            total = float(row.get("total_score", 0))
            max_total = sum(v for _, v in score_components.values())
            st.metric("Total Score", f"{total:.1f}", delta=f"/ {max_total:.0f}")

        # Radar chart
        categories = list(score_components.keys())
        values = [float(row.get(col_name, 0)) for col_name, _ in score_components.values()]
        max_values = [max_val for _, max_val in score_components.values()]
        normalized = [v / m * 100 if m > 0 else 0 for v, m in zip(values, max_values)]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=normalized + [normalized[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=ticker,
            fillcolor='rgba(15, 52, 96, 0.3)',
            line=dict(color='#0f3460'),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"{ticker} Score Profile",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def page_watchlist(no_cache):
    """Watchlist: bottom S&P 500 + at-risk members."""
    st.header("Watchlist")
    st.caption("Monitor S&P 500 members at risk of removal and top inclusion candidates.")

    if st.button("Load Watchlist Data", type="primary", use_container_width=True):
        client = get_client(use_cache=not no_cache)

        with st.status("Loading watchlist data...", expanded=True) as status:
            st.write("Fetching bottom S&P 500 members by market cap...")
            bottom = get_bottom_n_sp500(client, n=10)

            st.write("Checking for at-risk members (< $10B)...")
            at_risk = check_removal_candidates(client)

            status.update(label="Watchlist loaded!", state="complete")

        st.session_state["watchlist_bottom"] = bottom
        st.session_state["watchlist_at_risk"] = at_risk

    if "watchlist_bottom" in st.session_state:
        bottom = st.session_state["watchlist_bottom"]
        at_risk = st.session_state["watchlist_at_risk"]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bottom 10 by Market Cap (Removal Risk)")
            if not bottom.empty:
                display = bottom[["symbol", "companyName", "marketCap", "sector"]].copy()
                display["marketCap"] = display["marketCap"].apply(format_market_cap)
                display.index = range(1, len(display) + 1)
                st.dataframe(display, use_container_width=True)

                # Chart
                chart_data = bottom[["symbol", "marketCap"]].copy()
                chart_data["marketCap_B"] = chart_data["marketCap"] / 1e9
                fig = px.bar(
                    chart_data, x="symbol", y="marketCap_B",
                    title="Smallest S&P 500 Members ($B)",
                    color="marketCap_B",
                    color_continuous_scale="Reds_r",
                )
                fig.add_hline(y=REMOVAL_RISK_CAP/1e9, line_dash="dash",
                              annotation_text=f"${REMOVAL_RISK_CAP/1e9:.0f}B Threshold")
                fig.update_layout(xaxis_title="", yaxis_title="Market Cap ($B)",
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available.")

        with col2:
            st.subheader("At-Risk Members (< $10B)")
            if not at_risk.empty:
                display = at_risk[["symbol", "companyName", "marketCap"]].copy()
                display["marketCap"] = display["marketCap"].apply(format_market_cap)
                st.dataframe(display, use_container_width=True)
                st.warning(f"{len(at_risk)} member(s) below the $10B removal threshold.")
            else:
                st.success("No S&P 500 members currently below the $10B threshold.")


def page_removal_sim(top_n, no_cache):
    """Simulate a removal and predict replacement."""
    st.header("Removal Simulator")
    st.caption("Simulate removing a company from the S&P 500 and predict who replaces them.")

    sp500 = cached_sp500_tickers()
    ticker_list = sorted(sp500["Symbol"].astype(str).str.upper().tolist())

    col1, col2 = st.columns([2, 1])
    with col1:
        removed = st.selectbox("Select company to remove", ticker_list,
                               index=None, placeholder="Choose a ticker...")
    with col2:
        sim_top_n = st.number_input("Show top N replacements", min_value=1,
                                     max_value=30, value=min(top_n, 10))

    if removed and st.button("Run Simulation", type="primary"):
        client = get_client(use_cache=not no_cache)

        with st.status(f"Simulating removal of {removed}...", expanded=True) as status:
            # Get profile of removed company
            try:
                profile = client.company_profile(removed)
                if profile:
                    st.write(f"Removed: **{profile.get('companyName', removed)}** | "
                             f"Sector: {profile.get('sector', 'Unknown')} | "
                             f"Market Cap: {format_market_cap(profile.get('mktCap', profile.get('marketCap')))}")
            except Exception:
                pass

            st.write("Running prediction pipeline...")
            candidates = get_eligible_universe(client=client, skip_profitability=True)
            if not candidates.empty:
                ranked = score_candidates(candidates, client=client, use_sentiment=False)
                top = ranked.head(sim_top_n)
            else:
                top = pd.DataFrame()

            status.update(label="Simulation complete!", state="complete")

        if not top.empty:
            winner = top.iloc[0]
            st.success(f"Top predicted replacement: **{winner['symbol']}** "
                       f"({winner.get('companyName', '')}) — Score: {winner['total_score']:.1f}")

            display_cols = ["symbol", "companyName", "sector", "marketCap",
                           "sector_gap_score", "midcap_premium", "market_cap_score",
                           "profitability_score", "total_score"]
            display = top[[c for c in display_cols if c in top.columns]].copy()
            if "marketCap" in display.columns:
                display["marketCap"] = display["marketCap"].apply(format_market_cap)
            display.index = range(1, len(display) + 1)
            display.index.name = "Rank"
            st.dataframe(display, use_container_width=True)
        else:
            st.error("No candidates found.")


def page_backtest(top_n, no_cache):
    """Walk-forward backtesting page."""
    st.header("Backtest")
    st.caption("Replay historical S&P 500 events through the prediction pipeline.")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_year = st.number_input("Min Year", min_value=2016, max_value=2026,
                                    value=2020)
    with col2:
        max_events = st.number_input("Max Events (0 = all)", min_value=0,
                                      max_value=200, value=10)
    with col3:
        skip_liq = st.checkbox("Skip Liquidity Filter", value=True,
                                help="Faster backtesting")

    # Show available events
    with st.expander("Preview Historical Events"):
        events = cached_load_events(min_year=min_year)
        if not events.empty:
            st.dataframe(events[["effective_date", "added_ticker", "added_name",
                                "removed_ticker", "reason", "confidence"]],
                        hide_index=True, use_container_width=True, height=300)
            st.caption(f"{len(events)} events from {min_year} onward")
        else:
            st.info("No events found for this date range.")

    if st.button("Run Backtest", type="primary", use_container_width=True):
        client = get_client(use_cache=not no_cache)
        effective_max = max_events if max_events > 0 else None

        progress_text = st.empty()
        progress_bar = st.progress(0)

        with st.spinner("Running walk-forward backtest..."):
            results = run_backtest(
                client=client,
                min_year=min_year,
                max_events=effective_max,
                top_n=top_n,
                skip_liquidity=skip_liq,
                show_progress=False,
            )

        progress_bar.empty()
        progress_text.empty()

        if not results:
            st.error("No backtest results generated.")
            return

        st.session_state["backtest_results"] = results
        st.session_state["backtest_top_n"] = top_n

    # Display results
    if "backtest_results" in st.session_state:
        results = st.session_state["backtest_results"]
        bt_top_n = st.session_state["backtest_top_n"]
        metrics = compute_metrics(results, top_n=bt_top_n)

        # Summary metrics
        st.subheader("Summary Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Events", metrics["total_events"])
        with c2:
            st.metric("Events With Prediction", metrics.get("events_with_rank", 0))
        with c3:
            hit_rate = metrics.get("hit_rate_top_n", 0)
            st.metric(f"Hit Rate (Top {bt_top_n})", f"{hit_rate}%")
        with c4:
            mean_rank = metrics.get("mean_rank")
            st.metric("Mean Rank", str(mean_rank) if mean_rank else "N/A")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Hit Rate (Top 1)", f"{metrics.get('hit_rate_top_1', 0)}%")
        with c6:
            ranked_hr = metrics.get("hit_rate_top_n_of_ranked")
            st.metric(f"Hit Rate (of ranked)", f"{ranked_hr}%" if ranked_hr else "N/A")
        with c7:
            st.metric("Median Rank", str(metrics.get("median_rank", "N/A")))
        with c8:
            st.metric("Events Not Found", metrics.get("events_not_found", 0))

        st.markdown("---")

        # Charts
        df = pd.DataFrame(results)

        col_left, col_right = st.columns(2)

        with col_left:
            # Hit rate visualization
            hits = int(df["hit_top_n"].sum()) if "hit_top_n" in df.columns else 0
            misses = len(df) - hits
            fig_hits = go.Figure(data=[go.Pie(
                labels=["Hit (Top N)", "Miss"],
                values=[hits, misses],
                hole=0.5,
                marker_colors=["#28a745", "#dc3545"],
            )])
            fig_hits.update_layout(
                title=f"Top-{bt_top_n} Hit Rate",
                annotations=[dict(text=f"{hit_rate}%", x=0.5, y=0.5,
                                  font_size=24, showarrow=False)],
            )
            st.plotly_chart(fig_hits, use_container_width=True)

        with col_right:
            # Rank distribution
            ranks = df["prediction_rank"].dropna()
            if not ranks.empty:
                fig_rank = px.histogram(
                    ranks, nbins=max(int(ranks.max()), 10),
                    title="Prediction Rank Distribution",
                    labels={"value": "Rank", "count": "Count"},
                    color_discrete_sequence=["#0f3460"],
                )
                fig_rank.add_vline(x=bt_top_n, line_dash="dash", line_color="red",
                                   annotation_text=f"Top {bt_top_n} cutoff")
                fig_rank.update_layout(xaxis_title="Prediction Rank",
                                       yaxis_title="Number of Events",
                                       showlegend=False)
                st.plotly_chart(fig_rank, use_container_width=True)
            else:
                st.info("No ranked predictions to chart.")

        # Event detail table
        st.subheader("Event Details")
        detail_cols = ["effective_date", "added_ticker", "prediction_rank",
                      "hit_top_n", "top_predicted", "event_confidence",
                      "candidates_ranked", "removed_ticker", "reason"]
        available = [c for c in detail_cols if c in df.columns]
        detail_df = df[available].copy()
        detail_df["hit_top_n"] = detail_df["hit_top_n"].map({True: "YES", False: "no"})
        detail_df.index = range(1, len(detail_df) + 1)
        detail_df.index.name = "#"
        st.dataframe(detail_df, use_container_width=True)

        # Confidence tier breakdown
        by_conf = metrics.get("by_confidence", {})
        if by_conf:
            st.subheader("By Event Confidence Tier")
            conf_rows = []
            for tier in ["A", "B", "C"]:
                if tier in by_conf:
                    t = by_conf[tier]
                    conf_rows.append({
                        "Tier": tier,
                        "Events": t["total"],
                        "Ranked": t["ranked"],
                        f"Hit Rate (Top {bt_top_n})": f"{t['hit_rate_top_n']}%",
                        "Mean Rank": t["mean_rank"] or "-",
                    })
            if conf_rows:
                st.dataframe(pd.DataFrame(conf_rows), hide_index=True,
                             use_container_width=True)


def page_sentiment(no_cache):
    """Sentiment analysis page."""
    st.header("Sentiment Analysis")
    st.caption("Analyze financial news sentiment using FinBERT NLP model.")

    st.info("This page uses the ProsusAI/FinBERT model to classify financial news sentiment. "
            "Headlines are fetched from Google News RSS in real-time.")

    ticker_input = st.text_input("Enter ticker(s) — comma separated",
                                 placeholder="e.g., PLTR, CRWD, DASH, UTHR")

    if ticker_input and st.button("Analyze Sentiment", type="primary"):
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

        from sentiment import fetch_headlines, score_headlines, _get_finbert

        all_results = []

        with st.status(f"Analyzing {len(tickers)} ticker(s)...", expanded=True) as status:
            st.write("Loading FinBERT model...")
            _get_finbert()  # Pre-load

            for ticker in tickers:
                st.write(f"Fetching headlines for {ticker}...")
                headlines = fetch_headlines(ticker, limit=15)

                if not headlines:
                    all_results.append({
                        "ticker": ticker,
                        "headlines_found": 0,
                        "net_sentiment": 0.0,
                        "classification": "Neutral",
                        "headlines": [],
                    })
                    continue

                score = score_headlines(headlines)
                classification = "Positive" if score > 0.1 else ("Negative" if score < -0.1 else "Neutral")

                all_results.append({
                    "ticker": ticker,
                    "headlines_found": len(headlines),
                    "net_sentiment": round(score, 4),
                    "classification": classification,
                    "headlines": headlines,
                })

            status.update(label="Sentiment analysis complete!", state="complete")

        st.session_state["sentiment_results"] = all_results

    if "sentiment_results" in st.session_state:
        results = st.session_state["sentiment_results"]

        # Summary chart
        summary_df = pd.DataFrame([{
            "Ticker": r["ticker"],
            "Sentiment": r["net_sentiment"],
            "Headlines": r["headlines_found"],
            "Classification": r["classification"],
        } for r in results])

        if not summary_df.empty:
            colors = summary_df["Sentiment"].apply(
                lambda x: "#28a745" if x > 0.1 else ("#dc3545" if x < -0.1 else "#ffc107")
            )
            fig = go.Figure(data=[
                go.Bar(
                    x=summary_df["Ticker"],
                    y=summary_df["Sentiment"],
                    marker_color=colors,
                    text=summary_df["Classification"],
                    textposition="outside",
                )
            ])
            fig.update_layout(
                title="Net Sentiment Score by Ticker",
                xaxis_title="Ticker",
                yaxis_title="Sentiment (-1 to +1)",
                yaxis=dict(range=[-1, 1]),
            )
            fig.add_hline(y=0, line_dash="solid", line_color="gray")
            fig.add_hline(y=0.1, line_dash="dash", line_color="#28a745", opacity=0.3)
            fig.add_hline(y=-0.1, line_dash="dash", line_color="#dc3545", opacity=0.3)
            st.plotly_chart(fig, use_container_width=True)

        # Detailed headlines per ticker
        for r in results:
            with st.expander(f"{r['ticker']} — {r['classification']} "
                             f"({r['net_sentiment']:+.3f}, {r['headlines_found']} headlines)"):
                if r["headlines"]:
                    for i, h in enumerate(r["headlines"], 1):
                        st.markdown(f"{i}. {h}")
                else:
                    st.write("No headlines found.")


def page_sector_analysis(no_cache):
    """Sector weight comparison: S&P 500 vs broad market."""
    st.header("Sector Analysis")
    st.caption("Compare S&P 500 sector weights against the broader US market to identify gaps.")

    if st.button("Load Sector Data", type="primary", use_container_width=True):
        client = get_client(use_cache=not no_cache)

        with st.status("Loading sector data...", expanded=True) as status:
            st.write("Computing S&P 500 sector weights...")
            sp500_weights = get_sp500_sector_weights(client)

            st.write("Computing broad market proxy weights...")
            from data_sources import get_us_market_proxy_sector_weights, get_sp500_sector_market_weights
            sp500_mkt = get_sp500_sector_market_weights(client)
            market_proxy = get_us_market_proxy_sector_weights(client)

            status.update(label="Sector data loaded!", state="complete")

        st.session_state["sector_sp500"] = sp500_weights
        st.session_state["sector_sp500_mkt"] = sp500_mkt
        st.session_state["sector_market_proxy"] = market_proxy

    if "sector_sp500" in st.session_state:
        sp500_weights = st.session_state["sector_sp500"]
        sp500_mkt = st.session_state["sector_sp500_mkt"]
        market_proxy = st.session_state["sector_market_proxy"]

        # S&P 500 sector breakdown table
        if not sp500_weights.empty:
            st.subheader("S&P 500 Sector Weights (by Market Cap)")
            st.dataframe(sp500_weights, hide_index=True, use_container_width=True)

            # Pie chart
            fig_pie = px.pie(
                sp500_weights, values="weight_pct", names="sector",
                title="S&P 500 Sector Composition",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Gap analysis
        if not sp500_mkt.empty and not market_proxy.empty:
            st.subheader("Sector Gap Analysis: S&P 500 vs Broad Market")
            st.caption("Positive gap = sector is underrepresented in S&P 500 (candidates from these sectors get bonus points)")

            all_sectors = sorted(set(sp500_mkt.index) | set(market_proxy.index))
            gap_data = []
            for sector in all_sectors:
                sp_w = float(sp500_mkt.get(sector, 0)) * 100
                mkt_w = float(market_proxy.get(sector, 0)) * 100
                gap_data.append({
                    "Sector": sector,
                    "S&P 500 Weight (%)": round(sp_w, 2),
                    "Market Weight (%)": round(mkt_w, 2),
                    "Gap (%)": round(mkt_w - sp_w, 2),
                })

            gap_df = pd.DataFrame(gap_data).sort_values("Gap (%)", ascending=False)
            st.dataframe(gap_df, hide_index=True, use_container_width=True)

            # Grouped bar chart
            fig_gap = go.Figure()
            fig_gap.add_trace(go.Bar(
                name="S&P 500",
                x=gap_df["Sector"],
                y=gap_df["S&P 500 Weight (%)"],
                marker_color="#0f3460",
            ))
            fig_gap.add_trace(go.Bar(
                name="Broad Market",
                x=gap_df["Sector"],
                y=gap_df["Market Weight (%)"],
                marker_color="#e94560",
            ))
            fig_gap.update_layout(
                title="S&P 500 vs Broad Market Sector Weights",
                barmode="group",
                xaxis_title="Sector",
                yaxis_title="Weight (%)",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_gap, use_container_width=True)

            # Gap direction chart
            colors = gap_df["Gap (%)"].apply(
                lambda x: "#28a745" if x > 0 else "#dc3545"
            )
            fig_gaps = go.Figure(data=[
                go.Bar(
                    x=gap_df["Sector"],
                    y=gap_df["Gap (%)"],
                    marker_color=colors,
                )
            ])
            fig_gaps.update_layout(
                title="Sector Gaps (Positive = Underrepresented in S&P 500)",
                xaxis_title="Sector",
                yaxis_title="Gap (%)",
                xaxis_tickangle=-45,
            )
            fig_gaps.add_hline(y=0, line_dash="solid", line_color="gray")
            st.plotly_chart(fig_gaps, use_container_width=True)


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    if not check_api_key():
        render_header()
        st.error(
            "No API key configured. Set your FMP API key in the `.env` file:\n\n"
            "```\nFMP_API_KEY=your_key_here\n```\n\n"
            "Get a free key at: https://financialmodelingprep.com/"
        )
        return

    page, top_n, skip_prof, no_cache = render_sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Live Predictions":
        page_live_predictions(top_n, skip_prof, no_cache)
    elif page == "Ticker Lookup":
        page_ticker_lookup(skip_prof, no_cache)
    elif page == "Watchlist":
        page_watchlist(no_cache)
    elif page == "Removal Simulator":
        page_removal_sim(top_n, no_cache)
    elif page == "Backtest":
        page_backtest(top_n, no_cache)
    elif page == "Sentiment Analysis":
        page_sentiment(no_cache)
    elif page == "Sector Analysis":
        page_sector_analysis(no_cache)


if __name__ == "__main__":
    main()
