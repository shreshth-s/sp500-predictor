# S&P 500 Inclusion Predictor — "Shadow Committee" Model

A quantitative pipeline that reverse-engineers the S&P Dow Jones Indices Committee's selection process to predict the next companies added to the S&P 500 — before the announcement.

## The Core Idea

When a company is added to the S&P 500, over **$16 trillion** in passive index funds (SPY, VOO, etc.) are contractually obligated to buy its shares. This creates a predictable **3–7% "inclusion pop."** If you can identify the short-list of candidates _before_ the announcement, you can position ahead of that institutional buying pressure.

## How It Works

The pipeline operates as a two-phase funnel:

**Phase 1 — Hard Filters** (S&P's published eligibility criteria):
| Filter | Threshold |
|--------|-----------|
| Market Cap | ≥ $22.7 billion |
| Domicile | US-headquartered only |
| GAAP Profitability | Most recent quarter > 0 **AND** trailing 4 quarters > 0 |
| Liquidity (FALR) | Float-Adjusted Liquidity Ratio ≥ 0.75 |
| Not already in S&P 500 | Excluded from candidate pool |

**Phase 2 — Soft Scoring** (approximating how the committee thinks):
| Component | Max Points | Logic |
|-----------|-----------|-------|
| Sector Gap | 20 | Underrepresented sectors vs. total US market get a bonus |
| MidCap 400 Premium | 15 | Flat bonus — S&P 400 members are historically favored for promotion |
| Market Cap | 10 | Min-max normalized: larger cap = higher score |
| Profitability Margin | 15 | TTM net margin, min-max normalized |
| NLP Sentiment | 10 | FinBERT sentiment on recent news headlines |

## Sample Output (March 2026)

| Rank | Ticker | Company | Score |
|------|--------|---------|-------|
| 1 | UTHR | United Therapeutics | 41.78 |
| 2 | MPLX | MPLX LP | 37.40 |
| 3 | LNG | Cheniere Energy | 32.98 |
| 4 | EPD | Enterprise Products | 28.82 |
| 5 | VG | Vonage / Telecom | 27.48 |

Full ranked report: [`reports/SP500_Predictor_Report.html`](reports/SP500_Predictor_Report.html)

## Setup

```bash
git clone https://github.com/your-username/sp500-predictor.git
cd sp500-predictor
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your FMP API key (free tier works for basic runs)
```

Get a free API key at [financialmodelingprep.com](https://financialmodelingprep.com).

## Usage

**CLI:**
```bash
# Full pipeline — top 10 predictions
python cli.py run --top 10 --export

# Skip slow profitability API calls (faster, less accurate)
python cli.py run --skip-profitability

# Monitor bottom of S&P 500 (removal watch)
python cli.py watchlist

# Simulate a removal and predict the replacement
python cli.py simulate AAPL

# Clear cache
python cli.py clear-cache
```

**Streamlit Dashboard:**
```bash
streamlit run app.py
```

The dashboard provides:
- Live predictions with pipeline funnel visualization
- Per-ticker eligibility lookup
- Sector gap analysis
- News sentiment (FinBERT NLP)
- Backtesting results
- Removal scenario simulator

## Project Structure

```
├── config.py              # All tunable constants and scoring weights
├── fmp_client.py          # FMP API wrapper with rate limiting + disk cache
├── filters.py             # Hard filter implementations
├── scoring.py             # Soft scoring components
├── pipeline.py            # Orchestrator: filters → scoring → output
├── data_sources.py        # S&P 500/400 constituent fetching
├── monitor.py             # Bottom-of-index watchlist
├── sentiment.py           # FinBERT NLP sentiment scoring
├── cli.py                 # argparse CLI entry point
├── app.py                 # Streamlit web dashboard
├── backtest/              # Walk-forward backtesting module
│   ├── runner.py          # Event-driven backtest runner
│   ├── snapshot.py        # Point-in-time data reconstruction
│   ├── events.py          # Historical S&P 500 addition events
│   └── metrics.py         # Hit-rate and accuracy metrics
├── generate_report_html.py # HTML report generator
├── reports/               # Latest generated report (tracked)
├── cache/                 # API response cache (gitignored)
└── output/                # Run outputs (gitignored)
```

## Data Sources

| Source | What it provides |
|--------|-----------------|
| [FMP Stable API](https://financialmodelingprep.com/stable) | Market cap, income statements, price history, float data |
| [Wikipedia — S&P 500](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) | Current constituents + historical changes |
| [Wikipedia — S&P 400](https://en.wikipedia.org/wiki/List_of_S%26P_400_companies) | MidCap 400 membership |
| [Google News RSS](https://news.google.com/rss/search) | News headlines for sentiment analysis |
| [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) | Financial NLP sentiment model |

## Backtesting

The model uses **walk-forward, point-in-time backtesting** (2020–present) against confirmed S&P 500 addition events sourced from Wikipedia's change log. Each event is evaluated using only data that was publicly available _before_ the announcement date to avoid look-ahead bias.

```bash
python cli.py backtest --min-year 2020
```

## Disclaimer

This tool is for **research and educational purposes only.** Nothing here constitutes financial advice or a recommendation to buy or sell any security. Past model performance does not guarantee future results. Always do your own due diligence.

## License

MIT
