Here is the complete Master Project Document for building your \*\*S\&P 500 Inclusion Predictor\*\* (what we'll call the "Shadow Committee" model). 



This outlines the full context, the architectural pipeline, and the step-by-step instructions to build a working prototype.



\---



\### Project Context \& Objective

\*\*The Goal:\*\* To build a quantitative data pipeline that reverse-engineers the S\&P Dow Jones Indices Committee's selection process. 

\*\*The "Why":\*\* When a company is added to the S\&P 500, trillions of dollars in passive index funds (like SPY and VOO) are forced to buy the stock. This creates an "inclusion pop" (a sudden surge in the stock price). If we can accurately predict the short-list of candidates \*before\* the announcement, we can theoretically front-run this institutional buying pressure.

\*\*The Challenge:\*\* The S\&P 500 is not purely math-based; it is subjective. Our model must combine strict quantitative filters (GAAP profitability, market cap) with qualitative approximations (sector balancing).



\---



\### Phase 1: Environment \& Tooling Setup

You will need a Python environment designed for financial data processing.



\*\*1. The Stack:\*\*

\* \*\*Language:\*\* Python 3.9+

\* \*\*Data Processing:\*\* `pandas`, `numpy`

\* \*\*Data Acquisition:\*\* `requests` (for REST APIs), `yfinance` (for quick price data)

\* \*\*Environment:\*\* Google Colab (recommended for instant setup) or a local VS Code environment.



\*\*2. The Data Provider:\*\*

For this build, we will target \*\*Financial Modeling Prep (FMP)\*\*. It offers a free tier that provides the necessary Market Cap and Income Statement data to build the prototype. 

\* \*Action:\* Go to financialmodelingprep.com, create a free account, and save your API Key.



\---



\### Phase 2: The Core Algorithm (The Pipeline)

The system operates like a funnel, taking the \~5,000 publicly traded U.S. stocks and filtering them down to the top 5 most likely candidates.



\#### Step 1: Define the Eligible Universe (The Hard Filters)

The code must strictly enforce the S\&P's published criteria. If a company fails these, the human committee cannot legally add them.



\* \*\*Filter A (Size):\*\* Current Market Capitalization > $22.7 Billion.

\* \*\*Filter B (Domicile):\*\* Must be a U.S.-headquartered company.

\* \*\*Filter C (Profitability):\*\* The most critical hurdle. The company must have positive GAAP Net Income in the most recent quarter, \*\*AND\*\* the sum of the last four quarters' Net Income must be positive. 



\#### Step 2: Score the Candidates (The Soft Filters)

Once you have the 50–100 companies that passed Step 1, you must rank them based on how the human committee thinks.



\* \*\*The Sector Gap Score:\*\* The committee wants the S\&P 500 to mirror the broader U.S. economy. Your script must compare the current sector weights of the S\&P 500 against a total market index. If the S\&P 500 is lacking in "Industrials," any Industrial stock on your shortlist gets a massive score multiplier.

\* \*\*The MidCap Premium:\*\* Companies already in the S\&P MidCap 400 index are historically heavily favored for promotion over outside stocks. 



\---



\### Phase 3: The Python Implementation

Here is AN EXAMPLE OF structural code to build the actual engine. You can copy this directly into your Python environment.



```python

import pandas as pd

import requests



\# 1. Configuration

API\_KEY = "YOUR\_FMP\_API\_KEY\_HERE"

MIN\_MARKET\_CAP = 22.7e9 # $22.7 Billion



def get\_us\_large\_caps():

&#x20;   """Fetches a list of US companies over the Market Cap threshold."""

&#x20;   url = f"https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan={MIN\_MARKET\_CAP}\&country=US\&apikey={API\_KEY}"

&#x20;   response = requests.get(url)

&#x20;   return pd.DataFrame(response.json())



def check\_profitability(ticker):

&#x20;   """Checks the S\&P GAAP Profitability Rule (Most recent Q > 0 AND TTM > 0)"""

&#x20;   url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter\&limit=4\&apikey={API\_KEY}"

&#x20;   try:

&#x20;       data = requests.get(url).json()

&#x20;       if len(data) < 4: return False # Not enough data

&#x20;       

&#x20;       # S\&P strictly uses GAAP Net Income

&#x20;       q1\_net\_income = data\[0]\['netIncome']

&#x20;       ttm\_net\_income = sum(\[q\['netIncome'] for q in data])

&#x20;       

&#x20;       return (q1\_net\_income > 0) and (ttm\_net\_income > 0)

&#x20;   except:

&#x20;       return False



def build\_shadow\_list():

&#x20;   print("Fetching US Large Caps...")

&#x20;   df = get\_us\_large\_caps()

&#x20;   tickers = df\['symbol'].tolist()

&#x20;   

&#x20;   candidates = \[]

&#x20;   print(f"Scanning {len(tickers)} companies for profitability...")

&#x20;   

&#x20;   # In a real build, add a time.sleep() here to respect API rate limits

&#x20;   for ticker in tickers\[:50]: # Testing with first 50 to save API calls

&#x20;       is\_profitable = check\_profitability(ticker)

&#x20;       if is\_profitable:

&#x20;           candidates.append(ticker)

&#x20;           print(f" \[PASS] {ticker} meets S\&P criteria.")

&#x20;           

&#x20;   return candidates



\# Run the pipeline

\# shortlist = build\_shadow\_list()

\# print("Final Shortlist:", shortlist)

```



\---



\### Phase 4: The Event Trigger (Execution)

Having the list is only half the battle; you must know \*when\* to check it. The S\&P Committee only adds a company when a seat opens up or during quarterly rebalances.



\*\*Your Action Plan for Execution:\*\*

1\.  \*\*Monitor M\&A:\*\* Set up a script or alert to monitor financial news. If a current S\&P 500 company announces it is being acquired (e.g., \*Pioneer Natural Resources being bought by Exxon\*), a seat will open up the day the deal closes.

2\.  \*\*Monitor the Bottom 10:\*\* Keep a live tracker of the 10 smallest companies currently in the S\&P 500. If their market cap drops significantly below $10 Billion, they are at high risk of being demoted.

3\.  \*\*The Strike:\*\* When a removal catalyst occurs, run your `build\_shadow\_list()` script immediately. The top-ranked company on that list is your predicted addition. 



\---



\### Phase 5: Backtesting \& Validation (The Next Horizon)

Before trading real capital, you must prove the model works. 



\* \*\*The Trap:\*\* Do not use standard historical data to backtest. Financial databases overwrite old data with modern corrections (Look-Ahead Bias). 

\* \*\*The Solution:\*\* You must eventually integrate a \*\*Point-in-Time (PIT)\*\* database (like Sharadar) that shows you exactly what the financial reports looked like \*on the day the committee made their decision three years ago\*.



\*\*\*



\*\*Next Step:\*\* To get this prototype breathing, we need to plug in a real data source. Have you decided if you want to use the free tier of an API like Financial Modeling Prep, or would you prefer a different route to fetch the market cap and income data?

