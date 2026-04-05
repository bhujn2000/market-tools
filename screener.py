import yfinance as yf
import pandas as pd

tickers = tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", 
           "NVDA", "TSM", "BRK-B", "JPM", "V"]

rows = []

for ticker in tickers:
    stock = yf.Ticker(ticker)
    info = stock.info

    rows.append({
        "ticker": ticker,
        "price": info.get("currentPrice"),
        "market_cap_B": round(info.get("marketCap", 0) / 1e9, 1),
        "pe_ratio": info.get("trailingPE"),
        "fwd_pe": info.get("forwardPE"),
        "revenue_growth": info.get("revenueGrowth"),
        "gross_margin": info.get("grossMargins"),
        "free_cashflow_B": round(info.get("freeCashflow", 0) / 1e9, 1),
        "analyst_target": info.get("targetMeanPrice"),
    })

df = pd.DataFrame(rows)
df["upside_pct"] = ((df["analyst_target"] - df["price"]) / df["price"] * 100).round(1)
df = df.set_index("ticker")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.2f}".format)

print(df.to_string())

# sort by any column
df_sorted = df.sort_values("fwd_pe", ascending=True)
print("\n--- sorted by forward P/E (cheapest first) ---")
print(df_sorted.to_string())

# filter: only show stocks where revenue growth > 15%
df_filtered = df[df["revenue_growth"] > 0.15]
print("\n--- revenue growth > 15% ---")
print(df_filtered.to_string())

# filter: cheap and growing — fwd P/E under 25 AND revenue growth over 15%
df_value_growth = df[(df["fwd_pe"] < 25) & (df["revenue_growth"] > 0.15)]
df_value_growth = df_value_growth.sort_values("upside_pct", ascending=False)
print("\n--- value + growth, sorted by upside to analyst target ---")
print(df_value_growth.to_string())