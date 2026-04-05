import yfinance as yf
import pandas as pd
import pandas_ta as ta
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN",
           "NVDA", "TSM", "BRK-B", "JPM", "V"]

rows = []

for ticker in tickers:
    data = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
    close = data["Close"].squeeze()
    
    rsi = ta.rsi(close, length=14)
    current_rsi = round(rsi.iloc[-1], 1)
    current_price = round(float(close.iloc[-1]), 2)
    change_1w = round((close.iloc[-1] / close.iloc[-5] - 1) * 100, 1)
    change_1m = round((close.iloc[-1] / close.iloc[-21] - 1) * 100, 1)
    signal = "oversold" if current_rsi < 30 else "overbought" if current_rsi > 70 else "neutral"

    rows.append({
        "ticker": ticker,
        "price": current_price,
        "rsi_14": current_rsi,
        "signal": signal,
        "1w_chg%": float(change_1w),
        "1m_chg%": float(change_1m),
    })

df = pd.DataFrame(rows).set_index("ticker")
df_sorted = df.sort_values("rsi_14", ascending=True)

print("--- market data fetched ---")
print(df_sorted.to_string())

data_summary = df_sorted.to_string()

prompt = f"""
You are a concise market analyst. I will give you RSI and price 
change data for a watchlist of stocks.

Analyze this data and return ONLY a JSON object:

{{
  "market_mood": "risk-on / risk-off / mixed",
  "key_observation": "",
  "watchlist": [
    {{
      "ticker": "",
      "rsi": 0,
      "stance": "bullish / bearish / neutral",
      "one_line": ""
    }}
  ],
  "top_opportunity": "",
  "top_risk": ""
}}

Rules:
- one_line max 15 words
- key_observation max 25 words
- top_opportunity and top_risk max 20 words each
- rank watchlist by conviction, highest first

Data:
{data_summary}
"""

print("\n--- sending to Claude ---")

message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

response = message.content[0].text
print("\n--- morning brief ---")
print(response)

with open("morning_brief.txt", "w") as f:
    f.write(response)

print("\nsaved to morning_brief.txt")

import json
import re

clean = re.sub(r"```json|```", "", response).strip()
parsed = json.loads(clean)

print("\n--- top opportunity ---")
print(parsed["top_opportunity"])

print("\n--- top risk ---")
print(parsed["top_risk"])