import yfinance as yf
import pandas as pd
import pandas_ta as ta

tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN",
           "NVDA", "TSM", "BRK-B", "JPM", "V"]

rows = []

for ticker in tickers:
    data = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
    
    close = data["Close"].squeeze()
    
    rsi = ta.rsi(close, length=14)
    current_rsi = round(rsi.iloc[-1], 1)
    
    current_price = round(close.iloc[-1], 2)
    change_1m = round((close.iloc[-1] / close.iloc[-21] - 1) * 100, 1)
    change_1w = round((close.iloc[-1] / close.iloc[-5] - 1) * 100, 1)
    
    signal = "oversold" if current_rsi < 30 else "overbought" if current_rsi > 70 else "neutral"
    
    rows.append({
        "ticker": ticker,
        "price": current_price,
        "rsi_14": current_rsi,
        "signal": signal,
        "1w_chg%": change_1w,
        "1m_chg%": change_1m,
    })

df = pd.DataFrame(rows).set_index("ticker")
df = df.sort_values("rsi_14", ascending=True)

print(df.to_string())

print("\n--- oversold (RSI < 30) ---")
oversold = df[df["rsi_14"] < 30]
print(oversold.to_string() if not oversold.empty else "none in watchlist")

print("\n--- overbought (RSI > 70) ---")
overbought = df[df["rsi_14"] > 70]
print(overbought.to_string() if not overbought.empty else "none in watchlist")

df.to_csv("rsi_output.csv")
print("\nsaved to rsi_output.csv")