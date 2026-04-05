import yfinance as yf
import plotly.graph_objects as go

tickers = ["SPY", "QQQ", "BTC-USD"]

fig = go.Figure()

for ticker in tickers:
    data = yf.download(ticker, period="1y", auto_adjust=True)
    normalized = data["Close"] / data["Close"].iloc[0] * 100
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized.squeeze(), name=ticker))

fig.update_layout(
    title="SPY vs QQQ vs BTC — normalized to 100 (1 year)",
    yaxis_title="Indexed price",
    xaxis_title="Date",
    hovermode="x unified"
)

fig.show()