import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import anthropic
import json
import re
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Morning Brief", layout="wide")
st.title("Morning Brief")
st.caption("RSI signals + Claude analysis for your watchlist")

with st.sidebar:
    st.header("Watchlist")
    default_tickers = "AAPL, MSFT, GOOGL, META, AMZN, NVDA, TSM, BRK-B, JPM, V"
    ticker_input = st.text_area("Tickers (comma separated)", value=default_tickers, height=150)
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)

if run_button:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    with st.spinner("Fetching market data..."):
        rows = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                data = stock.history(period="3mo", auto_adjust=True)
                close = data["Close"].squeeze()
                rsi = ta.rsi(close, length=14)
                current_rsi = round(rsi.iloc[-1], 1)
                current_price = round(float(close.iloc[-1]), 2)
                change_1w = round((close.iloc[-1] / close.iloc[-5] - 1) * 100, 1)
                change_1m = round((close.iloc[-1] / close.iloc[-21] - 1) * 100, 1)
                signal = "oversold" if current_rsi < 30 else "overbought" if current_rsi > 70 else "neutral"

                pe = info.get("forwardPE")
                gross_margin = info.get("grossMargins")
                rev_growth = info.get("revenueGrowth")
                fcf = info.get("freeCashflow")
                analyst_target = info.get("targetMeanPrice")
                upside = round((analyst_target - current_price) / current_price * 100, 1) if analyst_target else None

                rows.append({
                    "ticker": ticker,
                    "price": current_price,
                    "rsi_14": current_rsi,
                    "signal": signal,
                    "1w_chg%": float(change_1w),
                    "1m_chg%": float(change_1m),
                    "fwd_pe": round(pe, 1) if pe else None,
                    "gross_margin%": round(gross_margin * 100, 1) if gross_margin else None,
                    "rev_growth%": round(rev_growth * 100, 1) if rev_growth else None,
                    "fcf_B": round(fcf / 1e9, 1) if fcf else None,
                    "upside%": upside,
                })
            except Exception as e:
                st.warning(f"Could not fetch {ticker}: {e}")

        df = pd.DataFrame(rows).set_index("ticker")
        df_sorted = df.sort_values("rsi_14", ascending=True)

    st.subheader("Market Data")
    st.dataframe(df_sorted, use_container_width=True)

    with st.spinner("Asking Claude..."):
        data_summary = df_sorted.to_string()
        prompt = f"""
You are a concise market analyst. I will give you RSI, price change,
and fundamental data for a watchlist of stocks.

Analyze this data and return ONLY a JSON object with no markdown:

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
- one_line max 15 words — reference both momentum AND valuation where relevant
- key_observation max 25 words
- top_opportunity and top_risk max 20 words each
- rank watchlist by conviction, highest first — weight cheap+oversold highest
- flag any stock where upside% > 30 and rsi < 45 as high conviction
- return raw JSON only, no backticks, no markdown

Data:
{data_summary}
"""
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text
        clean = re.sub(r"```json|```", "", response).strip()
        parsed = json.loads(clean)

    st.subheader("Claude's Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Mood", parsed["market_mood"])
    with col2:
        st.metric("Top Opportunity", "")
        st.caption(parsed["top_opportunity"])
    with col3:
        st.metric("Top Risk", "")
        st.caption(parsed["top_risk"])

    st.info(parsed["key_observation"])

    st.subheader("Price Charts")

    chart_tabs = st.tabs(tickers)

    for i, ticker in enumerate(tickers):
        with chart_tabs[i]:
            try:
                hist = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
                close = hist["Close"].squeeze()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=close.index,
                    y=close.values.flatten(),
                    name=ticker,
                    line=dict(width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(99, 110, 250, 0.08)"
                ))

                ticker_row = df_sorted.loc[ticker] if ticker in df_sorted.index else None
                if ticker_row is not None:
                    rsi_val = ticker_row["rsi_14"]
                    chg_1m = ticker_row["1m_chg%"]
                    chg_1w = ticker_row["1w_chg%"]
                    fig.update_layout(
                        title=f"{ticker}   |   RSI {rsi_val}   |   1W {chg_1w:+.1f}%   |   1M {chg_1m:+.1f}%",
                    )

                fig.update_layout(
                    height=320,
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis_title=None,
                    yaxis_title="Price (USD)",
                    yaxis=dict(range=[close.min() * 0.95, close.max() * 1.05]),
                    hovermode="x unified",
                    showlegend=False,
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Chart unavailable for {ticker}: {e}")

    st.subheader("Watchlist Breakdown")
    for item in parsed["watchlist"]:
        stance = item["stance"]
        color = "🟢" if stance == "bullish" else "🔴" if stance == "bearish" else "🟡"
        with st.expander(f"{color} {item['ticker']} — RSI {item['rsi']} — {stance.upper()}"):
            st.write(item["one_line"])

else:
    st.info("Configure your watchlist in the sidebar and click Run Analysis.")