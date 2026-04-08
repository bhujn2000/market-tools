import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import anthropic
import json
import re
import plotly.graph_objects as go
from dotenv import load_dotenv
from fredapi import Fred
import os

load_dotenv()
client = anthropic.Anthropic()

st.set_page_config(page_title="Morning Brief", layout="wide")
st.title("Morning Brief")
st.caption("RSI signals + Claude analysis for your watchlist")

with st.sidebar:
    st.header("Watchlist")
    default_tickers = default_tickers = "GRAB, TSM, PFE, CLS, CDE, UNFI, LITE"
    ticker_input = st.text_area("Tickers (comma separated)", value=default_tickers, height=150)
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)

tab1, tab2 = st.tabs(["Watchlist", "Macro"])

with tab2:
    st.subheader("Macro Dashboard")
    macro_button = st.button("Load Macro Data", type="primary")

    if macro_button:
        fred = Fred(api_key=os.environ.get("FRED_API_KEY"))

        def get_latest(series_id):
            try:
                s = fred.get_series(series_id, observation_start="2020-01-01")
                s = s.dropna()
                return s
            except:
                return pd.Series(dtype=float)

        with st.spinner("Fetching macro data from FRED..."):
            series = {
                "Fed Funds Rate": "FEDFUNDS",
                "CPI YoY": "CPIAUCSL",
                "Core PCE YoY": "PCEPILFE",
                "Unemployment Rate": "UNRATE",
                "10Y Treasury": "DGS10",
                "2Y Treasury": "DGS2",
                "10Y Breakeven Inflation": "T10YIE",
            }

            macro_rows = {}
        for label, code in series.items():
            s = get_latest(code)
            if not s.empty:
                if label in ["CPI YoY", "Core PCE YoY"]:
                    s = s.pct_change(12) * 100
                    s = s.dropna()
                macro_rows[label] = s

        macro_df = pd.DataFrame(macro_rows)
        macro_df.index = pd.to_datetime(macro_df.index)
        latest = macro_df.ffill().iloc[-1]
        prev = macro_df.ffill().iloc[-2]

        st.subheader("Current readings")
        cols = st.columns(len(latest))
        for i, (label, val) in enumerate(latest.items()):
            delta = round(val - prev[label], 2)
            with cols[i]:
                st.metric(label, f"{val:.2f}", delta=f"{delta:+.2f}")

        st.subheader("Yield curve")
        if "10Y Treasury" in macro_df.columns and "2Y Treasury" in macro_df.columns:
            spread = macro_df["10Y Treasury"].dropna() - macro_df["2Y Treasury"].dropna()
            spread = spread.dropna()
            spread.index = pd.to_datetime(spread.index)
            fig_yc = go.Figure()
            fig_yc.add_trace(go.Scatter(
                x=spread.index,
                y=spread.values,
                fill="tozeroy",
                line=dict(width=1.5),
                fillcolor="rgba(99,110,250,0.08)",
                name="10Y-2Y spread"
            ))
            fig_yc.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
            fig_yc.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_title="Spread (%)",
                hovermode="x unified",
                title="10Y - 2Y yield spread (inversion = recession signal)"
            )
            st.plotly_chart(fig_yc, use_container_width=True)

        st.subheader("Inflation vs Fed funds rate")
        fig_inf = go.Figure()
        if "CPI YoY" in macro_df.columns:
            cpi_yoy = macro_df["CPI YoY"].dropna()
            cpi_yoy.index = pd.to_datetime(cpi_yoy.index)
            fig_inf.add_trace(go.Scatter(
                x=cpi_yoy.index,
                y=cpi_yoy.values,
                name="CPI YoY %",
                line=dict(width=1.5)
            ))
        if "Fed Funds Rate" in macro_df.columns:
            ffr = macro_df["Fed Funds Rate"].dropna()
            ffr.index = pd.to_datetime(ffr.index)
            fig_inf.add_trace(go.Scatter(
                x=ffr.index,
                y=ffr.values,
                name="Fed Funds Rate",
                line=dict(width=1.5, dash="dot")
            ))
        fig_inf.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="%",
            hovermode="x unified",
            title="CPI vs Fed funds rate"
        )
        st.plotly_chart(fig_inf, use_container_width=True)

        with st.spinner("Claude's macro read..."):
            latest_str = "\n".join([f"{k}: {v:.2f}" for k, v in latest.items()])
            spread_now = round(
                latest["10Y Treasury"] - latest["2Y Treasury"], 2
            ) if "10Y Treasury" in latest and "2Y Treasury" in latest else "N/A"

            macro_prompt = f"""
You are a macro strategist. Here are the latest US macro readings:

{latest_str}
10Y-2Y yield spread: {spread_now}%

Return ONLY a JSON object with no markdown:
{{
  "macro_regime": "one of: expansion / slowdown / stagflation / recession / recovery",
  "fed_stance": "one of: dovish / neutral / hawkish",
  "yield_curve_signal": "",
  "inflation_read": "",
  "key_macro_risk": "",
  "asset_class_implications": {{
    "equities": "",
    "bonds": "",
    "crypto": "",
    "commodities": ""
  }},
  "one_line_summary": ""
}}

Rules:
- all fields max 20 words
- be direct, no hedging
- return raw JSON only
"""
            macro_message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=800,
                messages=[{"role": "user", "content": macro_prompt}]
            )
            macro_response = macro_message.content[0].text
            macro_clean = re.sub(r"```json|```", "", macro_response).strip()
            macro_parsed = json.loads(macro_clean)

        st.subheader("Claude's macro read")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Macro regime", macro_parsed["macro_regime"])
        with m2:
            st.metric("Fed stance", macro_parsed["fed_stance"])
        with m3:
            st.metric("Yield curve", macro_parsed["yield_curve_signal"])

        st.info(macro_parsed["one_line_summary"])

        st.subheader("Asset class implications")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.caption("Equities")
            st.write(macro_parsed["asset_class_implications"]["equities"])
        with a2:
            st.caption("Bonds")
            st.write(macro_parsed["asset_class_implications"]["bonds"])
        with a3:
            st.caption("Crypto")
            st.write(macro_parsed["asset_class_implications"]["crypto"])
        with a4:
            st.caption("Commodities")
            st.write(macro_parsed["asset_class_implications"]["commodities"])

        st.subheader("Key macro risk")
        st.warning(macro_parsed["key_macro_risk"])

with tab1:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if run_button:
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