import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import anthropic
import json
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from fredapi import Fred
import os
import json as jsonlib

load_dotenv()
client = anthropic.Anthropic()

def find_pivots(close, window=5):
    highs = []
    lows = []
    prices = close.values
    dates = close.index

    for i in range(window, len(prices) - window):
        if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
           all(prices[i] >= prices[i+j] for j in range(1, window+1)):
            highs.append({"date": str(dates[i].date()), "price": round(float(prices[i]), 2)})

        if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
           all(prices[i] <= prices[i+j] for j in range(1, window+1)):
            lows.append({"date": str(dates[i].date()), "price": round(float(prices[i]), 2)})

    return highs[-6:], lows[-6:]


def analyze_pattern(ticker, close, highs, lows):
    current_price = round(float(close.iloc[-1]), 2)

    high_prices = [h["price"] for h in highs]
    low_prices = [l["price"] for l in lows]

    structure = ""
    if len(high_prices) >= 2:
        if high_prices[-1] < high_prices[-2] * 0.995:
            structure += "lower highs forming. "
        elif high_prices[-1] > high_prices[-2] * 1.005:
            structure += "higher highs forming. "
        else:
            structure += "equal highs (within 0.5%). "

    if len(low_prices) >= 2:
        if low_prices[-1] > low_prices[-2] * 1.005:
            structure += "higher lows forming. "
        elif low_prices[-1] < low_prices[-2] * 0.995:
            structure += "lower lows forming. "
        else:
            structure += "equal lows (within 0.5%). "

    prompt = f"""
You are a technical analyst. Analyze this price structure data and identify the most likely chart pattern.

Ticker: {ticker}
Current price: {current_price}
3-month price range: {round(float(close.min()), 2)} - {round(float(close.max()), 2)}
Price structure: {structure}
Recent pivot highs (last 6): {highs}
Recent pivot lows (last 6): {lows}

Patterns to consider: double top, double bottom, head and shoulders, inverse head and shoulders,
ascending triangle, descending triangle, symmetrical triangle, rising wedge, falling wedge,
cup and handle, flag, pennant, or no clear pattern.

Return ONLY this JSON with no markdown:
{{
  "pattern": "",
  "confidence": "high / medium / low",
  "key_levels": {{
    "resistance": 0,
    "support": 0,
    "target": 0
  }},
  "implication": "bullish / bearish / neutral",
  "reasoning": "",
  "invalidation": ""
}}

Rules:
- reasoning max 20 words
- invalidation max 15 words
- target should be a realistic price projection based on the pattern
- if no clear pattern exists, set pattern to "no clear pattern" and confidence to "low"
- return raw JSON only
"""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    response = message.content[0].text
    clean = re.sub(r"```json|```", "", response).strip()
    return json.loads(clean)


WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f:
            return jsonlib.load(f).get("tickers", "GRAB, TSM, PFE, CLS, CDE, UNFI, LITE")
    return "GRAB, TSM, PFE, CLS, CDE, UNFI, LITE"

def save_watchlist(tickers_str):
    with open(WATCHLIST_FILE, "w") as f:
        jsonlib.dump({"tickers": tickers_str}, f)


st.set_page_config(page_title="Morning Brief", layout="wide")
st.title("Morning Brief")
st.caption("RSI signals + Claude analysis for your watchlist")

with st.sidebar:
    st.header("Watchlist")
    ticker_input = st.text_area("Tickers (comma separated)", value=load_watchlist(), height=150)
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)
    if run_button:
        save_watchlist(ticker_input)

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

        st.session_state["df_sorted"] = df_sorted
        st.session_state["tickers"] = tickers

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
            st.session_state["parsed"] = parsed

    if "df_sorted" in st.session_state and "parsed" in st.session_state:
        df_sorted = st.session_state["df_sorted"]
        parsed = st.session_state["parsed"]
        tickers = st.session_state.get("tickers", tickers)

        st.subheader("Market Data")
        st.dataframe(df_sorted, use_container_width=True)

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
                    period_options = {"1W": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}
                    selected_period = st.radio(
                        "Time range",
                        options=list(period_options.keys()),
                        index=2,
                        horizontal=True,
                        key=f"period_{ticker}"
                    )
                    hist = yf.download(ticker, period=period_options[selected_period], auto_adjust=True, progress=False)
                    close = hist["Close"].squeeze()

                    sma_col1, sma_col2, sma_col3 = st.columns(3)
                    with sma_col1:
                        show_sma_short = st.checkbox("SMA 20 (short)", value=True, key=f"sma_short_{ticker}")
                    with sma_col2:
                        show_sma_mid = st.checkbox("SMA 50 (mid)", value=True, key=f"sma_mid_{ticker}")
                    with sma_col3:
                        show_sma_long = st.checkbox("SMA 200 (long)", value=True, key=f"sma_long_{ticker}")

                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.72, 0.28],
                        vertical_spacing=0.03
                    )
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist["Open"].squeeze(),
                        high=hist["High"].squeeze(),
                        low=hist["Low"].squeeze(),
                        close=hist["Close"].squeeze(),
                        name=ticker,
                        increasing_line_color="rgba(0, 200, 100, 0.8)",
                        decreasing_line_color="rgba(220, 50, 50, 0.8)",
                    ), row=1, col=1)

                    if show_sma_short and len(close) >= 20:
                        sma20 = close.rolling(20).mean()
                        fig.add_trace(go.Scatter(
                            x=hist.index, y=sma20.values,
                            name="SMA 20",
                            line=dict(color="rgba(255, 200, 0, 0.8)", width=1.2),
                            hovertemplate="%{y:.2f}"
                        ), row=1, col=1)

                    if show_sma_mid and len(close) >= 50:
                        sma50 = close.rolling(50).mean()
                        fig.add_trace(go.Scatter(
                            x=hist.index, y=sma50.values,
                            name="SMA 50",
                            line=dict(color="rgba(0, 180, 255, 0.8)", width=1.2),
                            hovertemplate="%{y:.2f}"
                        ), row=1, col=1)

                    if show_sma_long and len(close) >= 200:
                        sma200 = close.rolling(200).mean()
                        fig.add_trace(go.Scatter(
                            x=hist.index, y=sma200.values,
                            name="SMA 200",
                            line=dict(color="rgba(255, 100, 200, 0.8)", width=1.2),
                            hovertemplate="%{y:.2f}"
                        ), row=1, col=1)

                    volume = hist["Volume"].squeeze()
                    vol_colors = [
                        "rgba(0, 200, 100, 0.5)" if c >= o else "rgba(220, 50, 50, 0.5)"
                        for c, o in zip(hist["Close"].squeeze(), hist["Open"].squeeze())
                    ]
                    fig.add_trace(go.Bar(
                        x=hist.index,
                        y=volume,
                        name="Volume",
                        marker_color=vol_colors,
                        showlegend=False,
                    ), row=2, col=1)

                    ticker_row = df_sorted.loc[ticker] if ticker in df_sorted.index else None
                    if ticker_row is not None:
                        rsi_val = ticker_row["rsi_14"]
                        chg_1m = ticker_row["1m_chg%"]
                        chg_1w = ticker_row["1w_chg%"]
                        fig.update_layout(
                            title=f"{ticker}   |   RSI {rsi_val}   |   1W {chg_1w:+.1f}%   |   1M {chg_1m:+.1f}%",
                        )

                    fig.update_layout(
                        height=480,
                        margin=dict(l=0, r=0, t=40, b=0),
                        yaxis_title="Price (USD)",
                        yaxis=dict(range=[close.min() * 0.95, close.max() * 1.05]),
                        yaxis2_title="Volume",
                        xaxis2_rangeslider_visible=False,
                        hovermode="x unified",
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0)
                    )
                    fig.update_xaxes(rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                    if st.button(f"Detect pattern — {ticker}", key=f"pattern_{ticker}"):
                        with st.spinner("Analyzing price structure..."):
                            try:
                                highs, lows = find_pivots(close)
                                result = analyze_pattern(ticker, close, highs, lows)
                                st.session_state[f"pattern_result_{ticker}"] = {
                                    "result": result,
                                    "highs": highs,
                                    "lows": lows,
                                    "close_min": float(close.min()),
                                    "close_max": float(close.max()),
                                    "close_index": [str(x) for x in hist.index],
                                    "close_values": close.values.flatten().tolist(),
                                    "open_values": hist["Open"].squeeze().values.flatten().tolist(),
                                    "high_values": hist["High"].squeeze().values.flatten().tolist(),
                                    "low_values": hist["Low"].squeeze().values.flatten().tolist(),
                                }
                            except Exception as e:
                                st.warning(f"Pattern detection failed: {e}")

                    if f"pattern_result_{ticker}" in st.session_state:
                        cached = st.session_state[f"pattern_result_{ticker}"]
                        result = cached["result"]
                        highs = cached["highs"]
                        lows = cached["lows"]

                        pattern = result["pattern"]
                        confidence = result["confidence"]
                        implication = result["implication"]

                        color = "🟢" if implication == "bullish" else "🔴" if implication == "bearish" else "🟡"
                        conf_color = {"high": "🔵", "medium": "🟠", "low": "⚪"}.get(confidence, "⚪")

                        st.markdown(f"### {color} {pattern.title()}")
                        st.caption(f"{conf_color} Confidence: {confidence}   |   Implication: {implication}")
                        st.write(result["reasoning"])

                        k1, k2, k3 = st.columns(3)
                        with k1:
                            st.metric("Resistance", f"${result['key_levels']['resistance']}")
                        with k2:
                            st.metric("Support", f"${result['key_levels']['support']}")
                        with k3:
                            st.metric("Target", f"${result['key_levels']['target']}")

                        st.caption(f"Invalidation: {result['invalidation']}")

                        fig_pattern = go.Figure()
                        fig_pattern.add_trace(go.Candlestick(
                            x=cached["close_index"],
                            open=cached["open_values"],
                            high=cached["high_values"],
                            low=cached["low_values"],
                            close=cached["close_values"],
                            name=ticker,
                            increasing_line_color="rgba(0, 200, 100, 0.8)",
                            decreasing_line_color="rgba(220, 50, 50, 0.8)",
                        ))
                        fig_pattern.add_trace(go.Scatter(
                            x=[h["date"] for h in highs],
                            y=[h["price"] for h in highs],
                            mode="markers",
                            marker=dict(color="red", size=8, symbol="triangle-down"),
                            name="Pivot highs"
                        ))
                        fig_pattern.add_trace(go.Scatter(
                            x=[l["date"] for l in lows],
                            y=[l["price"] for l in lows],
                            mode="markers",
                            marker=dict(color="green", size=8, symbol="triangle-up"),
                            name="Pivot lows"
                        ))
                        fig_pattern.update_layout(
                            height=320,
                            margin=dict(l=0, r=0, t=40, b=0),
                            yaxis_title="Price (USD)",
                            yaxis=dict(range=[cached["close_min"] * 0.95, cached["close_max"] * 1.05]),
                            xaxis_rangeslider_visible=False,
                            hovermode="x unified",
                            title=f"{ticker} — {pattern.title()}",
                        )
                        st.plotly_chart(fig_pattern, use_container_width=True)

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