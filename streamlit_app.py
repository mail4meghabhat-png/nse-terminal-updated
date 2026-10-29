"""
NSE Stock Dashboard v2 — Streamlit (free hosting on Streamlit Community Cloud)

Repo files needed together:
    streamlit_app.py      (rename this v2 file to streamlit_app.py)
    nse_analysis_api.py
    extra_indicators.py
    requirements.txt

Run locally:   streamlit run streamlit_app.py
"""

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nse_analysis_api import (
    calculate_rsi,
    calculate_macd,
    calculate_moving_averages,
    get_technical_snapshot,
    get_fundamentals,
    safe_round,
)
from extra_indicators import (
    atr, stochastic, obv, annualised_volatility, bollinger_bands,
    swing_levels, last_cross, detect_candle_patterns,
    backtest_ma_cross, portfolio_stats, headline_tag,
)

st.set_page_config(page_title="NSE Stock Dashboard", page_icon="📈", layout="wide")

DEFAULT_UNIVERSE = (
    "RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY, BHARTIARTL, ITC, SBIN, LT, HINDUNILVR, "
    "KOTAKBANK, AXISBANK, MARUTI, SUNPHARMA, TITAN, ULTRACEMCO, ASIANPAINT, BAJFINANCE, "
    "NTPC, POWERGRID, ONGC, TATASTEEL, M&M, WIPRO, HCLTECH, ADANIPORTS, COALINDIA, "
    "NESTLEIND, TECHM, DRREDDY"
)


# ── helpers ────────────────────────────────────────────────────────────────

def to_ticker(symbol: str) -> str:
    symbol = symbol.upper().strip()
    return symbol if symbol.endswith(".NS") or symbol.startswith("^") else f"{symbol}.NS"


def parse_symbols(raw: str, limit: int):
    return [to_ticker(s) for s in raw.replace("\n", ",").split(",") if s.strip()][:limit]


def fmt(value, suffix="", prefix=""):
    return "N/A" if value is None else f"{prefix}{value:,}{suffix}"


@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period, auto_adjust=True)


@st.cache_data(ttl=900, show_spinner=False)
def load_batch(tickers: tuple, period: str = "2y") -> dict:
    """Download many tickers in one request; returns {ticker: DataFrame}."""
    if len(tickers) == 1:
        return {tickers[0]: load_history(tickers[0], period)}
    data = yf.download(list(tickers), period=period, group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    out = {}
    for t in tickers:
        try:
            df = data[t].dropna(how="all")
            if not df.empty:
                out[t] = df
        except KeyError:
            pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(ticker: str) -> dict:
    return get_fundamentals(yf.Ticker(ticker))


@st.cache_data(ttl=3600, show_spinner=False)
def load_valuation(ticker: str) -> dict:
    """P/E, P/B, dividend yield from statements (avoids stock.info)."""
    out = {"pe": None, "pb": None, "dividend_yield_pct": None}
    stock = yf.Ticker(ticker)
    mcap = None
    try:
        fi = stock.fast_info
        mcap = fi.get("marketCap") if fi else None
    except Exception:
        pass
    try:
        inc = stock.quarterly_income_stmt
        if inc is not None and "Net Income" in inc.index and mcap:
            ni = inc.loc["Net Income"].dropna()
            if len(ni) >= 4 and ni.iloc[:4].sum() > 0:
                out["pe"] = safe_round(mcap / ni.iloc[:4].sum())
    except Exception:
        pass
    try:
        bs = stock.quarterly_balance_sheet
        if bs is not None and mcap:
            for key in ("Stockholders Equity", "Common Stock Equity",
                        "Total Equity Gross Minority Interest"):
                if key in bs.index:
                    eq = bs.loc[key].dropna()
                    if len(eq) and eq.iloc[0] > 0:
                        out["pb"] = safe_round(mcap / eq.iloc[0])
                        break
    except Exception:
        pass
    try:
        div = stock.dividends
        price = stock.fast_info.get("lastPrice") if stock.fast_info else None
        if div is not None and len(div) and price:
            cutoff = pd.Timestamp.now(tz=div.index.tz) - pd.DateOffset(years=1)
            out["dividend_yield_pct"] = safe_round(div[div.index >= cutoff].sum() / price * 100)
    except Exception:
        pass
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def load_news(ticker: str) -> list:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for it in items[:12]:
        c = it.get("content", it)
        title = c.get("title")
        if not title:
            continue
        link = ((c.get("canonicalUrl") or {}).get("url")
                or (c.get("clickThroughUrl") or {}).get("url") or c.get("link"))
        source = (c.get("provider") or {}).get("displayName") or c.get("publisher") or ""
        date = c.get("pubDate") or c.get("providerPublishTime") or ""
        out.append({"title": title, "link": link, "source": source, "date": str(date)[:10]})
    return out


def relative_strength(hist, bench):
    out = {}
    for label, n in (("1M", 21), ("3M", 63), ("6M", 126)):
        if len(hist) > n and len(bench) > n:
            s = hist["Close"].iloc[-1] / hist["Close"].iloc[-n - 1] - 1
            b = bench["Close"].iloc[-1] / bench["Close"].iloc[-n - 1] - 1
            out[label] = {"stock": round(s * 100, 2), "nifty": round(b * 100, 2),
                          "excess": round((s - b) * 100, 2)}
    return out


def calc_beta(hist, bench):
    joined = pd.concat([hist["Close"].pct_change(), bench["Close"].pct_change()],
                       axis=1, join="inner").dropna().tail(252)
    if len(joined) < 60:
        return None
    var = joined.iloc[:, 1].var()
    return safe_round(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / var) if var else None


def composite_score(tech, fund, val, rel):
    parts = []

    def add(name, pts, mx):
        parts.append({"Factor": name, "Points": pts, "Max": mx})

    add("Trend (MA50 vs MA200)", 20 if tech.get("ma_trend") == "bullish" else 0, 20)
    mom = 0
    if (tech.get("macd_histogram") or 0) > 0:
        mom += 10
    rsi = tech.get("rsi_14")
    if rsi is not None:
        if 40 <= rsi <= 70:
            mom += 10
        elif rsi < 30:
            mom += 5
    add("Momentum (MACD + RSI)", mom, 20)
    adx = tech.get("adx_14") or 0
    add("Trend strength (ADX)", 10 if adx > 25 else 5 if adx > 20 else 0, 10)
    ex = rel.get("3M", {}).get("excess")
    add("Relative strength vs Nifty (3M)", 15 if (ex is not None and ex > 0) else 0, 15)
    growth = (8 if (fund.get("yoy_earnings_growth_pct") or 0) > 0 else 0) + \
             (7 if (fund.get("yoy_revenue_growth_pct") or 0) > 0 else 0)
    add("Growth (earnings + revenue YoY)", growth, 15)
    de = fund.get("debt_to_equity")
    quality = (5 if (fund.get("roe_pct") or 0) > 15 else 0) + \
              (5 if (de is not None and de < 1) else 0)
    add("Quality (ROE + debt)", quality, 10)
    pe = val.get("pe")
    v = 10 if (pe is not None and 0 < pe < 25) else 5 if (pe is not None and 25 <= pe < 40) else 0
    add("Valuation (P/E)", v, 10)
    score = sum(p["Points"] for p in parts)
    label = "Bullish" if score >= 70 else "Neutral" if score >= 45 else "Weak"
    return score, label, pd.DataFrame(parts)


@st.cache_data(ttl=900, show_spinner=False)
def build_report(ticker: str):
    hist = load_history(ticker)
    if hist.empty:
        return None
    bench = load_history("^NSEI")
    tech, flag = get_technical_snapshot(hist)
    fund = load_fundamentals(ticker)
    val = load_valuation(ticker)
    rel = relative_strength(hist, bench)
    beta = calc_beta(hist, bench)
    score, label, breakdown = composite_score(tech, fund, val, rel)
    return {"hist": hist, "tech": tech, "flag": flag, "fund": fund, "val": val,
            "rel": rel, "beta": beta, "score": score, "label": label, "breakdown": breakdown}


def extras(hist: pd.DataFrame) -> dict:
    """Volume, volatility, levels, cross and candle info for the Analysis tab."""
    close, high, low, vol = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    price = float(close.iloc[-1])
    atr_val = atr(high, low, close).iloc[-1]
    k, d = stochastic(high, low, close)
    ob = obv(close, vol)
    ma50, ma200 = calculate_moving_averages(close)
    avg_vol = vol.tail(21).iloc[:-1].mean()
    levels = swing_levels(hist)
    return {
        "price": price,
        "vol_ratio": safe_round(vol.iloc[-1] / avg_vol) if avg_vol else None,
        "atr": safe_round(atr_val),
        "atr_pct": safe_round(atr_val / price * 100),
        "volatility": safe_round(annualised_volatility(close)),
        "stoch_k": safe_round(k.iloc[-1]),
        "stoch_d": safe_round(d.iloc[-1]),
        "obv_trend": "Rising" if ob.iloc[-1] > ob.rolling(20).mean().iloc[-1] else "Falling",
        "support": levels["support"],
        "resistance": levels["resistance"],
        "atr_stop": safe_round(price - 2 * atr_val),
        "cross": last_cross(ma50, ma200),
        "patterns": detect_candle_patterns(hist),
    }


def make_chart(hist, months, show_bb, show_levels):
    close = hist["Close"]
    rsi = calculate_rsi(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    lo, mid, up = bollinger_bands(close)

    d = hist.tail(months * 21)
    idx = d.index
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.12, 0.16, 0.22], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=idx, open=d["Open"], high=d["High"], low=d["Low"],
                                 close=d["Close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ma50.loc[idx], name="MA50", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ma200.loc[idx], name="MA200", line=dict(width=1.5)), row=1, col=1)
    if show_bb:
        for series, name in ((up, "BB upper"), (lo, "BB lower")):
            fig.add_trace(go.Scatter(x=idx, y=series.loc[idx], name=name,
                                     line=dict(width=1, dash="dot")), row=1, col=1)
    if show_levels:
        lv = swing_levels(hist)
        fig.add_hline(y=lv["support"], line_dash="dash", annotation_text="Support", row=1, col=1)
        fig.add_hline(y=lv["resistance"], line_dash="dash", annotation_text="Resistance", row=1, col=1)

    vcolors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(d["Close"], d["Open"])]
    fig.add_trace(go.Bar(x=idx, y=d["Volume"], name="Volume", marker_color=vcolors), row=2, col=1)

    fig.add_trace(go.Scatter(x=idx, y=rsi.loc[idx], name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", row=3, col=1)

    hcolors = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_hist.loc[idx]]
    fig.add_trace(go.Bar(x=idx, y=macd_hist.loc[idx], name="MACD hist", marker_color=hcolors), row=4, col=1)
    fig.add_trace(go.Scatter(x=idx, y=macd_line.loc[idx], name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=idx, y=signal_line.loc[idx], name="Signal"), row=4, col=1)

    fig.update_layout(height=860, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.04), margin=dict(l=10, r=10, t=30, b=10))
    return fig


def csv_button(df: pd.DataFrame, label: str, filename: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")


# ── sidebar ────────────────────────────────────────────────────────────────

st.title("📈 NSE Stock Dashboard")
st.caption("Free, Python-only dashboard. Data: Yahoo Finance (may be delayed). "
           "For education only — not investment advice.")

with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("NSE symbol", value="RELIANCE", help="e.g. TCS, INFY, HDFCBANK")
    months = st.selectbox("Chart range (months)", [3, 6, 12, 24], index=1)
    if st.button("🔄 Refresh data (clear cache)"):
        st.cache_data.clear()
        st.rerun()

ticker = to_ticker(symbol)

tabs = st.tabs(["Analysis", "Chart", "Compare", "Screener", "Portfolio",
                "Backtest", "News", "Watchlist"])
(tab_analysis, tab_chart, tab_compare, tab_screen,
 tab_port, tab_back, tab_news, tab_watch) = tabs


# ── Analysis ───────────────────────────────────────────────────────────────

with tab_analysis:
    rep = build_report(ticker)
    if rep is None:
        st.error(f"No data found for {ticker}. Check the symbol.")
    else:
        tech, fund, val, rel = rep["tech"], rep["fund"], rep["val"], rep["rel"]
        ex = extras(rep["hist"])

        if rep["flag"]:
            st.warning("A >15% single-day price move was detected in the last 2 years — "
                       "moving averages and Bollinger figures may be distorted by a split "
                       "or demerger. Verify before trusting the technical signals.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Price (₹)", fmt(tech["current_price"]))
        c2.metric("Composite score", f"{rep['score']}/100", rep["label"])
        c3.metric("Beta vs Nifty", fmt(rep["beta"]))

        st.subheader("Technical")
        t = st.columns(5)
        t[0].metric("RSI (14)", fmt(tech["rsi_14"]))
        t[1].metric("MACD hist", fmt(tech["macd_histogram"]))
        t[2].metric("MA trend", (tech["ma_trend"] or "N/A").title())
        t[3].metric("ADX (14)", fmt(tech["adx_14"]))
        t[4].metric("% of 52w high", fmt(tech["pct_of_52w_high"], "%"))

        st.subheader("Volume & volatility")
        vv = st.columns(5)
        vv[0].metric("Volume vs 20d avg", fmt(ex["vol_ratio"], "x"))
        vv[1].metric("OBV trend", ex["obv_trend"])
        vv[2].metric("ATR (14)", fmt(ex["atr"], prefix="₹"), fmt(ex["atr_pct"], "% of price"), delta_color="off")
        vv[3].metric("Volatility (60d, ann.)", fmt(ex["volatility"], "%"))
        vv[4].metric("Stochastic %K / %D", f"{fmt(ex['stoch_k'])} / {fmt(ex['stoch_d'])}")

        st.subheader("Levels & signals")
        lv = st.columns(4)
        lv[0].metric("Support", fmt(ex["support"], prefix="₹"))
        lv[1].metric("Resistance", fmt(ex["resistance"], prefix="₹"))
        lv[2].metric("2×ATR stop level", fmt(ex["atr_stop"], prefix="₹"),
                     help="Price minus two ATRs — a common volatility-based stop reference.")
        cross = ex["cross"]
        lv[3].metric("Last MA50/200 cross",
                     cross["type"] if cross else "None in range",
                     f"{cross['date']} ({cross['days_ago']}d ago)" if cross else None,
                     delta_color="off")
        st.write("**Latest candle:** " + (", ".join(ex["patterns"]) if ex["patterns"] else "No notable pattern"))

        st.subheader("Valuation")
        v = st.columns(3)
        v[0].metric("P/E (TTM)", fmt(val["pe"]))
        v[1].metric("P/B", fmt(val["pb"]))
        v[2].metric("Dividend yield", fmt(val["dividend_yield_pct"], "%"))

        st.subheader("Fundamentals")
        f = st.columns(5)
        f[0].metric("Earnings YoY", fmt(fund["yoy_earnings_growth_pct"], "%"))
        f[1].metric("Revenue YoY", fmt(fund["yoy_revenue_growth_pct"], "%"))
        f[2].metric("Net margin", fmt(fund["net_margin_pct"], "%"))
        f[3].metric("ROE", fmt(fund["roe_pct"], "%"))
        f[4].metric("Debt/Equity", fmt(fund["debt_to_equity"]))

        st.subheader("Relative strength vs Nifty 50")
        if rel:
            st.dataframe(pd.DataFrame(rel).T.rename(
                columns={"stock": "Stock %", "nifty": "Nifty %", "excess": "Excess %"}),
                width="stretch")
        else:
            st.info("Not enough history to compare against Nifty.")

        with st.expander("How the composite score is calculated"):
            st.dataframe(rep["breakdown"], width="stretch", hide_index=True)
            st.caption("Score ≥70 Bullish, 45–69 Neutral, <45 Weak. A simple rules-based "
                       "screen, not a recommendation.")


# ── Chart ──────────────────────────────────────────────────────────────────

with tab_chart:
    hist = load_history(ticker)
    if hist.empty:
        st.error(f"No data found for {ticker}.")
    else:
        o1, o2 = st.columns(2)
        show_bb = o1.checkbox("Bollinger Bands", value=False)
        show_levels = o2.checkbox("Support / resistance lines", value=True)
        st.plotly_chart(make_chart(hist, months, show_bb, show_levels), width="stretch")


# ── Compare ────────────────────────────────────────────────────────────────

with tab_compare:
    cc1, cc2 = st.columns(2)
    sym_a = cc1.text_input("Stock A", value=symbol, key="cmp_a")
    sym_b = cc2.text_input("Stock B", value="TCS", key="cmp_b")
    ta, tb = to_ticker(sym_a), to_ticker(sym_b)
    ra, rb = build_report(ta), build_report(tb)
    if ra is None or rb is None:
        st.error("Could not load data for one of the symbols.")
    else:
        closes = pd.concat([ra["hist"]["Close"].rename(ta), rb["hist"]["Close"].rename(tb)],
                           axis=1, join="inner").dropna().tail(months * 21)
        rebased = closes / closes.iloc[0] * 100
        fig = go.Figure()
        for col in rebased.columns:
            fig.add_trace(go.Scatter(x=rebased.index, y=rebased[col], name=col))
        fig.update_layout(height=380, title="Price performance (rebased to 100)",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

        def row(label, fa, fb):
            return {"Metric": label, ta: fa, tb: fb}

        table = pd.DataFrame([
            row("Price (₹)", ra["tech"]["current_price"], rb["tech"]["current_price"]),
            row("Composite score", ra["score"], rb["score"]),
            row("RSI (14)", ra["tech"]["rsi_14"], rb["tech"]["rsi_14"]),
            row("MA trend", ra["tech"]["ma_trend"], rb["tech"]["ma_trend"]),
            row("3M return %", ra["rel"].get("3M", {}).get("stock"), rb["rel"].get("3M", {}).get("stock")),
            row("3M excess vs Nifty %", ra["rel"].get("3M", {}).get("excess"), rb["rel"].get("3M", {}).get("excess")),
            row("Beta", ra["beta"], rb["beta"]),
            row("P/E", ra["val"]["pe"], rb["val"]["pe"]),
            row("P/B", ra["val"]["pb"], rb["val"]["pb"]),
            row("ROE %", ra["fund"]["roe_pct"], rb["fund"]["roe_pct"]),
            row("Earnings YoY %", ra["fund"]["yoy_earnings_growth_pct"], rb["fund"]["yoy_earnings_growth_pct"]),
            row("Net margin %", ra["fund"]["net_margin_pct"], rb["fund"]["net_margin_pct"]),
            row("Debt/Equity", ra["fund"]["debt_to_equity"], rb["fund"]["debt_to_equity"]),
        ]).astype(str)
        st.dataframe(table, width="stretch", hide_index=True)


# ── Screener ───────────────────────────────────────────────────────────────

with tab_screen:
    st.write("Scan a list of stocks on technical filters (one batch download, up to 50 symbols).")
    raw_universe = st.text_area("Stock universe (editable)", value=DEFAULT_UNIVERSE, height=100)
    if st.button("Run screener", type="primary"):
        syms = tuple(parse_symbols(raw_universe, 50))
        rows = []
        with st.spinner(f"Scanning {len(syms)} stocks..."):
            data = load_batch(syms)
            for t, h in data.items():
                try:
                    if len(h) < 210:
                        continue
                    tc, _ = get_technical_snapshot(h)
                    n = 63
                    ret3m = (h["Close"].iloc[-1] / h["Close"].iloc[-n - 1] - 1) * 100
                    avg_v = h["Volume"].tail(21).iloc[:-1].mean()
                    rows.append({
                        "Symbol": t.replace(".NS", ""),
                        "Price": tc["current_price"],
                        "RSI": tc["rsi_14"],
                        "MACD hist": tc["macd_histogram"],
                        "Trend": tc["ma_trend"],
                        "ADX": tc["adx_14"],
                        "% of 52w high": tc["pct_of_52w_high"],
                        "3M return %": round(float(ret3m), 2),
                        "Volume x avg": round(float(h["Volume"].iloc[-1] / avg_v), 2) if avg_v else None,
                    })
                except Exception:
                    continue
        st.session_state["screen"] = pd.DataFrame(rows)

    scr = st.session_state.get("screen")
    if scr is not None and not scr.empty:
        st.subheader("Filters")
        f1, f2, f3, f4 = st.columns(4)
        rsi_lo, rsi_hi = f1.slider("RSI range", 0, 100, (0, 100))
        trend_f = f2.selectbox("MA trend", ["Any", "bullish", "bearish"])
        min_high = f3.slider("Min % of 52w high", 0, 100, 0)
        macd_pos = f4.checkbox("MACD histogram > 0")

        out = scr[(scr["RSI"].fillna(-1) >= rsi_lo) & (scr["RSI"].fillna(101) <= rsi_hi)]
        if trend_f != "Any":
            out = out[out["Trend"] == trend_f]
        out = out[out["% of 52w high"].fillna(0) >= min_high]
        if macd_pos:
            out = out[out["MACD hist"].fillna(-1) > 0]
        st.write(f"**{len(out)}** of {len(scr)} stocks match")
        st.dataframe(out.sort_values("RSI"), width="stretch", hide_index=True)
        csv_button(out, "⬇️ Download results (CSV)", "screener_results.csv")
    elif scr is not None:
        st.warning("No usable data returned. Yahoo may be rate-limiting — try again shortly.")


# ── Portfolio ──────────────────────────────────────────────────────────────

with tab_port:
    st.write("Edit your holdings below (add rows with the + at the bottom of the table).")
    default_hold = pd.DataFrame({
        "Symbol": ["RELIANCE", "TCS", "HDFCBANK"],
        "Qty": [10, 5, 15],
        "Buy price": [2500.0, 3500.0, 1600.0],
    })
    holdings = st.data_editor(default_hold, num_rows="dynamic", width="stretch", key="holdings")
    rf = st.number_input("Risk-free rate (% p.a.) for Sharpe", value=6.5, step=0.1) / 100

    h = holdings.dropna(subset=["Symbol", "Qty", "Buy price"]).copy()
    h = h[(h["Qty"] > 0) & (h["Symbol"].astype(str).str.strip() != "")]
    if h.empty:
        st.info("Add at least one holding.")
    else:
        h["Ticker"] = h["Symbol"].map(to_ticker)
        data = load_batch(tuple(h["Ticker"].unique()), "1y")
        h = h[h["Ticker"].isin(data.keys())].copy()
        if h.empty:
            st.error("Could not load prices for any holding.")
        else:
            h["Price"] = h["Ticker"].map(lambda t: float(data[t]["Close"].iloc[-1]))
            h["Invested"] = h["Qty"] * h["Buy price"]
            h["Value"] = h["Qty"] * h["Price"]
            h["P&L"] = h["Value"] - h["Invested"]
            h["P&L %"] = h["P&L"] / h["Invested"] * 100
            h["Weight %"] = h["Value"] / h["Value"].sum() * 100

            m = st.columns(4)
            inv, cur = h["Invested"].sum(), h["Value"].sum()
            m[0].metric("Invested", f"₹{inv:,.0f}")
            m[1].metric("Current value", f"₹{cur:,.0f}")
            m[2].metric("Total P&L", f"₹{cur - inv:,.0f}", f"{(cur / inv - 1) * 100:.2f}%")
            m[3].metric("Holdings", len(h))

            show = h[["Symbol", "Qty", "Buy price", "Price", "Invested", "Value", "P&L", "P&L %", "Weight %"]].round(2)
            st.dataframe(show, width="stretch", hide_index=True)
            csv_button(show, "⬇️ Download portfolio (CSV)", "portfolio.csv")

            pie = go.Figure(go.Pie(labels=h["Symbol"], values=h["Value"], hole=0.45))
            pie.update_layout(height=340, title="Allocation by current value",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(pie, width="stretch")

            closes = pd.concat({t: data[t]["Close"] for t in h["Ticker"]}, axis=1, join="inner").dropna()
            weights = (h.set_index("Ticker")["Value"] / h["Value"].sum())
            result = portfolio_stats(closes, weights, rf) if len(h) else None
            if result:
                stats, corr = result
                st.subheader("Risk (last 1 year)")
                sc = st.columns(len(stats))
                for col, (k, v_) in zip(sc, stats.items()):
                    col.metric(k, fmt(v_))
                if len(h) > 1:
                    cfig = go.Figure(go.Heatmap(z=corr.values, x=[c.replace(".NS", "") for c in corr.columns],
                                                y=[c.replace(".NS", "") for c in corr.index],
                                                zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                                                text=corr.round(2).values, texttemplate="%{text}"))
                    cfig.update_layout(height=380, title="Correlation of daily returns",
                                       margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(cfig, width="stretch")
            else:
                st.info("Not enough overlapping price history to compute risk metrics.")


# ── Backtest ───────────────────────────────────────────────────────────────

with tab_back:
    st.write(f"Backtest a moving-average crossover on **{ticker}** vs simply buying and holding.")
    b1, b2, b3 = st.columns(3)
    short_w = b1.number_input("Short MA", 5, 100, 50)
    long_w = b2.number_input("Long MA", 20, 400, 200)
    yrs = b3.selectbox("History", ["5y", "10y"], index=0)
    hb = load_history(ticker, yrs)
    if hb.empty:
        st.error(f"No data found for {ticker}.")
    elif short_w >= long_w:
        st.warning("Short MA must be smaller than long MA.")
    else:
        res = backtest_ma_cross(hb["Close"], int(short_w), int(long_w))
        if res is None:
            st.info("Not enough history for these settings.")
        else:
            stats, curves = res
            fig = go.Figure()
            for col in curves.columns:
                fig.add_trace(go.Scatter(x=curves.index, y=curves[col], name=col))
            fig.update_layout(height=380, title="Growth of ₹1", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pd.DataFrame({"Metric": list(stats.keys()),
                                       "Value": [str(v) for v in stats.values()]}),
                         width="stretch", hide_index=True)
            st.caption("Trades happen at the next day's price after the signal. Ignores brokerage, "
                       "taxes and slippage. Past results do not predict future returns.")


# ── News ───────────────────────────────────────────────────────────────────

with tab_news:
    news = load_news(ticker)
    if not news:
        st.info("No news returned for this symbol right now.")
    else:
        for n in news:
            tag = headline_tag(n["title"])
            link = f"[{n['title']}]({n['link']})" if n["link"] else n["title"]
            st.markdown(f"{tag} &nbsp; {link}  \n<small>{n['source']} · {n['date']}</small>",
                        unsafe_allow_html=True)
        st.caption("Tags come from a simple keyword check on headlines — a rough guide, not real sentiment analysis.")


# ── Watchlist ──────────────────────────────────────────────────────────────

with tab_watch:
    st.write("Enter up to 10 symbols, comma-separated.")
    raw = st.text_area("Watchlist", value="RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK")
    syms = tuple(parse_symbols(raw, 10))
    rows = []
    with st.spinner("Loading watchlist..."):
        data = load_batch(syms) if syms else {}
        for t in syms:
            h = data.get(t)
            if h is None or h.empty:
                rows.append({"Symbol": t, "Note": "No data"})
                continue
            try:
                tc, flag = get_technical_snapshot(h)
                rows.append({
                    "Symbol": t, "Price (₹)": tc["current_price"], "RSI": tc["rsi_14"],
                    "MACD hist": tc["macd_histogram"], "Trend": tc["ma_trend"],
                    "% of 52w high": tc["pct_of_52w_high"], "Corp. action?": "⚠️" if flag else "",
                })
            except Exception as e:
                rows.append({"Symbol": t, "Note": str(e)[:60]})
    if rows:
        wl = pd.DataFrame(rows)
        st.dataframe(wl, width="stretch", hide_index=True)
        csv_button(wl, "⬇️ Download watchlist (CSV)", "watchlist.csv")
