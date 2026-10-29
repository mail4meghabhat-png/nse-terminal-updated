"""
extra_indicators.py — second-round features for the NSE dashboard.
Pure pandas/numpy functions (no Streamlit, no network) so they are easy to test.
"""

import numpy as np
import pandas as pd


# ── volatility / momentum / volume ────────────────────────────────────────

def atr(high, low, close, period=14):
    """Average True Range (Wilder smoothing)."""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic oscillator: returns (%K, %D)."""
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    return k, k.rolling(d_period).mean()


def obv(close, volume):
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def annualised_volatility(close, window=60):
    """Annualised volatility (%) from the last `window` daily returns."""
    return close.pct_change().tail(window).std() * np.sqrt(252) * 100


def bollinger_bands(close, window=20, num_std=2):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid - num_std * std, mid, mid + num_std * std


# ── levels, crosses, candlestick patterns ─────────────────────────────────

def swing_levels(hist, lookback=120, window=5):
    """Nearest swing-low below price (support) and swing-high above price (resistance)."""
    d = hist.tail(lookback)
    price = d["Close"].iloc[-1]
    size = 2 * window + 1
    swing_lows = d["Low"][d["Low"] == d["Low"].rolling(size, center=True).min()]
    swing_highs = d["High"][d["High"] == d["High"].rolling(size, center=True).max()]
    below = swing_lows[swing_lows < price]
    above = swing_highs[swing_highs > price]
    support = float(below.max()) if len(below) else float(d["Low"].min())
    resistance = float(above.min()) if len(above) else float(d["High"].max())
    return {"support": round(support, 2), "resistance": round(resistance, 2)}


def last_cross(ma_short, ma_long):
    """Most recent golden/death cross between two moving averages."""
    diff = (ma_short - ma_long).dropna()
    if len(diff) < 2:
        return None
    sign = np.sign(diff)
    changed = sign.ne(sign.shift()) & sign.shift().notna() & sign.ne(0)
    idx = diff.index[changed]
    if len(idx) == 0:
        return None
    last = idx[-1]
    return {
        "type": "Golden cross" if diff.loc[last] > 0 else "Death cross",
        "date": str(last.date()),
        "days_ago": int((diff.index[-1] - last).days),
    }


def detect_candle_patterns(hist):
    """Simple rule-based patterns on the latest candle(s)."""
    if len(hist) < 2:
        return []
    c, p = hist.iloc[-1], hist.iloc[-2]
    body = abs(c["Close"] - c["Open"])
    rng = c["High"] - c["Low"]
    found = []
    if rng > 0:
        upper = c["High"] - max(c["Open"], c["Close"])
        lower = min(c["Open"], c["Close"]) - c["Low"]
        if body <= 0.1 * rng:
            found.append("Doji (indecision)")
        elif lower >= 2 * body and upper <= body:
            found.append("Hammer-type candle (long lower wick)")
        elif upper >= 2 * body and lower <= body:
            found.append("Shooting-star-type candle (long upper wick)")
    if (p["Close"] < p["Open"] and c["Close"] > c["Open"]
            and c["Open"] <= p["Close"] and c["Close"] >= p["Open"]):
        found.append("Bullish engulfing")
    if (p["Close"] > p["Open"] and c["Close"] < c["Open"]
            and c["Open"] >= p["Close"] and c["Close"] <= p["Open"]):
        found.append("Bearish engulfing")
    return found


# ── backtest: moving-average crossover vs buy & hold ──────────────────────

def max_drawdown_pct(curve):
    return float((curve / curve.cummax() - 1).min() * 100)


def backtest_ma_cross(close, short=50, long=200):
    """Long when MA(short) > MA(long) (signal acts next day), else cash.
    Ignores brokerage, taxes and slippage."""
    ma_s = close.rolling(short).mean()
    ma_l = close.rolling(long).mean()
    valid = ma_l.notna()
    ret = close.pct_change().fillna(0)
    sig = (ma_s > ma_l).astype(int).shift(1).fillna(0)

    r, s = ret[valid], sig[valid]
    if len(r) < 30:
        return None
    strat = (1 + r * s).cumprod()
    hold = (1 + r).cumprod()

    run_id = (s.diff().fillna(0) != 0).cumsum()
    trade_returns = [
        float((1 + r[g.index]).prod() - 1)
        for _, g in s[s == 1].groupby(run_id[s == 1])
    ]
    years = len(r) / 252

    def cagr(curve):
        return float((curve.iloc[-1] ** (1 / years) - 1) * 100) if years > 0 else None

    stats = {
        "Strategy total return %": round(float((strat.iloc[-1] - 1) * 100), 2),
        "Buy & hold total return %": round(float((hold.iloc[-1] - 1) * 100), 2),
        "Strategy CAGR %": round(cagr(strat), 2),
        "Buy & hold CAGR %": round(cagr(hold), 2),
        "Strategy max drawdown %": round(max_drawdown_pct(strat), 2),
        "Buy & hold max drawdown %": round(max_drawdown_pct(hold), 2),
        "Trades": len(trade_returns),
        "Win rate %": round(100 * sum(t > 0 for t in trade_returns) / len(trade_returns), 1)
        if trade_returns else None,
        "Time in market %": round(float(s.mean() * 100), 1),
    }
    curves = pd.DataFrame({"Strategy": strat, "Buy & hold": hold})
    return stats, curves


# ── portfolio ─────────────────────────────────────────────────────────────

def portfolio_stats(prices, weights, rf=0.065):
    """prices: DataFrame of closes (columns = tickers); weights: Series summing to 1."""
    rets = prices.pct_change().dropna()
    if len(rets) < 30:
        return None
    w = weights.reindex(rets.columns).fillna(0)
    port = (rets * w).sum(axis=1)
    curve = (1 + port).cumprod()
    ann_ret = float(curve.iloc[-1] ** (252 / len(port)) - 1)
    vol = float(port.std() * np.sqrt(252))
    return {
        "Annualised return %": round(ann_ret * 100, 2),
        "Annualised volatility %": round(vol * 100, 2),
        "Sharpe ratio": round((ann_ret - rf) / vol, 2) if vol else None,
        "Max drawdown %": round(max_drawdown_pct(curve), 2),
        "1-day 95% VaR %": round(float(-np.percentile(port, 5) * 100), 2),
    }, rets.corr()


# ── tiny keyword tagger for headlines (rough, not real sentiment analysis) ──

_POS = {"surge", "jumps", "rally", "gains", "profit", "beats", "record", "upgrade",
        "growth", "wins", "rises", "strong", "bullish", "buy", "outperform"}
_NEG = {"falls", "drops", "plunge", "loss", "misses", "downgrade", "probe", "fraud",
        "slump", "weak", "cuts", "bearish", "sell", "penalty", "decline", "crash"}


def headline_tag(title):
    words = {w.strip(".,:;!?'\"()").lower() for w in (title or "").split()}
    pos, neg = len(words & _POS), len(words & _NEG)
    if pos > neg:
        return "🟢 Positive"
    if neg > pos:
        return "🔴 Negative"
    return "⚪ Neutral"
