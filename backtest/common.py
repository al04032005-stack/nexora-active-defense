"""Shared data loading, indicators and performance statistics for the backtests.

Price file format (CSV, optionally gzipped):
    symbol,date,open,high,low,close,volume
Daily bars, split-adjusted.
"""
import numpy as np
import pandas as pd

BENCHMARKS = {"SPY", "QQQ", "SMH", "SOXX", "IWM"}


def load_prices(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["low"] > 0)]
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def rma(x, n):
    """Wilder's moving average, as used by TradingView's ta.rsi."""
    return x.ewm(alpha=1.0 / n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn)


def perf_stats(equity, periods_per_year):
    """CAGR, annualized vol, Sharpe (rf=0), max drawdown for an equity curve."""
    equity = pd.Series(equity).dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan, "total": np.nan}
    rets = equity.pct_change().dropna()
    years = len(rets) / periods_per_year
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = rets.mean() / rets.std() * np.sqrt(periods_per_year) if rets.std() > 0 else np.nan
    maxdd = (equity / equity.cummax() - 1).min()
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": maxdd, "total": total}
