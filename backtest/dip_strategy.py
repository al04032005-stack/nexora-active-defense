#!/usr/bin/env python3
"""Systematic version of the short-term mean-reversion style found in the trade journal.

Long side ("抄底"):  high-volatility stock trading below its 50-day average after a sharp
                     multi-day drop -> buy a 2x long exposure at the next open.
Short side ("做空"): high-volatility stock below its 50-day average after a sharp bounce
                     -> buy a 2x inverse exposure at the next open.
Exits: first close back above the 5-day average (long) / below it (short), or a time limit.

2x exposure is simulated as a daily-reset leveraged ETF: each held day returns 2x the
underlying's daily return, compounded, minus a per-side cost.

Usage:
    python3 backtest/dip_strategy.py --prices prices.csv.gz
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from common import BENCHMARKS, load_prices, perf_stats, rsi
from trade_journal import LEVERAGED

NOT_STOCKS = set(LEVERAGED) | {"DRAM", "CBRX"}

LEVERAGE = 2.0
COST = 0.001  # per side: spread + slippage on single-stock leveraged ETFs


def indicators(bars):
    c = bars["close"]
    out = pd.DataFrame(index=bars.index)
    out["open"], out["close"] = bars["open"], c
    out["sma5"] = c.rolling(5).mean()
    out["sma50"] = c.rolling(50).mean()
    out["ret5"] = c / c.shift(5) - 1
    out["rsi2"] = rsi(c, 2)
    out["vol60"] = c.pct_change().rolling(60).std() * np.sqrt(252)
    return out


def signals(ind, side, trigger, min_vol):
    base = (ind["vol60"] >= min_vol) & (ind["close"] < ind["sma50"])
    kind, level = trigger
    if side == "long":
        trig = ind["ret5"] <= -level if kind == "ret5" else ind["rsi2"] < level
    else:
        trig = ind["ret5"] >= level if kind == "ret5" else ind["rsi2"] > level
    return (base & trig).fillna(False)


def run_symbol(sym, ind, side, trigger, min_vol, exit_rule, max_days):
    """Trades for one symbol: enter at the open after a signal, exit at a close.

    Each trade keeps its daily mark-to-market path (cumulative growth at each close)
    and the signal strength known at entry (the 5-day move), used to rank same-day signals.
    """
    sig = signals(ind, side, trigger, min_vol).values
    o, c, sma5, r5 = ind["open"].values, ind["close"].values, ind["sma5"].values, ind["ret5"].values
    dates = ind.index
    direction = 1.0 if side == "long" else -1.0
    trades, i, n = [], 0, len(ind)
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        j = i + 1
        growth = (1 + LEVERAGE * direction * (c[j] / o[j] - 1)) * (1 - COST)
        path = {dates[j]: max(growth, 0.0)}
        held = 1
        while True:
            bounce = (c[j] > sma5[j]) if side == "long" else (c[j] < sma5[j])
            if (exit_rule == "sma5" and bounce) or held >= max_days or j >= n - 1:
                break
            j += 1
            held += 1
            growth *= 1 + LEVERAGE * direction * (c[j] / c[j - 1] - 1)
            growth = max(growth, 0.0)
            path[dates[j]] = growth
        growth = growth * (1 - COST)
        path[dates[j]] = growth
        trades.append((sym, dates[i + 1], dates[j], held, growth - 1, abs(r5[i]), path))
        i = j + 1
    return trades


def all_trades(inds, side, trigger, min_vol, exit_rule, max_days):
    rows = []
    for sym, ind in inds.items():
        rows += run_symbol(sym, ind, side, trigger, min_vol, exit_rule, max_days)
    return pd.DataFrame(rows, columns=["symbol", "entry", "exit", "days", "ret", "strength", "path"])


def trade_stats(t):
    if t.empty:
        return {"n": 0, "avg": np.nan, "win": np.nan, "pf": np.nan, "worst": np.nan}
    wins, losses = t.ret[t.ret > 0], t.ret[t.ret < 0]
    return {"n": len(t), "avg": t.ret.mean(), "win": (t.ret > 0).mean(),
            "pf": wins.sum() / -losses.sum() if len(losses) else np.inf, "worst": t.ret.min()}


def portfolio(trades, slots, calendar):
    """Equal-slot portfolio marked to market daily.

    At each open, free slots take that day's new signals, strongest 5-day move first
    (known before entry); each new position gets 1/slots of the last marked equity.
    Positions are marked at every close and their proceeds return to cash at the exit close.
    """
    by_entry = {d: g.sort_values("strength", ascending=False) for d, g in trades.groupby("entry")}
    cash, last_equity, open_pos, curve = 1.0, 1.0, [], {}
    for d in calendar:
        todays = by_entry.get(d)
        if todays is not None:
            for row in todays.head(max(slots - len(open_pos), 0)).itertuples():
                alloc = min(last_equity / slots, cash)
                if alloc <= 0:
                    break
                cash -= alloc
                open_pos.append({"exit": row.exit, "alloc": alloc, "path": row.path, "mark": 1.0})
        for p in open_pos:
            p["mark"] = p["path"].get(d, p["mark"])
        last_equity = cash + sum(p["alloc"] * p["mark"] for p in open_pos)
        curve[d] = last_equity
        for p in [p for p in open_pos if p["exit"] == d]:
            cash += p["alloc"] * p["mark"]
            open_pos.remove(p)
    return pd.Series(curve)


def scan(prices, drop=0.15, watch_drop=0.10, min_vol=0.8):
    """Signals at the latest close under the selected rules."""
    rows = []
    for s, g in prices.groupby("symbol"):
        if s in BENCHMARKS or s in NOT_STOCKS or len(g) < 70:
            continue
        ind = indicators(g.set_index("date"))
        last = ind.iloc[-1]
        rows.append({"symbol": s, "date": ind.index[-1].date(), "close": last.close, "ret5": last.ret5,
                     "vs_sma50": last.close / last.sma50 - 1, "vol60": last.vol60, "rsi2": last.rsi2,
                     "exit_level_sma5": last.sma5})
    d = pd.DataFrame(rows)
    base = (d.vs_sma50 < 0) & (d.vol60 >= min_vol)
    fmt = {"ret5": "{:+.1%}".format, "vs_sma50": "{:+.1%}".format, "vol60": "{:.0%}".format,
           "rsi2": "{:.0f}".format, "close": "{:,.2f}".format, "exit_level_sma5": "{:,.2f}".format}
    for title, mask, asc in [
            (f"抄底信号 (5日跌≥{drop:.0%})", base & (d.ret5 <= -drop), True),
            (f"抄底观察 (5日跌{watch_drop:.0%}-{drop:.0%})", base & (d.ret5 <= -watch_drop) & (d.ret5 > -drop), True),
            (f"做空信号 (5日涨≥{drop:.0%})", base & (d.ret5 >= drop), False)]:
        x = d[mask].sort_values("ret5", ascending=asc)
        print(f"\n=== {title} ===")
        print(x.to_string(index=False, formatters=fmt) if len(x) else "  无")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True)
    p.add_argument("--insample-end", default="2022-12-31")
    p.add_argument("--oos-end", default="2026-06-30")
    p.add_argument("--scan", action="store_true", help="only list signals at the latest close")
    a = p.parse_args()

    prices = load_prices(a.prices)
    if a.scan:
        scan(prices)
        return
    stocks = sorted(set(prices.symbol) - BENCHMARKS - NOT_STOCKS)
    inds = {}
    for s, g in prices[prices.symbol.isin(stocks)].groupby("symbol"):
        if len(g) > 70:
            inds[s] = indicators(g.set_index("date"))
    print(f"股票池: {len(inds)} 只 | 数据 {prices.date.min().date()} → {prices.date.max().date()}")

    ins = lambda t: t[t.entry <= a.insample_end]
    oos = lambda t: t[(t.entry > a.insample_end) & (t.entry <= a.oos_end)]
    live = lambda t: t[t.entry > a.oos_end]

    grid = []
    for side in ["long", "short"]:
        triggers = [("ret5", 0.05), ("ret5", 0.10), ("ret5", 0.15),
                    ("rsi2", 10 if side == "long" else 90), ("rsi2", 25 if side == "long" else 75)]
        for trig, min_vol, (exit_rule, max_days) in itertools.product(
                triggers, [0.5, 0.8], [("sma5", 5), ("sma5", 10), ("time", 1), ("time", 3)]):
            t = all_trades(inds, side, trig, min_vol, exit_rule, max_days)
            label = (f"5日{'跌' if side == 'long' else '涨'}≥{trig[1]:.0%}" if trig[0] == "ret5"
                     else f"RSI2{'<' if side == 'long' else '>'}{trig[1]}")
            grid.append({"side": side, "trigger": label,
                         "min_vol": min_vol, "exit": f"{exit_rule}/{max_days}d", "trades": t,
                         **{f"is_{k}": v for k, v in trade_stats(ins(t)).items()},
                         **{f"oos_{k}": v for k, v in trade_stats(oos(t)).items()},
                         **{f"live_{k}": v for k, v in trade_stats(live(t)).items()}})
    g = pd.DataFrame(grid)
    show = ["side", "trigger", "min_vol", "exit", "is_n", "is_avg", "is_win", "is_pf",
            "oos_n", "oos_avg", "oos_win", "oos_pf", "live_n", "live_avg", "live_win"]
    fmt = {k: "{:+.2%}".format for k in ["is_avg", "oos_avg", "live_avg"]}
    fmt.update({k: "{:.0%}".format for k in ["is_win", "oos_win", "live_win"]})
    fmt.update({k: "{:.2f}".format for k in ["is_pf", "oos_pf"]})
    chosen = {}
    for side in ["long", "short"]:
        sub = g[(g.side == side) & (g.is_n >= 100)].sort_values("is_pf", ascending=False)
        print(f"\n=== {side} 参数网格 (按样本内利润因子排序; 样本内 ≤{a.insample_end}, "
              f"样本外 ≤{a.oos_end}, 之后为你的实盘期) ===")
        print(sub[show].head(12).to_string(index=False, formatters=fmt))
        print(f"  全部{len(g[g.side == side])}组参数中, 样本内利润因子>1: {(g[g.side == side].is_pf > 1).mean():.0%}, "
              f"样本外利润因子>1: {(g[g.side == side].oos_pf > 1).mean():.0%}")
        chosen[side] = sub.iloc[0]

    # year-by-year for the in-sample winners: is the edge stable or one regime?
    for side, row in chosen.items():
        t = row["trades"]
        by_year = t.groupby(t.entry.dt.year)["ret"].agg(["count", "mean", lambda x: (x > 0).mean()])
        by_year.columns = ["笔数", "平均每笔", "胜率"]
        print(f"\n{side} 最优参数 [{row['trigger']}, vol≥{row['min_vol']}, {row['exit']}] 逐年表现:")
        print(by_year.to_string(formatters={"平均每笔": "{:+.2%}".format, "胜率": "{:.0%}".format}))

    # portfolio view: 4 slots, each 25% of equity in a 2x product (full book = 2x exposure)
    spy = prices[prices.symbol == "SPY"].set_index("date")["close"]
    qqq = prices[prices.symbol == "QQQ"].set_index("date")["close"]
    qld = (1 + 2 * qqq.pct_change().fillna(0)).cumprod()
    periods = [("样本内", "2018-03-01", a.insample_end), ("样本外", a.insample_end, a.oos_end),
               ("你的实盘期", a.oos_end, "2026-12-31")]
    print("\n=== 组合回测: 最多4个仓位, 每仓25%资金买2倍产品 ===")
    print(f"{'区间':<8}{'策略':<16}{'区间收益':>10}{'年化':>9}{'最大回撤':>10}{'夏普':>7}")
    for label, s, e in periods:
        cal = spy.loc[s:e].index
        for name, row in [("抄底(long)", chosen["long"]), ("做空(short)", chosen["short"])]:
            t = row["trades"]
            curve = portfolio(t[(t.entry >= cal[0]) & (t.entry <= cal[-1])], 4, cal)
            st = perf_stats(curve, 252)
            print(f"{label:<8}{name:<16}{st['total']:>+10.1%}{st['cagr']:>+9.1%}{st['maxdd']:>+10.1%}{st['sharpe']:>7.2f}")
        for name, ser in [("SPY 持有", spy), ("QQQ 持有", qqq), ("QQQ 2倍持有", qld)]:
            st = perf_stats(ser.loc[s:e], 252)
            print(f"{label:<8}{name:<16}{st['total']:>+10.1%}{st['cagr']:>+9.1%}{st['maxdd']:>+10.1%}{st['sharpe']:>7.2f}")
    return g, inds


if __name__ == "__main__":
    main()
