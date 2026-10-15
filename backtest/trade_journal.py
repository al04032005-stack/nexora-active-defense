#!/usr/bin/env python3
"""Trade-journal backtest.

Rebuilds round trips from broker fills (FIFO, longs and shorts), attributes P&L,
marks open positions to market every day, and replays simple rule changes
(hard stops, time stops, size caps) against the exact same trades.

Usage:
    python3 backtest/trade_journal.py --trades fills.csv --prices prices.csv.gz \
        --opening XYZ:100:50.00 --start 2026-07-01

fills.csv columns: time (UTC ISO), symbol, side (buy|sell|sell_short), qty, avg_price, hours, type, fees
prices.csv.gz:     symbol, date, open, high, low, close, volume (daily, split-adjusted)

Limitations: the replays use daily bars, so a stop is only evaluated on days strictly
after the entry day, and on the exit day only when the actual exit was already worse
than the stop. Same-day round trips are left unchanged.
"""
import argparse
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from common import load_prices

ET = "America/New_York"

# Leveraged/inverse single-stock ETFs and the underlying each one tracks.
# Verified by regressing daily returns on the underlying (see leverage_check()).
LEVERAGED = {
    "SNDQ": "SNDK", "SNXX": "SNDK", "SNDU": "SNDK",
    "NBIL": "NBIS", "NBIZ": "NBIS", "NEBX": "NBIS",
    "CONI": "COIN", "GGLL": "GOOGL", "SKUU": "SKHY",
    "TSLL": "TSLA", "TSLQ": "TSLA", "IRE": "IREN",
    "AVL": "AVGO", "INTW": "INTC", "SPCQ": "SPCX",
}

INVERSE = {"SNDQ", "NBIZ", "CONI", "TSLQ", "SPCQ"}

THEMES = {
    "SanDisk/存储": {"SNDK", "SNDQ", "SNXX", "SNDU", "MU", "DRAM", "SKHY", "SKUU"},
    "Nebius": {"NBIS", "NBIL", "NBIZ", "NEBX"},
    "SpaceX/航天": {"SPCX", "SPCQ", "RKLB"},
    "Intel": {"INTC", "INTW"},
}


def theme_of(sym):
    for name, members in THEMES.items():
        if sym in members:
            return name
    return "其他"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", required=True)
    p.add_argument("--prices", required=True)
    p.add_argument("--raw-closes", default="",
                   help="CSV symbol,date,raw_close of unadjusted closes; used to put fills on the split-adjusted basis")
    p.add_argument("--opening", default="", help="opening positions SYMBOL:QTY:COST,... held before --start")
    p.add_argument("--start", default="2026-07-01")
    p.add_argument("--out", default="", help="optional directory for CSV outputs")
    return p.parse_args()


def trading_calendar(prices):
    return pd.DatetimeIndex(sorted(prices.loc[prices.symbol == "SPY", "date"].unique()))


def assign_session_day(times_et, calendar):
    """Map each fill to the trading day whose 16:00 ET close it precedes."""
    closes = pd.DatetimeIndex([d.tz_localize(ET) + pd.Timedelta(hours=16) for d in calendar])
    idx = closes.searchsorted(times_et, side="left")
    idx = np.minimum(idx, len(calendar) - 1)
    return calendar[idx]


def adjust_fills_for_splits(fills, prices, raw_path, close_px):
    """Restate fill price/qty on the split-adjusted basis of the daily bars.

    Fills carry the share basis in force when they traded; the bars are split-adjusted.
    For each fill the adjustment factor (adjusted close / raw close) is taken from the
    trading days around it, choosing the factor that puts the fill nearest that day's bar.
    Price x quantity, and therefore P&L, is unchanged.
    """
    raw = pd.read_csv(raw_path, parse_dates=["date"])
    merged = raw.merge(prices[["symbol", "date", "close"]], on=["symbol", "date"])
    merged["factor"] = merged["close"] / merged["raw_close"]
    factors = {s: g.set_index("date")["factor"] for s, g in merged.groupby("symbol")}
    fills = fills.copy()
    changed = 0
    for i, f in fills.iterrows():
        series = factors.get(f["symbol"])
        if series is None or (series.round(2) == 1.0).all():
            continue
        near = series[(series.index >= f["day"] - pd.Timedelta(days=7)) & (series.index <= f["day"] + pd.Timedelta(days=7))]
        ref = close_px[f["symbol"]].get(f["day"])
        if near.empty or ref is None:
            continue
        best = min(set(near.round(4)), key=lambda c: abs(np.log(f["avg_price"] * c / ref)))
        if abs(best - 1.0) > 1e-3:
            fills.at[i, "avg_price"] = f["avg_price"] * best
            fills.at[i, "qty"] = f["qty"] / best
            changed += 1
    return fills, changed


def build_round_trips(fills, opening, start_time):
    """FIFO matching. Returns (closed lot pieces, episodes)."""
    lots = defaultdict(deque)  # symbol -> deque of lot dicts
    pos = defaultdict(float)
    episodes, current = [], {}
    pieces = []

    def ep_start(sym, t, direction):
        current[sym] = {"symbol": sym, "dir": direction, "start": t, "end": None, "pnl": 0.0, "fees": 0.0,
                        "n_fills": 0, "max_qty": 0.0, "max_cost": 0.0, "entry_hours": None}

    for sym, qty, cost in opening:
        lots[sym].append({"qty": qty, "price": cost, "time": start_time, "dir": "long", "hours": "opening",
                          "type": "opening", "day": None})
        pos[sym] = qty
        ep_start(sym, start_time, "long")
        current[sym]["entry_hours"] = "opening"
        current[sym]["max_qty"] = qty
        current[sym]["max_cost"] = qty * cost

    for f in fills.to_dict("records"):
        sym, q, px = f["symbol"], f["qty"], f["avg_price"]
        opening_dir = "long" if f["side"] == "buy" else "short"
        closing_dir = "short" if f["side"] == "buy" else "long"
        if sym not in current:
            ep_start(sym, f["time"], opening_dir)
        remaining = q
        while remaining > 1e-9 and lots[sym] and lots[sym][0]["dir"] == closing_dir:
            lot = lots[sym][0]
            take = min(remaining, lot["qty"])
            sign = 1 if closing_dir == "long" else -1
            pnl = sign * (px - lot["price"]) * take
            pieces.append({"symbol": sym, "dir": closing_dir, "qty": take, "entry_time": lot["time"],
                           "entry_price": lot["price"], "exit_time": f["time"], "exit_price": px, "pnl": pnl,
                           "entry_hours": lot["hours"], "exit_hours": f["hours"], "entry_type": lot["type"],
                           "exit_type": f["type"], "entry_day": lot["day"], "exit_day": f["day"]})
            current[sym]["pnl"] += pnl
            lot["qty"] -= take
            remaining -= take
            if lot["qty"] <= 1e-9:
                lots[sym].popleft()
        closed = q - remaining
        pos[sym] += closed if f["side"] == "buy" else -closed
        if closed > 1e-9:
            ep = current[sym]
            ep["n_fills"] += 1
            ep["fees"] += f["fees"] * closed / q
            if abs(pos[sym]) <= 1e-9:
                pos[sym] = 0.0
                ep["end"] = f["time"]
                episodes.append(current.pop(sym))
        if remaining > 1e-9:
            if sym not in current:  # the fill flattened the old position and opened the other side
                ep_start(sym, f["time"], opening_dir)
            lots[sym].append({"qty": remaining, "price": px, "time": f["time"], "dir": opening_dir,
                              "hours": f["hours"], "type": f["type"], "day": f["day"]})
            pos[sym] += remaining if f["side"] == "buy" else -remaining
            ep = current[sym]
            ep["n_fills"] += 1
            ep["fees"] += f["fees"] * remaining / q
            if ep["entry_hours"] is None:
                ep["entry_hours"] = f["hours"]
            ep["max_qty"] = max(ep["max_qty"], abs(pos[sym]))
            ep["max_cost"] = max(ep["max_cost"], sum(l["qty"] * l["price"] for l in lots[sym]))

    open_left = {s: sum(l["qty"] * (1 if l["dir"] == "long" else -1) for l in d) for s, d in lots.items() if d}
    return pd.DataFrame(pieces), pd.DataFrame(episodes), open_left


def daily_mark_to_market(fills, opening, start_day, calendar, close_px):
    """Cumulative P&L (realized + unrealized) at every daily close."""
    days = calendar[(calendar >= start_day) & (calendar <= fills["day"].max())]
    lots = defaultdict(deque)
    for sym, qty, cost in opening:
        lots[sym].append([qty, cost, 1])
    realized = 0.0
    out = []
    by_day = {d: g for d, g in fills.groupby("day")}
    for d in days:
        for f in by_day.get(d, pd.DataFrame()).to_dict("records"):
            sym, q, px = f["symbol"], f["qty"], f["avg_price"]
            closing_sign = -1 if f["side"] == "buy" else 1
            while q > 1e-9 and lots[sym] and lots[sym][0][2] == closing_sign:
                lot = lots[sym][0]
                take = min(q, lot[0])
                realized += closing_sign * (px - lot[1]) * take
                lot[0] -= take
                q -= take
                if lot[0] <= 1e-9:
                    lots[sym].popleft()
            if q > 1e-9:
                lots[sym].append([q, px, 1 if f["side"] == "buy" else -1])
            realized -= f["fees"]
        unreal, gross = 0.0, 0.0
        for sym, d_lots in lots.items():
            if not d_lots:
                continue
            c = close_px.get(sym, {}).get(d)
            if c is None:
                continue
            for qty, cost, sign in d_lots:
                unreal += sign * (c - cost) * qty
                gross += qty * c
        out.append({"day": d, "realized": realized, "unrealized": unreal, "total": realized + unreal,
                    "gross_exposure": gross})
    return pd.DataFrame(out).set_index("day")


def replay_stop(pieces, bars, stop_pct):
    """P&L of every lot if a hard stop at stop_pct from its entry had been in place."""
    new_pnl = []
    for p in pieces.to_dict("records"):
        pnl = p["pnl"]
        b = bars.get(p["symbol"])
        if b is None or p["entry_day"] is None or pd.isna(p["entry_day"]) or p["entry_day"] == p["exit_day"]:
            new_pnl.append(pnl)
            continue
        long_ = p["dir"] == "long"
        level = p["entry_price"] * (1 - stop_pct) if long_ else p["entry_price"] * (1 + stop_pct)
        window = b.loc[(b.index > p["entry_day"]) & (b.index <= p["exit_day"])]
        for day, row in window.iterrows():
            is_exit_day = day == p["exit_day"]
            breached = row["low"] <= level if long_ else row["high"] >= level
            if not breached:
                continue
            if is_exit_day:
                worse = p["exit_price"] < level if long_ else p["exit_price"] > level
                if not worse:
                    break
            fill = min(row["open"], level) if long_ else max(row["open"], level)
            pnl = ((fill - p["entry_price"]) if long_ else (p["entry_price"] - fill)) * p["qty"]
            break
        new_pnl.append(pnl)
    return np.array(new_pnl)


def replay_time_stop(pieces, bars, calendar, max_days):
    """P&L if every lot were closed at the close of the max_days-th trading day after its entry day."""
    new_pnl = []
    cal = pd.DatetimeIndex(calendar)
    for p in pieces.to_dict("records"):
        pnl = p["pnl"]
        b = bars.get(p["symbol"])
        if b is None or p["entry_day"] is None or pd.isna(p["entry_day"]):
            new_pnl.append(pnl)
            continue
        i = cal.searchsorted(p["entry_day"])
        if i + max_days < len(cal):
            cutoff = cal[i + max_days]
            if p["exit_day"] > cutoff and cutoff in b.index:
                c = b.loc[cutoff, "close"]
                pnl = ((c - p["entry_price"]) if p["dir"] == "long" else (p["entry_price"] - c)) * p["qty"]
        new_pnl.append(pnl)
    return np.array(new_pnl)


def leverage_check(prices):
    """Regress each leveraged ETF's daily return on its underlying; slope ≈ leverage."""
    rets = prices.pivot(index="date", columns="symbol", values="close").pct_change()
    out = {}
    for etf, und in LEVERAGED.items():
        if etf in rets and und in rets:
            pair = rets[[etf, und]].dropna()
            pair = pair[pair.index >= "2026-06-01"]
            if len(pair) > 15 and pair[und].var() > 0:
                beta = np.cov(pair[etf], pair[und])[0, 1] / pair[und].var()
                out[etf] = (und, beta, pair[etf].corr(pair[und]), len(pair))
    return out


def entry_setup(episodes, calendar, close):
    """Classify each episode by the underlying's position vs its 50-day average at the prior close."""
    sma50 = close.rolling(50).mean()
    underlying = episodes["symbol"].map(lambda s: LEVERAGED.get(s, s))
    bullish = ~(episodes["symbol"].isin(INVERSE) ^ (episodes["dir"] == "short"))
    days = assign_session_day(episodes["start"], calendar)
    labels = []
    for u, bull, d in zip(underlying, bullish, days):
        prev = calendar[max(calendar.searchsorted(d) - 1, 0)]
        c = close.at[prev, u] if u in close else np.nan
        m = sma50.at[prev, u] if u in sma50 else np.nan
        if not bull:
            labels.append("看空(反向ETF/做空)")
        elif pd.isna(c) or pd.isna(m):
            labels.append("做多(标的上市不足50天)")
        else:
            labels.append("抄底(标的在50日线下方做多)" if c < m else "追多(标的在50日线上方做多)")
    return labels


def stress_test(fills, opening, calendar, close, lookback_start):
    """Each day's overnight book hit by every underlying's worst (or best, if short) day of the prior year."""
    rets = close.pct_change().loc[lookback_start:]
    worst_dn, worst_up = rets.min(), rets.max()
    lev = {s: (-2.0 if s in INVERSE else 2.0) for s in LEVERAGED}
    pos = defaultdict(float)
    for s, q, _ in opening:
        pos[s] += q
    by_day = {d: g for d, g in fills.groupby("day")}
    rows = []
    for d in calendar[(calendar >= fills["day"].min()) & (calendar <= fills["day"].max())]:
        for f in by_day.get(d, pd.DataFrame()).to_dict("records"):
            pos[f["symbol"]] += f["qty"] if f["side"] == "buy" else -f["qty"]
        exposure, loss = 0.0, 0.0
        for s, q in pos.items():
            if abs(q) < 1e-9 or s not in close or pd.isna(close.at[d, s]):
                continue
            u = LEVERAGED.get(s, s)
            e = q * close.at[d, s] * lev.get(s, 1.0)
            exposure += abs(e)
            loss += e * (worst_dn[u] if e > 0 else worst_up[u])
        rows.append((d, exposure, loss))
    return pd.DataFrame(rows, columns=["day", "exposure", "stress_loss"]).set_index("day"), worst_dn


def stats_block(pnl):
    pnl = pd.Series(pnl)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "n": len(pnl), "total": pnl.sum(), "win_rate": len(wins) / len(pnl) if len(pnl) else np.nan,
        "avg_win": wins.mean() if len(wins) else 0.0, "avg_loss": losses.mean() if len(losses) else 0.0,
        "profit_factor": wins.sum() / -losses.sum() if len(losses) else np.inf,
        "worst": pnl.min(), "best": pnl.max(),
    }


def main():
    a = parse_args()
    prices = load_prices(a.prices)
    calendar = trading_calendar(prices)
    bars = {s: g.set_index("date")[["open", "high", "low", "close"]] for s, g in prices.groupby("symbol")}
    close_px = {s: b["close"].to_dict() for s, b in bars.items()}

    fills = pd.read_csv(a.trades)
    fills["time"] = pd.to_datetime(fills["time"], utc=True).dt.tz_convert(ET)
    fills = fills.sort_values("time").reset_index(drop=True)
    fills["fees"] = fills["fees"].fillna(0.0)
    fills["day"] = assign_session_day(fills["time"], calendar)
    if a.raw_closes:
        fills, n_adj = adjust_fills_for_splits(fills, prices, a.raw_closes, close_px)
        print(f"按合股/拆股口径换算的成交: {n_adj} 笔")

    start_day = pd.Timestamp(a.start)
    prev_day = calendar[calendar < start_day][-1]
    start_time = prev_day.tz_localize(ET) + pd.Timedelta(hours=16)
    opening = []
    for spec in filter(None, a.opening.split(",")):
        s, q, c = spec.split(":")
        opening.append((s, float(q), float(c)))
    # mark opening positions at the prior close so the report measures P&L earned inside the window
    opening_marked = [(s, q, close_px[s][prev_day]) for s, q, _ in opening]

    pieces, episodes, open_left = build_round_trips(fills, opening_marked, start_time)
    pieces["entry_day"] = pieces["entry_day"].where(pieces["entry_hours"] != "opening", prev_day)
    if open_left:
        print("WARNING: positions still open after last fill:", open_left)

    fees = fills["fees"].sum()
    gross = pieces["pnl"].sum()
    opening_cost_adj = sum(q * (close_px[s][prev_day] - c) for s, q, c in opening)
    print(f"=== 交易日志回测 {start_day.date()} → {fills['day'].max().date()} ===")
    print(f"成交笔数: {len(fills)} | 买入额 ${fills.loc[fills.side == 'buy', 'qty'].mul(fills.avg_price).sum():,.0f}")
    print(f"窗口内盈亏(期初持仓按{prev_day.date()}收盘价计): ${gross - fees:,.0f}  (含费用 ${fees:,.0f})")
    print(f"按真实成本价计的已实现盈亏: ${gross - fees + opening_cost_adj:,.0f}")

    # --- leverage verification
    print("\n杠杆ETF校验(日收益对标的回归斜率):")
    for etf, (und, beta, corr, n) in leverage_check(prices).items():
        print(f"  {etf:5s} → {und:5s}  斜率 {beta:+.2f}  相关 {corr:.2f}  (n={n})")

    # --- episode-level stats
    episodes["hold_days"] = (episodes["end"] - episodes["start"]).dt.total_seconds() / 86400
    episodes["net"] = episodes["pnl"] - episodes["fees"]
    episodes["theme"] = episodes["symbol"].map(theme_of)
    episodes["kind"] = np.where(episodes["symbol"].isin(LEVERAGED), "杠杆ETF", "正股")
    bearish = episodes["symbol"].isin(INVERSE) ^ (episodes["dir"] == "short")
    episodes["view"] = np.where(bearish, "看空标的", "看多标的")
    s = stats_block(episodes["net"])
    print(f"\n=== 回合统计 (一个回合 = 从空仓到再次空仓) ===")
    print(f"回合数 {s['n']} | 胜率 {s['win_rate']:.0%} | 平均盈利 ${s['avg_win']:,.0f} | 平均亏损 ${s['avg_loss']:,.0f} | "
          f"盈亏比 {s['avg_win'] / -s['avg_loss']:.2f} | 利润因子 {s['profit_factor']:.2f}")
    print(f"最大单回合盈利 ${s['best']:,.0f} | 最大单回合亏损 ${s['worst']:,.0f}")
    top = episodes.sort_values("net", ascending=False)
    print(f"前5大盈利回合合计 ${top['net'].head(5).sum():,.0f} = 总盈亏的 {top['net'].head(5).sum() / episodes['net'].sum():.0%}")

    def group_table(col, label):
        g = episodes.groupby(col)["net"].agg(["count", "sum", lambda x: (x > 0).mean(), "min", "max"])
        g.columns = ["回合", "盈亏", "胜率", "最差", "最好"]
        print(f"\n按{label}:")
        print(g.sort_values("盈亏", ascending=False).to_string(
            formatters={"盈亏": "{:,.0f}".format, "胜率": "{:.0%}".format, "最差": "{:,.0f}".format,
                        "最好": "{:,.0f}".format}))

    group_table("theme", "主题")
    group_table("kind", "品种")
    group_table("view", "实际看多/看空(反向ETF计为看空)")
    close = prices.pivot(index="date", columns="symbol", values="close")
    episodes["setup"] = entry_setup(episodes, calendar, close)
    group_table("setup", "入场形态")
    episodes["hold_bucket"] = pd.cut(episodes["hold_days"], [-1, 0.75, 3, 10, 1e9],
                                     labels=["日内(<18h)", "1-3天", "3-10天", ">10天"])
    group_table("hold_bucket", "持仓时长")
    group_table("entry_hours", "入场时段")
    episodes["size_bucket"] = pd.cut(episodes["max_cost"], [0, 50e3, 150e3, 300e3, 1e12],
                                     labels=["<$5万", "$5-15万", "$15-30万", ">$30万"])
    group_table("size_bucket", "最大仓位")

    win_hold = episodes.loc[episodes.net > 0, "hold_days"].median()
    loss_hold = episodes.loc[episodes.net < 0, "hold_days"].median()
    print(f"\n持有时长中位数: 盈利回合 {win_hold:.2f} 天 vs 亏损回合 {loss_hold:.2f} 天")

    print("\n亏损最大的8个回合:")
    worst = episodes.sort_values("net").head(8)
    for e in worst.to_dict("records"):
        print(f"  {e['start'].strftime('%m-%d')}→{e['end'].strftime('%m-%d')} {e['symbol']:5s} {e['dir']:5s} "
              f"最大仓位 ${e['max_cost']:,.0f}  持有 {e['hold_days']:.1f}天  盈亏 ${e['net']:,.0f}")
    print("\n盈利最大的8个回合:")
    for e in top.head(8).to_dict("records"):
        print(f"  {e['start'].strftime('%m-%d')}→{e['end'].strftime('%m-%d')} {e['symbol']:5s} {e['dir']:5s} "
              f"最大仓位 ${e['max_cost']:,.0f}  持有 {e['hold_days']:.1f}天  盈亏 ${e['net']:,.0f}")

    # --- daily mark-to-market curve
    mtm = daily_mark_to_market(fills, opening_marked, start_day, calendar, close_px)
    daily = mtm["total"].diff().fillna(mtm["total"].iloc[0])
    dd = mtm["total"] - mtm["total"].cummax()
    print(f"\n=== 逐日盯市 ({len(mtm)} 个交易日) ===")
    print(f"盈利天数占比 {(daily > 0).mean():.0%} | 最好一天 ${daily.max():,.0f} ({daily.idxmax().date()}) | "
          f"最差一天 ${daily.min():,.0f} ({daily.idxmin().date()})")
    print(f"累计盈亏最大回撤 ${dd.min():,.0f} (谷底 {dd.idxmin().date()}) | "
          f"隔夜总敞口中位数 ${mtm['gross_exposure'].median():,.0f}, 最大 ${mtm['gross_exposure'].max():,.0f}")
    weekly = daily.resample("W-FRI").sum()
    print(f"周度盈亏: 正周 {(weekly > 0).sum()} / 负周 {(weekly < 0).sum()} | 最差一周 ${weekly.min():,.0f}")

    stress, worst_dn = stress_test(fills, opening_marked, calendar, close, start_day - pd.DateOffset(years=1))
    print("\n=== 压力测试: 隔夜持仓遇到各标的过去一年最差单日 ===")
    print("最差单日: " + ", ".join(f"{u} {worst_dn[u]:+.1%}" for u in ["SNDK", "NBIS", "SPCX", "INTC", "MU"] if u in worst_dn))
    for d, r in stress.sort_values("stress_loss").head(5).iterrows():
        print(f"  {d.date()}  等效标的敞口 ${r.exposure:,.0f}  → 压力亏损 ${r.stress_loss:,.0f}")

    # --- rule replays on the same trades
    base = pieces["pnl"].sum()
    print("\n=== 规则重放 (同一批交易, 只改出场规则) ===")
    print(f"{'规则':<22}{'总盈亏':>14}{'变化':>12}{'最大单笔亏损':>14}")
    print(f"{'实际':<22}{base:>14,.0f}{0:>12,.0f}{pieces['pnl'].min():>14,.0f}")
    for sp in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
        r = replay_stop(pieces, bars, sp)
        print(f"{f'硬止损 -{sp:.0%}':<22}{r.sum():>14,.0f}{r.sum() - base:>+12,.0f}{r.min():>14,.0f}")
    for nd in [1, 2, 3, 5, 10]:
        r = replay_time_stop(pieces, bars, calendar, nd)
        print(f"{f'最长持有 {nd} 个交易日':<22}{r.sum():>14,.0f}{r.sum() - base:>+12,.0f}{r.min():>14,.0f}")
    for cap in [100e3, 200e3, 300e3]:
        scaled = (episodes["net"] * np.minimum(1, cap / episodes["max_cost"].clip(lower=1))).sum()
        print(f"{f'单回合仓位上限 ${cap / 1e3:.0f}k':<22}{scaled:>14,.0f}{scaled - episodes['net'].sum():>+12,.0f}")
    chase = episodes["setup"].str.startswith("追多")
    kept = episodes.loc[~chase, "net"]
    print(f"{'不做追多(只抄底+看空)':<22}{kept.sum():>14,.0f}{kept.sum() - episodes['net'].sum():>+12,.0f}"
          f"{kept.min():>14,.0f}  (回合级)")

    if a.out:
        pieces.to_csv(f"{a.out}/lots.csv", index=False)
        episodes.to_csv(f"{a.out}/episodes.csv", index=False)
        mtm.to_csv(f"{a.out}/daily_mtm.csv")


if __name__ == "__main__":
    main()
