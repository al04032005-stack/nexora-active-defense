#!/usr/bin/env python3
"""Backtest of the emailed "Stock Dashboard" daily list (Top 5 picks per day).

Each pick carries Price, Buy @, Stop, Target and Conf. Two ways of following it are tested:

1. Next-open entry: buy at the first regular-session open after the email was sent,
   measure the return to the close 1, 5, 10 and 20 trading days later, and compare with
   SPY bought and sold at the same times.
2. Bracket: a buy order at "Buy @" (a limit when it is below the quoted price, a buy-stop
   when above), valid 5 trading days from the first session after the email, then exit at
   Stop or Target, or at the close of the 20th trading day. If one daily bar touches both
   levels the stop is assumed to come first. Only brackets with stop < buy @ < target count.

Usage:
    python3 backtest/daily_list.py --picks daily_list.csv --prices prices.csv.gz
"""
import argparse

import numpy as np
import pandas as pd

from common import load_prices

ET = "America/New_York"
HORIZONS = [1, 5, 10, 20]
ENTRY_WINDOW = 5
MAX_HOLD = 20


def load_picks(path):
    p = pd.read_csv(path)
    p["sent"] = pd.to_datetime(p["sent_utc"], utc=True).dt.tz_convert(ET)
    p = p[(p["rank"] > 0) & p["ticker"].notna()].copy()
    # one email per report date: the last one sent (the first days had several test sends)
    last = p.groupby("report_date")["sent"].transform("max")
    return p[p["sent"] == last].reset_index(drop=True)


def first_session_after(sent, calendar):
    """Index of the first trading day whose 09:30 open is after the send time."""
    opens = pd.DatetimeIndex([d.tz_localize(ET) + pd.Timedelta(hours=9, minutes=30) for d in calendar])
    return int(opens.searchsorted(sent, side="right"))


def bracket(bars, start_i, buy_at, stop, target, quoted):
    """Simulate the bracket from bar index start_i. Returns (status, return, days held)."""
    o, h, l, c = (bars[k].values for k in ("open", "high", "low", "close"))
    n = len(c)
    fill_i, fill_px = None, None
    for i in range(start_i, min(start_i + ENTRY_WINDOW, n)):
        if buy_at <= quoted and l[i] <= buy_at:
            fill_i, fill_px = i, min(o[i], buy_at)
            break
        if buy_at > quoted and h[i] >= buy_at:
            fill_i, fill_px = i, max(o[i], buy_at)
            break
    if fill_i is None:
        return "not_filled", np.nan, 0
    for i in range(fill_i, min(fill_i + MAX_HOLD, n)):
        hit_stop, hit_target = l[i] <= stop, h[i] >= target
        if hit_stop:
            px = min(o[i], stop) if i > fill_i else stop
            return "stop", px / fill_px - 1, i - fill_i + 1
        if hit_target:
            px = max(o[i], target) if i > fill_i else target
            return "target", px / fill_px - 1, i - fill_i + 1
    last_i = min(fill_i + MAX_HOLD, n) - 1
    status = "time_exit" if fill_i + MAX_HOLD <= n else "open"
    return status, c[last_i] / fill_px - 1, last_i - fill_i + 1


def summarize(df, col, label):
    g = df.groupby(col, observed=True).agg(笔数=("ret_5", "count"), 五日平均=("ret_5", "mean"),
                                           五日超额=("ex_5", "mean"),
                                           五日胜率=("ret_5", lambda x: (x.dropna() > 0).mean()),
                                           二十日超额=("ex_20", "mean"))
    print(f"\n按{label}:")
    print(g.to_string(formatters={"五日平均": "{:+.2%}".format, "五日超额": "{:+.2%}".format,
                                  "五日胜率": "{:.0%}".format, "二十日超额": "{:+.2%}".format}))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--picks", required=True)
    a.add_argument("--prices", required=True)
    a.add_argument("--out", default="")
    args = a.parse_args()

    prices = load_prices(args.prices)
    bars = {s: g.set_index("date") for s, g in prices.groupby("symbol")}
    calendar = bars["SPY"].index
    spy = bars["SPY"]
    picks = load_picks(args.picks)
    missing = sorted(set(picks.ticker) - set(bars))
    if missing:
        print("没有行情数据的代码(已跳过):", missing)
    print(f"清单天数 {picks.report_date.nunique()} | 选股 {len(picks)} 条 | "
          f"{picks.report_date.min()} → {picks.report_date.max()}")

    rows = []
    for p in picks.to_dict("records"):
        if p["ticker"] not in bars:
            continue
        b = bars[p["ticker"]]
        si = first_session_after(p["sent"], calendar)
        if si >= len(calendar):
            continue
        d0 = calendar[si]
        if d0 not in b.index:
            continue
        bi = b.index.get_loc(d0)
        entry, spy_entry = b["open"].iat[bi], spy.at[d0, "open"]
        r = dict(p)
        r["entry_day"] = d0
        for hzn in HORIZONS:
            j = bi + hzn - 1
            sj = si + hzn - 1
            if j < len(b) and sj < len(calendar) and b.index[j] == calendar[sj]:
                r[f"ret_{hzn}"] = b["close"].iat[j] / entry - 1
                r[f"ex_{hzn}"] = r[f"ret_{hzn}"] - (spy["close"].iat[sj] / spy_entry - 1)
            else:
                r[f"ret_{hzn}"] = r[f"ex_{hzn}"] = np.nan
        valid = p["stop"] < p["buy_at"] < p["target"]
        r["bracket_valid"] = valid
        if valid:
            status, ret, held = bracket(b, bi, p["buy_at"], p["stop"], p["target"], p["price"])
        else:
            status, ret, held = "invalid", np.nan, 0
        r["bracket_status"], r["bracket_ret"], r["bracket_days"] = status, ret, held
        c = b["close"]
        sma50 = c.rolling(50).mean()
        prev = bi - 1
        r["above_sma50"] = bool(c.iat[prev] > sma50.iat[prev]) if prev >= 50 else np.nan
        r["ret5_before"] = c.iat[prev] / c.iat[prev - 5] - 1 if prev >= 5 else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)

    print("\n=== 方式一: 收到邮件后的下一个开盘买入 ===")
    print(f"{'持有':<8}{'笔数':>6}{'平均收益':>10}{'中位数':>9}{'胜率':>7}{'平均超额(对SPY)':>16}{'超额胜率':>9}")
    for hzn in HORIZONS:
        x = df[[f"ret_{hzn}", f"ex_{hzn}"]].dropna()
        print(f"{f'{hzn}个交易日':<8}{len(x):>6}{x.iloc[:, 0].mean():>+10.2%}{x.iloc[:, 0].median():>+9.2%}"
              f"{(x.iloc[:, 0] > 0).mean():>7.0%}{x.iloc[:, 1].mean():>+16.2%}{(x.iloc[:, 1] > 0).mean():>9.0%}")

    summarize(df, "decision", "信号强度")
    summarize(df, "rank", "当天排名")
    df["conf_bucket"] = pd.cut(df["conf"], [0, 49.5, 52.5, 55.5, 100], labels=["<50%", "50-52%", "53-55%", ">55%"])
    summarize(df, "conf_bucket", "置信度")
    df["setup"] = np.where(df["above_sma50"] == True, "50日线上方(追涨)", np.where(df["above_sma50"] == False, "50日线下方(抄底)", "n/a"))
    summarize(df, "setup", "入选时相对50日线的位置")
    top = df["ticker"].value_counts().head(10).index
    summarize(df[df.ticker.isin(top)], "ticker", "最常入选的10只")

    print("\n=== 方式二: 按邮件的 Buy @ / Stop / Target 挂单 ===")
    print(f"止损-买入-目标价顺序不合理(无法照做)的: {(~df.bracket_valid).sum()} 条 / {len(df)} 条")
    v = df[df.bracket_valid]
    print(v["bracket_status"].value_counts().to_string())
    done = v[v.bracket_status.isin(["stop", "target", "time_exit"])]
    if len(done):
        print(f"已结束的 {len(done)} 笔: 平均每笔 {done.bracket_ret.mean():+.2%}, 胜率 {(done.bracket_ret > 0).mean():.0%}, "
              f"打到目标 {(done.bracket_status == 'target').mean():.0%}, 打到止损 {(done.bracket_status == 'stop').mean():.0%}")
        conf_hit = done.groupby("conf_bucket", observed=True).apply(lambda g: (g.bracket_status == "target").mean())
        print("各置信度档打到目标的比例:", {str(k): f"{val:.0%}" for k, val in conf_hit.items()})

    breadth = df.groupby("report_date").agg(n_buy=("n_buy", "first"), spy_ex5=("ret_5", "size"))
    spy5 = df.groupby("report_date").apply(lambda g: g["ret_5"].iloc[0] - g["ex_5"].iloc[0])
    breadth["spy_5d"] = spy5
    corr = breadth[["n_buy", "spy_5d"]].dropna().corr().iloc[0, 1]
    print(f"\n当天 BUY 信号个数 与 SPY 之后5日涨跌 的相关系数: {corr:+.2f} (天数 {breadth.spy_5d.notna().sum()})")

    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
