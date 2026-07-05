#!/usr/bin/env python3
"""
SNDK 蒙特卡洛价格模拟器
=======================
用法:
    python3 sndk_monte_carlo.py --price 1850
    python3 sndk_monte_carlo.py --price 1850 --days 20 --bull 0.40 --base 0.40 --bear 0.20
    python3 sndk_monte_carlo.py --price 1850 --vol 0.055 --shares 115 --cost 2120

每次关键事件后(海力士ADR上市/美联储纪要/财报)更新 --price 和三个剧本权重重跑。
剧本含义:
    bull  周期延续,资金回流    (漂移 +0.8%/日)
    base  高位震荡消化估值      (漂移  0.0%/日)
    bear  周期见顶,估值压缩    (漂移 -1.0%/日)
"""
import argparse, math, random

p = argparse.ArgumentParser()
p.add_argument("--price", type=float, required=True, help="当前价格")
p.add_argument("--days", type=int, default=20, help="剩余交易日")
p.add_argument("--vol", type=float, default=0.055, help="日波动率")
p.add_argument("--bull", type=float, default=0.35)
p.add_argument("--base", type=float, default=0.40)
p.add_argument("--bear", type=float, default=0.25)
p.add_argument("--shares", type=int, default=115, help="持仓股数")
p.add_argument("--cost", type=float, default=2120.0, help="持仓均价")
p.add_argument("--n", type=int, default=20000, help="模拟次数")
# 阶梯挂单: (类型, 数量, 价格)  类型 limit=向上限价卖 / stop=向下止损卖
p.add_argument("--ladder", type=str, default="limit:25:1900,limit:35:2120,stop:40:1480",
               help="挂单表,如 limit:25:1900,stop:40:1480")
p.add_argument("--presell", type=int, default=15, help="开盘先市价卖出的股数(按当前价成交)")
args = p.parse_args()

random.seed(42)
total = args.bull + args.base + args.bear
regimes = [(args.bull/total, 0.008), (args.base/total, 0.0), (args.bear/total, -0.010)]
orders = []
for spec in args.ladder.split(","):
    kind, qty, px = spec.strip().split(":")
    orders.append((kind, int(qty), float(px)))

def t4():
    u = random.gauss(0, 1)
    v = sum(random.gauss(0, 1) ** 2 for _ in range(4))
    return u / math.sqrt(v / 4) / math.sqrt(2.0)

COST_TOTAL = args.cost * args.shares
finals, ladder_pnl, hold_pnl = [], [], []
touch = {px: 0 for _, _, px in orders}
stop_first = 0

for _ in range(args.n):
    r, acc, mu = random.random(), 0.0, 0.0
    for prob, drift in regimes:
        acc += prob
        if r <= acc:
            mu = drift
            break
    S = args.price
    shares = args.shares - args.presell
    cash = args.presell * args.price
    live = [[k, q, px, False] for k, q, px in orders]
    first_stop_day, first_limit_day = 999, 999
    for d in range(args.days):
        S *= math.exp(mu - 0.5 * args.vol ** 2 + args.vol * t4())
        for o in live:
            kind, qty, px, done = o
            if done:
                continue
            hit = S >= px if kind == "limit" else S <= px
            if hit:
                sell = min(qty, shares)
                cash += sell * px
                shares -= sell
                o[3] = True
                touch[px] += 1
                if kind == "stop":
                    first_stop_day = min(first_stop_day, d)
                else:
                    first_limit_day = min(first_limit_day, d)
    finals.append(S)
    ladder_pnl.append(cash + shares * S - COST_TOTAL)
    hold_pnl.append(args.shares * S - COST_TOTAL)
    if first_stop_day < first_limit_day:
        stop_first += 1

finals.sort(); ladder_pnl.sort(); hold_pnl.sort()
pct = lambda arr, q: arr[int(len(arr) * q)]
N = args.n

print(f"=== SNDK 蒙特卡洛 | 现价 ${args.price:,.0f} | {args.days}个交易日 | "
      f"剧本权重 涨{regimes[0][0]:.0%}/震荡{regimes[1][0]:.0%}/跌{regimes[2][0]:.0%} ===\n")
print("月底价格分布:")
for q, label in [(0.05, "最悲观5%"), (0.25, "偏空25%"), (0.50, "中位数"),
                 (0.75, "偏多75%"), (0.95, "最乐观5%")]:
    print(f"  {label:>8}: ${pct(finals, q):,.0f}")
print(f"\n月底收在成本${args.cost:,.0f}上方概率: {sum(1 for s in finals if s >= args.cost)/N*100:.1f}%")
for kind, qty, px in orders:
    arrow = "触及" if kind == "limit" else "跌至"
    print(f"{arrow} ${px:,.0f} ({kind} {qty}股) 概率: {touch[px]/N*100:.1f}%")
print(f"止损先于任何限价单触发概率: {stop_first/N*100:.1f}%")
print(f"月底跌破$1,400概率: {sum(1 for s in finals if s < 1400)/N*100:.1f}%")
print(f"\n盈亏对比(相对总成本${COST_TOTAL:,.0f}):")
print(f"{'':>10}{'悲观5%':>13}{'中位数':>13}{'乐观5%':>13}{'平均':>13}")
print(f"{'阶梯方案':>8}: {pct(ladder_pnl,0.05):>+12,.0f} {pct(ladder_pnl,0.5):>+12,.0f} "
      f"{pct(ladder_pnl,0.95):>+12,.0f} {sum(ladder_pnl)/N:>+12,.0f}")
print(f"{'硬扛全部':>8}: {pct(hold_pnl,0.05):>+12,.0f} {pct(hold_pnl,0.5):>+12,.0f} "
      f"{pct(hold_pnl,0.95):>+12,.0f} {sum(hold_pnl)/N:>+12,.0f}")
