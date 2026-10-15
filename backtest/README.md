# 回测工具

两个脚本，数据文件都不进仓库（价格数据可随时重新拉取；成交记录是个人账户数据）。

## trade_journal.py — 用真实成交记录做回测

把券商成交记录按先进先出配成完整回合（支持做空和反向 ETF），然后：

- 按主题、品种、看多/看空、入场形态（标的相对 50 日线）、持仓时长、入场时段、仓位大小拆分盈亏
- 逐日盯市，算出最好/最差一天、累计盈亏最大回撤
- 压力测试：每天的隔夜持仓如果遇到各标的过去一年最差的一天，会亏多少
- 规则重放：同一批交易，换成硬止损、最长持有天数、仓位上限、"不追多"后的结果

```bash
python3 backtest/trade_journal.py --trades fills.csv --prices prices.csv.gz \
    --raw-closes raw_closes.csv --opening XYZ:100:50.00 --start 2026-07-01
```

- `fills.csv`：`time,symbol,side,qty,avg_price,hours,type,fees`（side 为 buy / sell / sell_short；包含部分成交后撤单的订单）
- `raw_closes.csv`：`symbol,date,raw_close`，不复权收盘价。杠杆/反向 ETF 经常合股，成交价是当时的股数口径，K 线是复权口径，用它把成交换算到同一口径
- `--opening`：窗口开始前已有的持仓，窗口内盈亏按开始前一天收盘价计

## dip_strategy.py — 把"抄底/做空"风格写成规则做 8 年回测

- 抄底：年化波动 ≥ 阈值、收盘在 50 日线下方、5 日跌幅 ≥ 阈值 → 次日开盘买入 2 倍多头
- 做空：同样条件但 5 日涨幅 ≥ 阈值 → 次日开盘买入 2 倍反向
- 出场：第一次收盘站上（做空为跌破）5 日线，或持有满 N 天
- 2 倍产品按每日重置模拟，单边成本 0.1%
- 参数网格：样本内 2018–2022 选参数，2023–2026/6 样本外验证，之后为实盘期
- 组合：最多 4 个仓位，每仓 25% 资金，逐日盯市

```bash
python3 backtest/dip_strategy.py --prices prices.csv.gz          # 网格 + 组合回测
python3 backtest/dip_strategy.py --prices prices.csv.gz --scan   # 只看最新收盘的信号
```

## 价格数据

`prices.csv.gz`：`symbol,date,open,high,low,close,volume`，日线、拆合股复权。本次数据通过 Robinhood 行情接口拉取，股票池为约 90 只美股（含 2021–2023 年大跌过的名字以减轻幸存者偏差）加上实际交易过的标的。股票池是按今天还在上市的股票选的，抄底类策略因此会被高估。
