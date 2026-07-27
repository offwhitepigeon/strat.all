# ETF 多策略量化信号系统

基于 A 股 22 只 ETF 的多策略反转/动量信号系统。每日 14:45 基于实时价格生成买入信号，追求 T+3（3 个交易日内）胜率最大化。20 种独立算法分别针对不同行业/板块特性优化，配合组合级资金管理控制风险。

## 核心概念

**信号分 0-100**，越高越推荐买入：

| 等级 | 分数区间 | 建议仓位 | 含义 |
|------|---------|---------|------|
| STRONG_BUY | 85-100 | 50% | 强烈买入 |
| BUY | 75-85 | 30% | 买入 |
| LIGHT_BUY | 60-75 | 15% | 轻仓买入 |
| WATCH | 40-60 | 5% | 关注 |
| WAIT | 0-40 | 0% | 等待 |

**胜率定义**：买入后 T+3（石油类 T+5）个交易日内的最高价 > 买入价 + 0.5%。

**组合资金管理**：
- 单只 ETF 仓位上限 50%（低于上限时可金字塔加仓）
- 组合总仓位上限 100%
- 信号发出后持仓 T+N 天，卖出后资金释放可用于新信号
- 信号冷却（可选安全网）：`cooldown_days` 参数，默认关闭，由资金管理替代

## ETF 池（22 只）

| # | 代码 | 名称 | 板块 | 算法 | T+N |
|---|------|------|------|------|-----|
| 1 | sh510880 | 红利ETF | 红利 | dividend_yield_reversal | T+3 |
| 2 | sh510300 | 沪深300ETF | 大盘宽基 | broad_reversal | T+3 |
| 3 | sz159782 | 双创50ETF | 中盘宽基 | broad_reversal | T+3 |
| 4 | sh513050 | 中概互联ETF | 中概互联 | extreme_reversal | T+3 |
| 5 | sh512480 | 半导体ETF | 科技硬件 | momentum_pullback | T+3 |
| 6 | sh512980 | 传媒ETF | 传媒 | momentum_pullback | T+3 |
| 7 | sh515880 | 通信ETF | 通信 | momentum_pullback | T+3 |
| 8 | sz159537 | 信创ETF | 信创 | momentum_pullback | T+3 |
| 9 | sh513310 | 中韩半导体ETF | 跨境半导体 | premium_rate | T+3 |
| 10 | sz159502 | 标普生物科技ETF | 生物科技 | biotech_trend_pullback | T+3 |
| 11 | sz159992 | 创新药ETF | 创新药 | pharma_reversal | T+3 |
| 12 | sh512690 | 酒ETF | 白酒 | wine_value_reversal | T+3 |
| 13 | sh512000 | 券商ETF | 证券 | financial_value | T+3 |
| 14 | sh516160 | 新能源ETF | 新能源 | new_energy_reversal | T+3 |
| 15 | sh562500 | 机器人ETF | 机器人 | robot_reversal | T+3 |
| 16 | sh512400 | 有色金属ETF | 有色金属 | volatility_breakout | T+3 |
| 17 | sh515220 | 煤炭ETF | 煤炭 | cycle_momentum | T+3 |
| 18 | sh517520 | 黄金股ETF | 黄金 | gold_pair_reversal | T+3 |
| 19 | sh518880 | 黄金ETF | 黄金 | gold_support_rebound | T+3 |
| 20 | sh560710 | 船舶ETF | 船舶 | cycle_momentum | T+3 |
| 21 | sz162411 | 石油LOF | 石油 | oil_pair_reversal | T+5 |
| 22 | sz161129 | 标普油气ETF | 石油 | oil_pair_reversal | T+5 |

## 算法体系（20 种）

每种算法针对特定板块特性设计，基于技术指标（RSI、布林带、Z-score、KDJ、MACD、MA偏离、量比、连跌等）多因子加权打分。

| 算法 | 适用ETF | 核心因子 |
|------|---------|---------|
| broad_reversal | 沪深300/双创50 | RSI超卖 + 布林带 + 连跌 |
| extreme_reversal | 中概互联 | 极端RSI + Z-score + 量能恐慌 |
| momentum_pullback | 半导体/传媒/通信/信创 | 正动量 + 短期回调 + RSI低位 |
| financial_value | 券商 | PB百分位 + RSI超卖 |
| volatility_breakout | 有色金属 | ATR收缩 + 方向确认 |
| cycle_momentum | 煤炭/船舶 | 商品周期 + 动量 + 超卖 |
| premium_rate | 中韩半导体 | 折价率 + 超卖 + Z-score |
| gold_pair_reversal | 黄金股 | 金价趋势确认 + 相对超卖 + 动量衰减 |
| gold_support_rebound | 黄金 | 支撑位 + MACD背离 + MA200趋势 |
| oil_pair_reversal | 石油LOF/标普油气 | 原油趋势 + 价格区间 + 超卖(T+5) |
| biotech_trend_pullback | 标普生物科技 | 趋势回调 + KDJ + 量能 |
| dividend_yield_reversal | 红利 | 股息率百分位 + RSI + MA偏离 |
| new_energy_reversal | 新能源 | RSI + MA200偏离 + Z-score + KDJ + 量能 |
| robot_reversal | 机器人 | 放宽RSI + 布林带 + KDJ + MA60偏离 |
| pharma_reversal | 创新药 | broad_reversal + MA200趋势过滤 |
| wine_value_reversal | 酒 | MA60/200偏离 + RSI + 白酒季节性因子 |
| dividend_value | （保留对比） | PE/PB百分位 + 股息率 |
| support_rebound | （保留对比） | 关键支撑 + MACD背离 |
| seasonal_value | （保留对比） | 季节性 + 超卖 |
| trend_pullback | （保留对比） | 上升趋势 + 回调至均线 |

> 注：dividend_value、support_rebound、seasonal_value、trend_pullback 已被各自板块的专用算法替代，保留在 ALGORITHM_MAP 中供对比测试。

## 项目结构

```
strat.all/
├── algorithms.py              # 20种算法类 + ALGORITHM_MAP
├── backtest_engine.py          # T+3胜率回测引擎
├── signal_engine.py            # 信号引擎（数据+指标+算法→信号）
├── data_engine.py              # 数据获取（akshare: 实时行情+历史K线）
├── indicators.py               # 技术指标库（RSI/MACD/BB/ATR/Z-score/KDJ...）
├── etf_config.py               # ETF池配置（22只ETF×算法映射）
├── report_generator.py         # HTML报告生成器
├── daily_run.py                # 每日定时运行 + 邮件推送
├── main.py                     # CLI入口（信号生成/回测）
├── run_intraday.py             # 盘中多次运行脚本
├── oil_data_fetch.py           # 石油数据 + COMEX原油获取
├── run_daily.bat               # Windows计划任务批处理
│
├── algo_optimization/          # 算法优化与验证脚本
│   ├── optimize_*.py           # 3种优化变体对比
│   └── verify_*.py             # 算法切换验证
│
├── backtest_2024/              # 回测与组合报告
│   ├── run_backtest.py         # 全量回测
│   ├── generate_portfolio_report.py  # 组合净值模拟报告
│   └── strategy_optimization.py     # 卖出策略优化
│
├── data/
│   ├── cache/                  # K线数据缓存（pkl, 可再生）
│   ├── backtest/               # 回测结果JSON
│   └── history/                # 每日信号历史JSON
├── reports/                    # 生成的HTML报告
├── logs/                       # 运行日志
├── mail_config.json.example    # 邮件配置模板
└── .gitignore
```

## 安装与依赖

### 环境要求

- Python 3.8+
- Windows（计划任务/盘中脚本依赖 Windows 环境）

### 依赖安装

```bash
pip install akshare pandas numpy
```

### 邮件配置

复制模板并填写 SMTP 信息：

```bash
cp mail_config.json.example mail_config.json
```

```json
{
    "smtp_server": "smtp.qq.com",
    "smtp_port": 465,
    "email_from": "your_email@qq.com",
    "email_to": "your_email@qq.com",
    "smtp_password": "your_smtp_authorization_code"
}
```

## 使用方法

### 每日定时运行

```bash
# 立即运行一次（信号生成 + 报告 + 邮件）
python daily_run.py

# 定时模式（等到14:45自动运行）
python daily_run.py --schedule

# 指定时间运行
python daily_run.py --time 14:30

# 不发邮件
python daily_run.py --no-mail

# 有信号才发邮件
python daily_run.py --conditional-mail
```

### CLI 工具

```bash
# 立即生成信号
python main.py

# 定时模式
python main.py --schedule

# 运行回测
python main.py --backtest

# 回测指定阈值
python main.py --backtest --threshold 65
```

### 盘中运行

```bash
# 开机自动运行（交易日9:45~14:45多次生成信号）
python run_intraday.py
```

盘中运行时间点：9:45、10:15、10:45、11:15、13:15、13:45、14:15（条件邮件）、14:45（最终邮件+HTML附件）。非交易日自动退出。

### Windows 计划任务

```bash
schtasks /create /tn "ETF信号系统" /tr "D:\workspace\strat.all\run_daily.bat" /sc daily /st 14:45
```

### 回测与组合报告

```bash
# 全量回测（生成JSON）
python backtest_engine.py

# 组合净值模拟报告（含资金管理）
python backtest_2024/generate_portfolio_report.py
```

## 回测系统

### T+3 胜率回测

- 回测区间：2024-01-01 至今
- 信号阈值：≥60 分
- 胜利条件：T+3 最高价收益 > 0.5%
- 按算法/ETF/等级/月份/分段统计

### 组合净值模拟

资金管理模型：
- 初始净值 1.0（2024-01-01）
- 每笔信号按建议仓位投入，T+N 收盘卖出
- 持仓以 lot 跟踪（入场日/退出日/仓位/收益）
- 单仓 ≤50%，总仓 ≤100%，T+N 卖出后资金释放
- 单仓低于 50% 时可金字塔加仓
- NAV 随仓位平仓复利累计

## 架构设计

```
DataEngine (akshare)
    ↓ K线 + 实时行情
Indicators (技术指标)
    ↓ RSI/MACD/BB/ATR/Z-score/KDJ...
Algorithms (20种算法)
    ↓ 多因子加权 → SignalResult(score, level, position_pct)
SignalEngine (信号引擎)
    ↓ 组合级资金管理过滤
ReportGenerator (HTML报告)
    ↓ signal_report_YYYYMMDD_HHMMSS.html
DailyRun (邮件推送)
```

信号引擎处理流程：
1. 加载 22 只 ETF 历史数据并计算技术指标
2. 每只 ETF 用指定算法计算信号分（0-100）
3. 组合级资金管理过滤（单仓≤50% / 总仓≤100%）
4. 按信号分排序输出

## 优化记录

| ETF | 原算法 | 优化后 | 胜率变化 | Sharpe |
|-----|--------|--------|---------|--------|
| 红利ETF | dividend_value | dividend_yield_reversal | 67%→75% | — |
| 创新药ETF | broad_reversal | pharma_reversal | 74%→91% | 1.22 |
| 机器人ETF | broad_reversal | robot_reversal | 84%→90% | 1.53 |
| 酒ETF | seasonal_value | wine_value_reversal | 77%→85% | 1.26 |
| 黄金ETF | support_rebound | gold_support_rebound | — | — |
| 新能源ETF | volatility_breakout | new_energy_reversal | — | — |

优化方法论：对每只 ETF 运行所有算法对比 → 设计 3 种优化变体 → 若变体优于原算法则创建专用算法类 → 验证后部署。

## 免责声明

本系统仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。历史回测数据不代表未来收益。
