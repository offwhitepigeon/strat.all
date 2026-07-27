# -*- coding: utf-8 -*-
"""
ETF配置模块 - 22只多行业ETF标的池
===================================

设计原则：
1. 红利标的仅保留1只（510880 红利ETF）
2. 覆盖23种不同行业/概念/大盘指数
3. 每个标的分配独特的算法类型
4. 追求T+3/T+5胜率（3/5日内最高价收益>0.5%）

算法类型说明：
- dividend_value:      红利估值型（PE/PB分位+股息率+均值偏离）
- broad_reversal:      宽基均值回归型（RSI超卖+布林带+连跌）
- trend_pullback:      趋势回踩型（上升趋势中回踩均线）
- extreme_reversal:    极端反转型（极端RSI+Z-score+量能恐慌）
- momentum_pullback:   动量回踩型（正动量+短期回踩）
- support_rebound:      支撑反弹型（关键支撑+MACD背离）
- seasonal_value:       季节估值型（季节性规律+超卖）
- financial_value:      金融价值型（PB分位+股息率+RSI）
- volatility_breakout:  波动率突破型（ATR收缩+突破）
- cycle_momentum:       周期动量型（商品周期+动量+超卖）
- premium_rate:         溢价率套利型（折价率+超卖+Z-score）
- gold_pair_reversal:   黄金股-黄金组合反弹型（黄金趋势确认+相对超跌+动能衰减）★新增
- oil_pair_reversal:    石油组合反弹型（原油趋势确认+价格区间+超卖T+5）★新增
- biotech_trend_pullback: 生物科技趋势回踩型（趋势回踩+KDJ+放量, 标普生物科技专属）★新增
- gold_support_rebound:  黄金支撑反弹型（支撑位+RSI+MACD背离+MA200趋势, 黄金专属）★新增
- new_energy_reversal:   新能源超卖反弹型（RSI+MA200偏离+Z-score+KDJ+量能+动量, 新能源专属）★新增
- dividend_yield_reversal: 股息率超卖反弹型（RSI+MA200偏离+Z-score+布林带+股息率分位, 红利专属）★新增
- robot_reversal:        机器人宽RSI反转型（放宽RSI+布林带+连跌+KDJ+Z-score+MA60偏离, 机器人专属）★新增
- pharma_reversal:       创新药反转型（broad_reversal核心+MA200趋势过滤, 创新药专属）★新增
- wine_value_reversal:   白酒价值季节型（financial_value核心+白酒季节性因子, 酒ETF专属）★新增
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ETFTarget:
    """ETF标配置"""
    code: str                    # 代码（如 sh512890）
    symbol: str                  # 纯数字代码（如 512890）
    name: str                    # 名称
    market: str                  # 市场: A股/港股/美股
    sector: str                  # 行业/概念分类
    algorithm: str               # 算法类型
    description: str = ""        # 备注说明
    hold_days: int = 3           # T+N胜率评估天数（默认T+3，石油用T+5）


# ===== 22只ETF标的池 =====
ETF_POOL: List[ETFTarget] = [
    # --- 红利类（1只）---
    ETFTarget(
        code="sh510880", symbol="510880", name="红利ETF",
        market="A股", sector="红利",
        algorithm="dividend_yield_reversal",
        description="上证红利指数，股息率分位+超卖反弹策略"
    ),

    # --- 大盘宽基（3只）---
    ETFTarget(
        code="sh510300", symbol="510300", name="沪深300ETF",
        market="A股", sector="大盘宽基",
        algorithm="broad_reversal",
        description="沪深300大盘蓝筹，均值回归特性强"
    ),
    ETFTarget(
        code="sz159782", symbol="159782", name="双创50ETF",
        market="A股", sector="中盘宽基",
        algorithm="broad_reversal",
        description="科创+创业板50，高波动适合超卖反弹"
    ),

    # --- 中概互联（1只）---
    ETFTarget(
        code="sh513050", symbol="513050", name="中概互联ETF",
        market="美股", sector="中概互联",
        algorithm="extreme_reversal",
        description="中概互联网，极端反转策略"
    ),

    # --- 科技硬件/软件（5只）---
    ETFTarget(
        code="sh512480", symbol="512480", name="半导体ETF",
        market="A股", sector="科技硬件",
        algorithm="momentum_pullback",
        description="半导体芯片，高动量回踩买入"
    ),
    ETFTarget(
        code="sh512980", symbol="512980", name="传媒ETF",
        market="A股", sector="传媒",
        algorithm="momentum_pullback",
        description="传媒板块，动量回踩策略"
    ),
    ETFTarget(
        code="sh515880", symbol="515880", name="通信ETF",
        market="A股", sector="通信",
        algorithm="momentum_pullback",
        description="通信设备板块，动量回踩策略"
    ),
    ETFTarget(
        code="sz159537", symbol="159537", name="信创ETF",
        market="A股", sector="信创",
        algorithm="momentum_pullback",
        description="信息技术应用创新，动量回踩"
    ),
    ETFTarget(
        code="sh513310", symbol="513310", name="中韩半导体ETF",
        market="A股", sector="跨境半导体",
        algorithm="premium_rate",
        description="中韩半导体，跨境折溢价套利★新增"
    ),

    # --- 医药/创新药/生物科技（2只）---
    ETFTarget(
        code="sz159502", symbol="159502", name="标普生物科技ETF",
        market="美股", sector="生物科技",
        algorithm="biotech_trend_pullback",
        description="标普生物科技，趋势回踩+KDJ+放量优化策略★专属"
    ),
    ETFTarget(
        code="sz159992", symbol="159992", name="创新药ETF",
        market="A股", sector="创新药",
        algorithm="pharma_reversal",
        description="创新药板块，宽基反转+MA200趋势过滤"
    ),

    # --- 酒（1只）---
    ETFTarget(
        code="sh512690", symbol="512690", name="酒ETF",
        market="A股", sector="白酒",
        algorithm="wine_value_reversal",
        description="白酒板块，价值+季节性因子优化策略"
    ),

    # --- 金融（2只）---
    ETFTarget(
        code="sh512000", symbol="512000", name="券商ETF",
        market="A股", sector="券商",
        algorithm="financial_value",
        description="券商板块，PB分位+RSI超卖"
    ),

    # --- 新能源/高端制造（2只）---
    ETFTarget(
        code="sh516160", symbol="516160", name="新能源ETF",
        market="A股", sector="新能源",
        algorithm="new_energy_reversal",
        description="新能源板块，超卖反弹策略（RSI+MA200+KDJ+量能）"
    ),
    ETFTarget(
        code="sh562500", symbol="562500", name="机器人ETF",
        market="A股", sector="机器人",
        algorithm="robot_reversal",
        description="机器人产业，宽RSI反转+MA60偏离"
    ),

    # --- 资源周期（4只）---
    ETFTarget(
        code="sh512400", symbol="512400", name="有色金属ETF",
        market="A股", sector="有色金属",
        algorithm="volatility_breakout",
        description="有色金属，波动率突破策略（原cycle_momentum优化替换）"
    ),
    ETFTarget(
        code="sh515220", symbol="515220", name="煤炭ETF",
        market="A股", sector="煤炭",
        algorithm="cycle_momentum",
        description="煤炭板块，周期动量策略"
    ),
    ETFTarget(
        code="sh517520", symbol="517520", name="黄金股ETF",
        market="A股", sector="黄金",
        algorithm="gold_pair_reversal",
        description="黄金股票，黄金股-黄金组合反弹策略★专属"
    ),
    ETFTarget(
        code="sh518880", symbol="518880", name="黄金ETF",
        market="A股", sector="黄金",
        algorithm="gold_support_rebound",
        description="华安黄金，追踪AU99.99，支撑反弹+MA200趋势确认★专属"
    ),
    ETFTarget(
        code="sh560710", symbol="560710", name="船舶ETF",
        market="A股", sector="船舶",
        algorithm="cycle_momentum",
        description="船舶制造，周期动量策略★新增"
    ),

    # --- 石油（2只）---
    ETFTarget(
        code="sz162411", symbol="162411", name="石油LOF",
        market="A股", sector="石油",
        algorithm="oil_pair_reversal",
        description="华宝石油，原油趋势确认+超卖反弹(T+5)★新增",
        hold_days=5
    ),
    ETFTarget(
        code="sz161129", symbol="161129", name="标普油气ETF",
        market="A股", sector="石油",
        algorithm="oil_pair_reversal",
        description="标普油气，原油趋势确认+超卖反弹(T+5)★新增",
        hold_days=5
    ),
]


def get_etf_by_code(code: str) -> ETFTarget:
    """根据代码获取ETF配置"""
    for etf in ETF_POOL:
        if etf.code == code or etf.symbol == code:
            return etf
    raise ValueError(f"ETF代码不存在: {code}")


def get_etfs_by_algorithm(algorithm: str) -> List[ETFTarget]:
    """根据算法类型获取ETF列表"""
    return [etf for etf in ETF_POOL if etf.algorithm == algorithm]


def get_algorithm_summary() -> dict:
    """获取算法分布统计"""
    summary = {}
    for etf in ETF_POOL:
        if etf.algorithm not in summary:
            summary[etf.algorithm] = []
        summary[etf.algorithm].append(etf.name)
    return summary


def get_cross_border_etfs() -> List[ETFTarget]:
    """获取跨境ETF（有折溢价数据）列表"""
    cross_border_sectors = {"跨境半导体", "海外科技", "海外指数", "中概互联", "生物科技"}
    return [etf for etf in ETF_POOL if etf.sector in cross_border_sectors]


if __name__ == '__main__':
    print(f"ETF总数: {len(ETF_POOL)}")
    print(f"\n算法分布:")
    for algo, names in get_algorithm_summary().items():
        print(f"  {algo}: {len(names)}只 - {', '.join(names)}")
    print(f"\n跨境ETF（含折溢价）:")
    for etf in get_cross_border_etfs():
        print(f"  {etf.code} {etf.name} - {etf.sector}")
