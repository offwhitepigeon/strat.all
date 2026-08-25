# -*- coding: utf-8 -*-
"""
算法模块 - 20种买入算法
==========================

每种算法针对不同类型的ETF特征设计，追求T+3胜率：
基础算法(12种):
1. 红利估值型(dividend_value):       PE/PB分位+股息率+均线偏离
2. 宽基均值回归型(broad_reversal):     RSI超卖+布林带+连续下跌
3. 趋势回踩型(trend_pullback):         上升趋势+回踩均线+RSI回升
4. 极端反转型(extreme_reversal):       极端RSI+Z-score+量能恐慌
5. 动量回踩型(momentum_pullback):      正动量+短期回踩+RSI低位
6. 支撑反弹型(support_rebound):         关键支撑+MACD背离
7. 季节估值型(seasonal_value):          季节性规律+超卖
8. 金融价值型(financial_value):         PB分位+RSI超卖
9. 波动率突破型(volatility_breakout):   ATR收缩+方向确认
10. 周期动量型(cycle_momentum):         周期动量+超卖
11. 溢价率套利型(premium_rate):         折价率+超卖+Z-score
12. 黄金股-黄金组合反弹型(gold_pair_reversal): 黄金趋势确认+相对超跌+动能衰减
13. 石油组合反弹型(oil_pair_reversal): 原油趋势确认+价格区间+超卖(T+5)

专属优化算法(7种,基于基础算法针对特定ETF调优):
14. 生物科技趋势回踩型(biotech_trend_pullback):   trend_pullback生物科技优化版
15. 黄金支撑反弹型(gold_support_rebound):           黄金专属支撑反弹
16. 新能源超卖反弹型(new_energy_reversal):          新能源多因子超卖
17. 股息率超卖反弹型(dividend_yield_reversal):      红利专属股息率增强版
18. 机器人宽RSI反转型(robot_reversal):              机器人放宽阈值版
19. 创新药反转型(pharma_reversal):                  宽基反转+趋势过滤
20. 白酒价值季节型(wine_value_reversal):            金融价值+季节性因子

当前22只ETF使用其中16种算法, 4种为基础算法预留(trend_pullback/support_rebound/seasonal_value/dividend_value)。

信号分0-100：
  0-40:   无信号(WAIT)
  40-60:  弱信号(WATCH)
  60-75:  中等信号(LIGHT_BUY)
  75-85:  强信号(BUY)
  85-100: 极强信号(STRONG_BUY)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from indicators import (
    calc_rsi, calc_rsi_fast, calc_macd, calc_bollinger,
    calc_atr, calc_zscore, calc_moving_averages,
    calc_consecutive_days, calc_volume_ratio, calc_momentum,
    calc_kdj, calc_support_resistance, calc_price_deviation,
    calc_volatility, calc_atr_percent, calc_all_indicators
)

logger = logging.getLogger(__name__)


# ===== 信号等级定义 =====
SIGNAL_LEVELS = {
    'WAIT':       (0, 40,  0),      # (min, max, position_pct)
    'WATCH':      (40, 60, 5),      # 观察仓位5%
    'LIGHT_BUY':  (60, 75, 15),     # 轻仓15%
    'BUY':        (75, 85, 30),     # 标准30%
    'STRONG_BUY': (85, 100, 50),    # 重仓50%
}


def get_signal_level(score: float) -> Tuple[str, int]:
    """根据信号分获取信号等级和建议仓位"""
    for level, (lo, hi, pos) in SIGNAL_LEVELS.items():
        if lo <= score < hi:
            return level, pos
    return 'STRONG_BUY', 50  # score=100


@dataclass
class SignalResult:
    """信号计算结果"""
    score: float                         # 信号分0-100
    level: str                            # 信号等级
    position_pct: int                     # 建议仓位百分比
    action: str                           # 操作建议（中文）
    reasons: List[str] = field(default_factory=list)  # 信号理由
    indicators: Dict = field(default_factory=dict)      # 关键指标值
    algorithm: str = ""                   # 算法名称

    def to_dict(self) -> dict:
        return {
            'score': round(self.score, 1),
            'level': self.level,
            'position_pct': self.position_pct,
            'action': self.action,
            'reasons': self.reasons,
            'indicators': {k: round(v, 2) if isinstance(v, float) else v
                          for k, v in self.indicators.items()},
            'algorithm': self.algorithm,
        }


class BaseAlgorithm:
    """算法基类"""

    name = "base"

    def calculate(self, df: pd.DataFrame, current_price: float = None,
                 extra_data: Dict = None) -> SignalResult:
        """
        计算信号

        Args:
            df: K线数据（含技术指标列）
            current_price: 当前实时价格（14:45），为空用最近收盘价
            extra_data: 额外数据（如折价率、IOPV等实时数据），可选

        Returns:
            SignalResult
        """
        if df is None or len(df) < 60:
            return SignalResult(
                score=0, level='WAIT', position_pct=0,
                action='数据不足', algorithm=self.name
            )

        if current_price is None:
            current_price = float(df['close'].iloc[-1])

        # 取最后一行指标
        last = df.iloc[-1]
        indicators = {}
        if extra_data:
            indicators.update(extra_data)

        return self._calc_signal(df, last, current_price, indicators,
                                extra_data or {})

    def _calc_signal(self, df: pd.DataFrame, last: pd.Series, price: float,
                     indicators: Dict, extra_data: Dict = None) -> SignalResult:
        """子类实现具体信号计算"""
        raise NotImplementedError

    def _build_result(self, score: float, reasons: List[str], indicators: Dict) -> SignalResult:
        """构建信号结果"""
        level, pos = get_signal_level(score)
        action_map = {
            'WAIT': '不买入',
            'WATCH': '观察',
            'LIGHT_BUY': '轻仓买入',
            'BUY': '买入',
            'STRONG_BUY': '重仓买入',
        }
        return SignalResult(
            score=score,
            level=level,
            position_pct=pos,
            action=action_map.get(level, '不买入'),
            reasons=reasons,
            indicators=indicators,
            algorithm=self.name,
        )


# ===================================================================
# 1. 红利估值型算法（红利专属）
# ===================================================================
class DividendValueAlgorithm(BaseAlgorithm):
    """
    红利估值型 - 红利ETF专属算法（sh510880）

    核心逻辑：
    - 40周均线偏离度（极度超卖时信号强）
    - RSI超卖
    - Z-Score偏低
    - 布林带位置
    - 底背离/微背离/动能衰减检测（底部反转信号）

    v4b改进（2026-07-22）— 动能衰减检测：
    - 因子5新增RSI方向判断，解决80-89分段50%胜率问题
    - 微背离：RSI回升(今日>昨日)+收盘新低 → 8分（底部反转信号）
    - 动能衰减惩罚：RSI仍降+连跌2-3日+无背离 → 因子5=0分
      针对下跌中继信号（价格新低+RSI未回升=动能未衰竭）
    - 连跌≥4日不惩罚（暴跌末期更接近底部，如6月26日连跌4日仍胜）
    - 回测效果：80-89分50%→100%，90+保持100%，总胜率68.1%→68.2%

    v3改进（2026-07-22）：
    - MA200偏离增加-9%档位(22分)
    - 因子5重构为底背离检测：收盘价创20日新低+今日RSI>前20日最低RSI+2→10分
    - 6月30日95分(底背离)>6月26日92分，正确识别最优买点

    v2改进（2026-07-22）：
    - MA200偏离阈值从-15%/-10%/-5%调整为-12%/-8%/-5%/-3%
    - Z-Score增加-1.8档位（18分）
    - 连跌因子放宽：连跌2日给5分，放量(vol>1.2)独立给3分
    """
    name = "dividend_value"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        # 指标获取
        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        consec_down = int(last.get('consec_down', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma200': dev_ma200, 'consec_down': consec_down,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
        })

        # 因子1: RSI超卖（权重30%）
        if rsi < 25:
            s1 = 30
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s1 = 25
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s1 = 18
            reasons.append(f"RSI={rsi:.0f}，偏低")
        elif rsi < 45:
            s1 = 10
        else:
            s1 = 0
        score += s1

        # 因子2: 40周均线偏离度（权重25%）
        if dev_ma200 < -12:
            s2 = 25
            reasons.append(f"40周均线偏离{dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -9:
            s2 = 22
            reasons.append(f"偏离{dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -8:
            s2 = 20
            reasons.append(f"偏离{dev_ma200:.1f}%，明显超卖")
        elif dev_ma200 < -5:
            s2 = 12
        elif dev_ma200 < -3:
            s2 = 7
        elif dev_ma200 < 0:
            s2 = 3
        else:
            s2 = 0
        score += s2

        # 因子3: Z-Score（权重20%）
        if zscore < -2:
            s3 = 20
            reasons.append(f"Z-score={zscore:.1f}，严重低于均值")
        elif zscore < -1.8:
            s3 = 18
            reasons.append(f"Z-score={zscore:.1f}，明显低于均值")
        elif zscore < -1.5:
            s3 = 15
        elif zscore < -1:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # 因子4: 布林带位置（权重15%）
        if bb_pctb < 0.05:
            s4 = 15
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s4 = 10
        elif bb_pctb < 0.3:
            s4 = 5
        else:
            s4 = 0
        score += s4

        # 因子5: 底背离/微背离/动能衰减/新低/恐慌抛售（权重10%）
        # v4b: 新增RSI方向判断，解决80-89分段50%胜率问题
        # - 底背离(10分): 收盘新低+今日RSI>前20日最低RSI+2
        # - 微背离(8分): RSI回升(今日>昨日)+收盘新低（底部反转信号）
        # - 动能衰减惩罚(0分): RSI仍降+连跌2-3日+无背离→下跌中继信号
        #   连跌≥4日不惩罚（暴跌末期更接近底部）
        rsi_at_prev_low = None
        is_new_low_20d = False
        rsi_prev = None
        try:
            if len(df) >= 21:
                close_20d_before = float(df.iloc[-21:-1]['close'].min())
                current_close = float(df.iloc[-1]['close'])
                is_new_low_20d = current_close <= close_20d_before
                if 'rsi_14' in df.columns:
                    rsi_at_prev_low = float(df.iloc[-21:-1]['rsi_14'].min())
            if len(df) >= 2 and 'rsi_14' in df.columns:
                rsi_prev = float(df.iloc[-2]['rsi_14'])
        except (IndexError, KeyError, TypeError, ValueError):
            pass

        has_divergence = (is_new_low_20d and rsi_at_prev_low is not None
                         and rsi > rsi_at_prev_low + 2)
        rsi_rising = rsi_prev is not None and rsi > rsi_prev
        # 动能未衰竭：RSI仍降+连跌2-3日+无背离（下跌中继，非底部）
        is_falling_knife = (not has_divergence and rsi_prev is not None
                           and rsi < rsi_prev and 2 <= consec_down <= 3)

        if has_divergence:
            s5 = 10
            reasons.append(f"底背离：价格新低但RSI未创新低(当前{rsi:.0f}>前低{rsi_at_prev_low:.0f})")
        elif rsi_rising and is_new_low_20d:
            s5 = 8
            reasons.append(f"微背离：RSI回升({rsi:.0f}>{rsi_prev:.0f})但价格新低")
        elif consec_down >= 4 and vol_ratio > 1.5:
            s5 = 10
            reasons.append(f"连跌{consec_down}日且放量，恐慌性抛售")
        elif is_falling_knife and is_new_low_20d:
            s5 = 0
            reasons.append(f"动能未衰竭(连跌{consec_down}日RSI仍降)")
        elif is_new_low_20d and consec_down >= 3:
            s5 = 7
            reasons.append(f"20日新低且连跌{consec_down}日")
        elif is_new_low_20d:
            s5 = 5
            reasons.append("创20日新低")
        elif is_falling_knife:
            s5 = 0
            reasons.append(f"动能未衰竭(连跌{consec_down}日RSI仍降)")
        elif consec_down >= 3:
            s5 = 5
            reasons.append(f"连跌{consec_down}日")
        elif vol_ratio > 1.2:
            s5 = 3
            reasons.append(f"量比{vol_ratio:.1f}，放量信号")
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 2. 宽基均值回归型算法
# ===================================================================
class BroadReversalAlgorithm(BaseAlgorithm):
    """
    宽基均值回归型 - 适用于沪深300/中证500等宽基ETF

    核心逻辑：
    - RSI极端超卖
    - 布林带下轨突破
    - 连续下跌天数
    - KDJ超卖
    - 成交量放量（卖出衰竭信号）
    """
    name = "broad_reversal"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        rsi7 = float(last.get('rsi_7', 50))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        consec_down = int(last.get('consec_down', 0))
        kdj_j = float(last.get('kdj_j', 50))
        kdj_k = float(last.get('kdj_k', 50))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        dev_ma20 = float(last.get('dev_ma20', 0))
        zscore = float(last.get('zscore_20', 0))

        indicators.update({
            'rsi_14': rsi, 'rsi_7': rsi7, 'bb_percent_b': bb_pctb,
            'consec_down': consec_down, 'kdj_j': kdj_j,
            'vol_ratio': vol_ratio, 'dev_ma20': dev_ma20,
        })

        # 因子1: RSI双重超卖（权重30%）
        if rsi < 25 and rsi7 < 20:
            s1 = 30
            reasons.append(f"RSI14={rsi:.0f}/RSI7={rsi7:.0f}，双重极度超卖")
        elif rsi < 30:
            s1 = 22
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s1 = 15
        elif rsi < 40:
            s1 = 8
        else:
            s1 = 0
        score += s1

        # 因子2: 布林带（权重25%）
        if bb_pctb < 0:
            s2 = 25
            reasons.append("跌破布林带下轨，极端偏离")
        elif bb_pctb < 0.05:
            s2 = 20
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s2 = 12
        else:
            s2 = 0
        score += s2

        # 因子3: 连续下跌（权重20%）
        if consec_down >= 5:
            s3 = 20
            reasons.append(f"连续下跌{consec_down}日，超跌反弹概率大")
        elif consec_down >= 4:
            s3 = 15
        elif consec_down >= 3:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # 因子4: KDJ超卖（权重15%）
        if kdj_j < -5 and kdj_k < 20:
            s4 = 15
            reasons.append(f"KDJ J={kdj_j:.0f}，超卖")
        elif kdj_j < 10:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # 因子5: Z-score + 放量（权重10%）
        if zscore < -1.5 and vol_ratio > 1.2:
            s5 = 10
            reasons.append("Z-score偏低+放量，卖出衰竭")
        elif zscore < -1:
            s5 = 5
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 3. 趋势回踩型算法
# ===================================================================
class TrendPullbackAlgorithm(BaseAlgorithm):
    """
    趋势回踩型 - 适用于纳指/标普500等海外指数ETF

    核心逻辑：
    - 确认上升趋势（MA200之上）
    - 价格回踩MA20/MA60支撑
    - RSI从超卖区开始回升
    - MACD柱线转正（反弹信号）
    """
    name = "trend_pullback"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        close = df['close']
        rsi = float(last.get('rsi_14', 50))
        macd_hist = float(last.get('macd_hist', 0))
        ma20 = float(last.get('ma20', 0))
        ma60 = float(last.get('ma60', 0))
        ma200 = float(last.get('ma200', 0))
        dev_ma20 = float(last.get('dev_ma20', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        momentum_20 = float(last.get('momentum_20', 0))

        indicators.update({
            'rsi_14': rsi, 'macd_hist': macd_hist,
            'dev_ma20': dev_ma20, 'ma200': ma200,
            'momentum_20': momentum_20, 'bb_percent_b': bb_pctb,
        })

        # 前提：必须在MA200之上（上升趋势）
        if ma200 > 0 and price < ma200 * 0.95:
            # 跌破MA200 5%以下，趋势可能已破坏
            return self._build_result(0, ["跌破MA200 5%，趋势破坏"], indicators)

        # 因子1: 上升趋势确认（权重20%）
        if ma200 > 0 and price > ma200 and momentum_20 > 0:
            s1 = 20
            reasons.append("处于上升趋势中（MA200之上）")
        elif ma200 > 0 and price > ma200:
            s1 = 12
            reasons.append("价格在MA200上方")
        else:
            s1 = 5
        score += s1

        # 因子2: 回踩MA20/MA60（权重30%）
        if ma20 > 0:
            dev_pct = (price - ma20) / ma20 * 100
            if -3 <= dev_pct <= 1:
                s2 = 30
                reasons.append(f"精确回踩MA20（偏离{dev_pct:+.1f}%）")
            elif -5 <= dev_pct < -3:
                s2 = 25
                reasons.append(f"回踩MA20附近（偏离{dev_pct:+.1f}%）")
            elif -8 <= dev_pct < -5:
                s2 = 15
                reasons.append("略低于MA20，超卖回踩")
            elif 1 < dev_pct <= 3:
                s2 = 10
            else:
                s2 = 0
        else:
            s2 = 0
        score += s2

        # 因子3: RSI低位回升（权重20%）
        if rsi < 30:
            s3 = 20
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s3 = 15
        elif rsi < 40:
            s3 = 10
        elif rsi < 45:
            s3 = 5
        else:
            s3 = 0
        score += s3

        # 因子4: MACD柱线转正/底背离（权重15%）
        if len(df) >= 3:
            prev_hist = float(df['macd_hist'].iloc[-2])
            if macd_hist > 0 and prev_hist < 0:
                s4 = 15
                reasons.append("MACD柱线翻红，反弹信号")
            elif macd_hist > prev_hist and macd_hist < 0:
                s4 = 10
                reasons.append("MACD柱线收窄，底背离")
            elif macd_hist > 0:
                s4 = 5
            else:
                s4 = 0
        else:
            s4 = 0
        score += s4

        # 因子5: 布林带位置（权重15%）
        if bb_pctb < 0.1:
            s5 = 15
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.2:
            s5 = 8
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 4. 极端反转型算法
# ===================================================================
class ExtremeReversalAlgorithm(BaseAlgorithm):
    """
    极端反转型 - 适用于中概互联ETF

    核心逻辑：
    - 极端RSI超卖（RSI<20）
    - 极端Z-Score（Z<-2.5）
    - 恐慌性放量
    - 布林带极度偏离
    - 连续暴跌
    """
    name = "extreme_reversal"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        rsi7 = float(last.get('rsi_7', 50))
        zscore = float(last.get('zscore_20', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        consec_down = int(last.get('consec_down', 0))
        dev_ma60 = float(last.get('dev_ma60', 0))
        kdj_j = float(last.get('kdj_j', 50))

        indicators.update({
            'rsi_14': rsi, 'rsi_7': rsi7, 'zscore_20': zscore,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
            'consec_down': consec_down, 'dev_ma60': dev_ma60,
            'kdj_j': kdj_j,
        })

        # 因子1: 极端RSI（权重30%）
        if rsi < 15:
            s1 = 30
            reasons.append(f"RSI={rsi:.0f}，极度超卖（罕见）")
        elif rsi < 20:
            s1 = 25
            reasons.append(f"RSI={rsi:.0f}，极端超卖")
        elif rsi < 25:
            s1 = 18
            reasons.append(f"RSI={rsi:.0f}，严重超卖")
        elif rsi < 30:
            s1 = 10
        else:
            s1 = 0
        score += s1

        # 因子2: 极端Z-Score（权重25%）
        if zscore < -3:
            s2 = 25
            reasons.append(f"Z-score={zscore:.1f}，极端偏离（3σ）")
        elif zscore < -2:
            s2 = 20
            reasons.append(f"Z-score={zscore:.1f}，严重偏离")
        elif zscore < -1.5:
            s2 = 12
        else:
            s2 = 0
        score += s2

        # 因子3: 布林带极端偏离（权重15%）
        if bb_pctb < 0:
            s3 = 15
            reasons.append("跌破布林带下轨，极端偏离")
        elif bb_pctb < 0.05:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # 因子4: 恐慌性放量（权重15%）
        if vol_ratio > 2.0:
            s4 = 15
            reasons.append(f"量比={vol_ratio:.1f}，恐慌性放量")
        elif vol_ratio > 1.5:
            s4 = 10
        elif vol_ratio > 1.2:
            s4 = 5
        else:
            s4 = 0
        score += s4

        # 因子5: 连续暴跌+MA60偏离（权重15%）
        if consec_down >= 4 and dev_ma60 < -10:
            s5 = 15
            reasons.append(f"连跌{consec_down}日+偏离MA60 {dev_ma60:.1f}%")
        elif consec_down >= 3 and dev_ma60 < -5:
            s5 = 10
        elif dev_ma60 < -10:
            s5 = 8
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 5. 动量回踩型算法
# ===================================================================
class MomentumPullbackAlgorithm(BaseAlgorithm):
    """
    动量回踩型 - 适用于半导体/计算机/传媒等科技ETF

    核心逻辑：
    - 20日动量为正（中期上涨趋势）
    - 短期3-5日回踩（回踩幅度3-8%）
    - RSI从超买回落到中性偏低
    - 成交量缩量回踩（回调时缩量是健康信号）
    """
    name = "momentum_pullback"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        close = df['close']
        rsi = float(last.get('rsi_14', 50))
        momentum_5 = float(last.get('momentum_5', 0))
        momentum_20 = float(last.get('momentum_20', 0))
        ma60 = float(last.get('ma60', 0))
        dev_ma20 = float(last.get('dev_ma20', 0))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        bb_pctb = float(last.get('bb_percent_b', 0.5))

        indicators.update({
            'rsi_14': rsi, 'momentum_5': momentum_5,
            'momentum_20': momentum_20, 'dev_ma20': dev_ma20,
            'vol_ratio': vol_ratio, 'bb_percent_b': bb_pctb,
        })

        # 前提：中期动量必须为正
        if momentum_20 < -5:
            return self._build_result(0, ["20日动量为负，趋势向下"], indicators)

        # 因子1: 中期动量正+短期回踩（权重30%）
        if momentum_20 > 5 and momentum_5 < -3:
            s1 = 30
            reasons.append(f"20日动量+{momentum_20:.1f}%且5日回踩{momentum_5:.1f}%，健康回调")
        elif momentum_20 > 3 and momentum_5 < -2:
            s1 = 22
            reasons.append("中期上涨+短期回调")
        elif momentum_20 > 0 and momentum_5 < -1:
            s1 = 12
        else:
            s1 = 0
        score += s1

        # 因子2: RSI低位（权重25%）
        if rsi < 30:
            s2 = 25
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s2 = 18
        elif rsi < 40:
            s2 = 12
        elif rsi < 45:
            s2 = 6
        else:
            s2 = 0
        score += s2

        # 因子3: 回踩MA20（权重20%）
        if -5 <= dev_ma20 <= 0:
            s3 = 20
            reasons.append(f"回踩MA20（偏离{dev_ma20:+.1f}%）")
        elif -8 <= dev_ma20 < -5:
            s3 = 12
        elif 0 < dev_ma20 <= 2:
            s3 = 8
        else:
            s3 = 0
        score += s3

        # 因子4: 缩量回调（权重15%）
        if vol_ratio < 0.8 and momentum_5 < 0:
            s4 = 15
            reasons.append("缩量回调，卖压减弱")
        elif vol_ratio < 1.0:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # 因子5: 布林带位置（权重10%）
        if bb_pctb < 0.1:
            s5 = 10
        elif bb_pctb < 0.2:
            s5 = 5
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 6. 支撑反弹型算法
# ===================================================================
class SupportReboundAlgorithm(BaseAlgorithm):
    """
    支撑反弹型 - 适用于医药/创新药ETF

    核心逻辑：
    - 价格接近60日支撑位
    - RSI超卖
    - MACD底背离
    - 布林带下轨
    """
    name = "support_rebound"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        close = df['close']
        high = df['high']
        low = df['low']
        rsi = float(last.get('rsi_14', 50))
        macd_hist = float(last.get('macd_hist', 0))
        ma60 = float(last.get('ma60', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        zscore = float(last.get('zscore_20', 0))
        kdj_j = float(last.get('kdj_j', 50))

        # 计算60日支撑位
        if len(low) >= 60:
            support = float(low.iloc[-60:].min())
        else:
            support = float(low.min())

        # 价格接近支撑位的程度
        if support > 0 and price > 0:
            dev_support = (price - support) / support * 100
        else:
            dev_support = 100

        indicators.update({
            'rsi_14': rsi, 'macd_hist': macd_hist,
            'support_60d': support, 'dev_support': dev_support,
            'bb_percent_b': bb_pctb, 'zscore_20': zscore,
            'kdj_j': kdj_j,
        })

        # 因子1: 接近支撑位（权重30%）
        if dev_support <= 2:
            s1 = 30
            reasons.append(f"触及60日支撑位（偏离{dev_support:+.1f}%）")
        elif dev_support <= 5:
            s1 = 22
            reasons.append(f"接近60日支撑位（偏离{dev_support:+.1f}%）")
        elif dev_support <= 8:
            s1 = 12
        else:
            s1 = 0
        score += s1

        # 因子2: RSI超卖（权重25%）
        if rsi < 25:
            s2 = 25
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s2 = 20
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s2 = 12
        else:
            s2 = 0
        score += s2

        # 因子3: MACD底背离（权重20%）
        if len(df) >= 10:
            # 检查近5日MACD柱线是否在收窄
            recent_hist = df['macd_hist'].iloc[-5:].values
            if macd_hist < 0 and recent_hist[-1] > recent_hist[0]:
                s3 = 20
                reasons.append("MACD柱线收窄，底背离信号")
            elif macd_hist > 0 and float(df['macd_hist'].iloc[-2]) < 0:
                s3 = 15
                reasons.append("MACD翻红，反弹确认")
            elif macd_hist < 0 and macd_hist > float(df['macd_hist'].iloc[-3]):
                s3 = 10
            else:
                s3 = 0
        else:
            s3 = 0
        score += s3

        # 因子4: 布林带（权重15%）
        if bb_pctb < 0.05:
            s4 = 15
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # 因子5: Z-score + KDJ（权重10%）
        if zscore < -1.5 and kdj_j < 10:
            s5 = 10
            reasons.append("Z-score偏低+KDJ超卖")
        elif zscore < -1:
            s5 = 5
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 7. 季节估值型算法
# ===================================================================
class SeasonalValueAlgorithm(BaseAlgorithm):
    """
    季节估值型 - 适用于消费/酒ETF

    核心逻辑：
    - 季节性规律（消费/白酒有明显的季节效应）
    - RSI超卖
    - 20日动量为负但幅度收窄
    - KDJ超卖
    - 价格低于MA20
    """
    name = "seasonal_value"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        # 获取当前月份
        if 'date' in df.columns:
            last_date = df['date'].iloc[-1]
            if hasattr(last_date, 'month'):
                month = last_date.month
            else:
                month = int(str(last_date)[5:7])
        else:
            month = datetime.now().month

        rsi = float(last.get('rsi_14', 50))
        momentum_5 = float(last.get('momentum_5', 0))
        momentum_20 = float(last.get('momentum_20', 0))
        kdj_j = float(last.get('kdj_j', 50))
        kdj_k = float(last.get('kdj_k', 50))
        dev_ma20 = float(last.get('dev_ma20', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        consec_down = int(last.get('consec_down', 0))

        indicators.update({
            'month': month, 'rsi_14': rsi,
            'momentum_5': momentum_5, 'momentum_20': momentum_20,
            'kdj_j': kdj_j, 'dev_ma20': dev_ma20,
            'bb_percent_b': bb_pctb, 'consec_down': consec_down,
        })

        # 因子1: 季节性（权重25%）
        # 消费/白酒的季节性规律：
        # 春节前(11-2月)消费旺季，3-4月调整，5-6月企稳，7-9月淡季，10月开始预热
        seasonal_score = 0
        if month in [3, 4]:  # 春季调整，是布局机会
            seasonal_score = 25
            reasons.append("春季消费淡季调整期，季节性布局机会")
        elif month in [7, 8]:  # 夏季淡季超跌
            seasonal_score = 18
            reasons.append("夏季淡季，超卖概率高")
        elif month in [10, 11]:  # 节前预热
            seasonal_score = 10
        elif month in [12, 1, 2]:  # 旺季，回调即机会
            seasonal_score = 15
        score += seasonal_score

        # 因子2: RSI超卖（权重25%）
        if rsi < 25:
            s2 = 25
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s2 = 20
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s2 = 12
        else:
            s2 = 0
        score += s2

        # 因子3: KDJ超卖（权重20%）
        if kdj_j < 0 and kdj_k < 20:
            s3 = 20
            reasons.append(f"KDJ J={kdj_j:.0f}/K={kdj_k:.0f}，超卖")
        elif kdj_j < 10:
            s3 = 12
        elif kdj_j < 20:
            s3 = 6
        else:
            s3 = 0
        score += s3

        # 因子4: 价格低于MA20+布林带（权重15%）
        if dev_ma20 < -3 and bb_pctb < 0.15:
            s4 = 15
            reasons.append("低于MA20+布林带下轨区域")
        elif dev_ma20 < -2:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # 因子5: 连续下跌收窄（权重15%）
        if consec_down >= 3 and momentum_5 > momentum_20:
            s5 = 15
            reasons.append("短期跌势放缓，反弹在即")
        elif consec_down >= 3:
            s5 = 8
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 8. 金融价值型算法
# ===================================================================
class FinancialValueAlgorithm(BaseAlgorithm):
    """
    金融价值型 - 适用于券商/银行ETF

    核心逻辑：
    - 价格严重低于MA60/MA200
    - RSI超卖
    - Z-score偏低
    - 布林带下轨
    - 成交量放量（底部信号）
    """
    name = "financial_value"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma60 = float(last.get('dev_ma60', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        kdj_j = float(last.get('kdj_j', 50))
        consec_down = int(last.get('consec_down', 0))

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma60': dev_ma60, 'dev_ma200': dev_ma200,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
        })

        # 因子1: 严重偏离MA60/MA200（权重30%）
        if dev_ma60 < -10 and dev_ma200 < -15:
            s1 = 30
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%，严重超卖")
        elif dev_ma60 < -8:
            s1 = 22
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%")
        elif dev_ma60 < -5:
            s1 = 12
        else:
            s1 = 0
        score += s1

        # 因子2: RSI超卖（权重25%）
        if rsi < 25:
            s2 = 25
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s2 = 18
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # 因子3: Z-score（权重15%）
        if zscore < -2:
            s3 = 15
            reasons.append(f"Z-score={zscore:.1f}，严重偏低")
        elif zscore < -1.5:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # 因子4: 布林带（权重15%）
        if bb_pctb < 0.05:
            s4 = 15
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # 因子5: 放量+KDJ（权重15%）
        if vol_ratio > 1.5 and kdj_j < 10:
            s5 = 15
            reasons.append("放量+KDJ超卖，底部信号")
        elif vol_ratio > 1.2:
            s5 = 8
        elif kdj_j < 10:
            s5 = 5
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 9. 波动率突破型算法
# ===================================================================
class VolatilityBreakoutAlgorithm(BaseAlgorithm):
    """
    波动率突破型 - 适用于新能源ETF

    核心逻辑：
    - ATR波动率收缩（波动率分位偏低）
    - 价格在布林带下轨附近
    - RSI从超卖区开始回升
    - 成交量突然放大
    """
    name = "volatility_breakout"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        close = df['close']
        high = df['high']
        low = df['low']

        rsi = float(last.get('rsi_14', 50))
        rsi7 = float(last.get('rsi_7', 50))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        bb_bw = float(last.get('bb_bandwidth', 0))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        atr_pct = float(last.get('atr_pct', 0))
        zscore = float(last.get('zscore_20', 0))
        consec_down = int(last.get('consec_down', 0))

        # 计算ATR波动率的历史分位
        if 'atr_pct' in df.columns and len(df) >= 60:
            atr_series = df['atr_pct'].iloc[-60:]
            atr_rank = float((atr_series < atr_pct).sum() / len(atr_series) * 100)
        else:
            atr_rank = 50

        # 计算布林带宽度的历史分位（是否在收缩）
        if 'bb_bandwidth' in df.columns and len(df) >= 60:
            bw_series = df['bb_bandwidth'].iloc[-60:]
            bw_rank = float((bw_series < bb_bw).sum() / len(bw_series) * 100)
        else:
            bw_rank = 50

        indicators.update({
            'rsi_14': rsi, 'rsi_7': rsi7,
            'bb_percent_b': bb_pctb, 'atr_pct': atr_pct,
            'atr_rank': atr_rank, 'bw_rank': bw_rank,
            'vol_ratio': vol_ratio, 'zscore_20': zscore,
        })

        # 因子1: 波动率收缩（权重25%）
        if atr_rank < 20 and bw_rank < 30:
            s1 = 25
            reasons.append("波动率+布林带宽度收缩，临近变盘")
        elif atr_rank < 30:
            s1 = 15
            reasons.append("波动率处于低位")
        else:
            s1 = 0
        score += s1

        # 因子2: 价格在布林带下轨（权重25%）
        if bb_pctb < 0.05:
            s2 = 25
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s2 = 15
        else:
            s2 = 0
        score += s2

        # 因子3: RSI超卖（权重20%）
        if rsi < 25:
            s3 = 20
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s3 = 15
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s3 = 8
        else:
            s3 = 0
        score += s3

        # 因子4: 成交量放量（权重15%）
        if vol_ratio > 1.8:
            s4 = 15
            reasons.append(f"量比={vol_ratio:.1f}，放量信号")
        elif vol_ratio > 1.3:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # 因子5: Z-score偏低（权重15%）
        if zscore < -1.5:
            s5 = 15
            reasons.append(f"Z-score={zscore:.1f}")
        elif zscore < -1:
            s5 = 8
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 10. 周期动量型算法
# ===================================================================
class CycleMomentumAlgorithm(BaseAlgorithm):
    """
    周期动量型 - 适用于有色/煤炭ETF

    核心逻辑：
    - 中期动量为正（周期向上）
    - 短期超卖回踩
    - RSI低位
    - 价格在MA60之上（周期趋势确认）
    - KDJ超卖
    """
    name = "cycle_momentum"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        momentum_10 = float(last.get('momentum_10', 0))
        momentum_20 = float(last.get('momentum_20', 0))
        ma60 = float(last.get('ma60', 0))
        dev_ma20 = float(last.get('dev_ma20', 0))
        kdj_j = float(last.get('kdj_j', 50))
        kdj_k = float(last.get('kdj_k', 50))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))

        indicators.update({
            'rsi_14': rsi, 'momentum_10': momentum_10,
            'momentum_20': momentum_20, 'dev_ma20': dev_ma20,
            'kdj_j': kdj_j, 'bb_percent_b': bb_pctb,
            'vol_ratio': vol_ratio,
        })

        # 前提：中期动量不为负
        if momentum_20 < -8:
            return self._build_result(0, ["中期动量负，周期向下"], indicators)

        # 因子1: 周期向上+短期超卖（权重30%）
        if momentum_20 > 5 and rsi < 35:
            s1 = 30
            reasons.append(f"周期向上(20日+{momentum_20:.1f}%)且RSI超卖")
        elif momentum_20 > 0 and rsi < 40:
            s1 = 20
            reasons.append("周期偏强+超卖")
        elif momentum_20 > 0:
            s1 = 8
        else:
            s1 = 0
        score += s1

        # 因子2: RSI超卖（权重25%）
        if rsi < 25:
            s2 = 25
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s2 = 18
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s2 = 12
        else:
            s2 = 0
        score += s2

        # 因子3: KDJ超卖（权重20%）
        if kdj_j < 0 and kdj_k < 20:
            s3 = 20
            reasons.append(f"KDJ超卖(J={kdj_j:.0f})")
        elif kdj_j < 10:
            s3 = 12
        elif kdj_j < 20:
            s3 = 6
        else:
            s3 = 0
        score += s3

        # 因子4: 布林带+MA20偏离（权重15%）
        if bb_pctb < 0.1 and dev_ma20 < -3:
            s4 = 15
            reasons.append("布林带下轨+低于MA20")
        elif bb_pctb < 0.15:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # 因子5: 缩量回踩（权重10%）
        if vol_ratio < 0.8:
            s5 = 10
            reasons.append("缩量回踩")
        elif vol_ratio < 1.0:
            s5 = 5
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 11. 溢价率套利型算法
# ===================================================================
class PremiumRateAlgorithm(BaseAlgorithm):
    """
    溢价率套利型 - 适用于中韩半导体等跨境ETF

    核心逻辑：
    跨境ETF由于时区差异、汇率波动、供需失衡等原因，市场价常偏离NAV。
    折价率（正值为折价=市场价低于NAV=买入机会）是该策略的核心因子。
    
    折价率均值回归机制：
    - 跨境ETF的折价率难以通过QDII额度套利消除，但长期存在均值回归
    - 折价率>3%时，市场价远低于实际净值，买入相当于"打折购买"
    - 结合技术超卖信号（RSI/Z-score/布林带），可提升T+3反弹概率
    
    因子设计（5因子共100分）：
    1. 折价率(30%): 核心因子，正折价为买入信号
    2. RSI超卖(25%): 技术确认超卖
    3. Z-Score(20%): 价格偏离均值的统计确认
    4. 布林带位置(15%): 价格在布林带下轨附近
    5. 成交量+连跌(10%): 恐慌性抛售的底部信号
    """
    name = "premium_rate"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        consec_down = int(last.get('consec_down', 0))
        dev_ma20 = float(last.get('dev_ma20', 0))

        # 从 extra_data 获取折价率（仅实时模式有值，回测时为0或None）
        # 注意：fund_etf_spot_em 中"基金折价率"为正值=折价（买入信号），负值=溢价
        premium_rate = None
        if extra_data:
            premium_rate = extra_data.get('premium_rate')
            if premium_rate is None:
                premium_rate = extra_data.get('discount_rate')

        # 折价率转正数（正折价=买入机会）
        discount = 0.0
        if premium_rate is not None:
            try:
                discount = float(premium_rate)
            except (ValueError, TypeError):
                discount = 0.0

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
            'consec_down': consec_down, 'dev_ma20': dev_ma20,
            'discount_rate': round(discount, 2),
        })

        # 因子1: 折价率（权重30%）
        if discount > 5:
            s1 = 30
            reasons.append(f"折价率{discount:.1f}%，极度折价（罕见买入机会）")
        elif discount > 3:
            s1 = 25
            reasons.append(f"折价率{discount:.1f}%，明显折价")
        elif discount > 1.5:
            s1 = 18
            reasons.append(f"折价率{discount:.1f}%，轻度折价")
        elif discount > 0:
            s1 = 10
            reasons.append(f"折价率{discount:.1f}%，小幅折价")
        else:
            s1 = 0
            if discount < -2:
                reasons.append(f"溢价{abs(discount):.1f}%，不适宜买入")
        score += s1

        # 因子2: RSI超卖（权重25%）
        if rsi < 25:
            s2 = 25
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s2 = 20
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s2 = 12
        elif rsi < 40:
            s2 = 5
        else:
            s2 = 0
        score += s2

        # 因子3: Z-Score（权重20%）
        if zscore < -2.5:
            s3 = 20
            reasons.append(f"Z-score={zscore:.1f}，极端偏离")
        elif zscore < -2:
            s3 = 16
            reasons.append(f"Z-score={zscore:.1f}，严重偏离")
        elif zscore < -1.5:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # 因子4: 布林带位置（权重15%）
        if bb_pctb < 0:
            s4 = 15
            reasons.append("跌破布林带下轨")
        elif bb_pctb < 0.05:
            s4 = 10
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s4 = 5
        else:
            s4 = 0
        score += s4

        # 因子5: 成交量+连跌（权重10%）
        if vol_ratio > 1.5 and consec_down >= 3:
            s5 = 10
            reasons.append(f"量比{vol_ratio:.1f}+连跌{consec_down}日，恐慌抛售")
        elif vol_ratio > 1.2 and consec_down >= 2:
            s5 = 5
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 13. 黄金股-黄金组合反弹型算法（黄金股专属）
# ===================================================================
class GoldPairReversalAlgorithm(BaseAlgorithm):
    """
    黄金股-黄金组合反弹型 - 黄金股ETF专属算法（sh517520）

    核心逻辑:
    - 利用黄金ETF(sh518880)作为黄金股ETF的趋势确认信号
    - 黄金股波动率约为黄金的1.8倍，存在杠杆放大效应
    - 当黄金趋势向上 + 黄金股超跌 → 高胜率反弹机会
    - 当黄金股相对黄金严重超跌 → 均值回归机会
    - 连跌≥3日+动能未衰竭 → 下跌中继信号，封顶65分

    分析依据: 659天回测数据(2023-11~2026-07)
    - 基准T+3胜率: 78.5%
    - RSI<30+黄金在MA20上: 91.7%
    - 黄金股跌>2%+黄金跌<0.5%: 88.9%
    - 连跌3日+黄金企稳: 53.3%(下跌中继)

    v3改进（2026-07-22）:
    - F1: RSI 30-35降权(18→12), 减少非深度超卖的分数
    - F2: 黄金RSI>65时降权(黄金超买可能回调拖累黄金股)
    - F5: 底背离要求RSI<30才给满分(RSI>=30的背离不够可靠)
    - F6: 新增组合信号加成(RSI<30+黄金趋势向上→+5分)
    - 动能衰减封顶: 连跌≥3日+RSI仍降+无背离 → 封顶65分
    - 回测效果: sh517520 85.7%胜率(21信号), 70-80分段100%

    因子结构(100分):
      F1: 黄金股RSI超卖(30%) + F2: 黄金趋势确认(30%)
      F3: 相对超跌度(15%) + F4: MA200偏离(10%)
      F5: 恐慌/背离(10%) + F6: 组合加成(5%)
    """
    name = "gold_pair_reversal"

    # 类级缓存: 黄金ETF数据
    _gold_cache = None

    @classmethod
    def _get_gold_df(cls):
        """获取黄金ETF(sh518880)数据(含指标), 类级缓存"""
        if cls._gold_cache is None:
            try:
                from data_engine import DataEngine  # 延迟导入避免循环依赖
                de = DataEngine()
                df_gold = de.get_history_kline("sh518880")
                if df_gold is not None:
                    df_gold = calc_all_indicators(df_gold)
                cls._gold_cache = df_gold
            except Exception as e:
                logger.warning(f"黄金ETF数据获取失败: {e}")
                cls._gold_cache = pd.DataFrame()  # 空DataFrame防止重复请求
        return cls._gold_cache

    def _get_gold_indicators(self, current_date):
        """获取截至current_date的黄金ETF指标"""
        df_gold = self._get_gold_df()
        if df_gold is None or df_gold.empty:
            return None
        mask = df_gold['date'] <= current_date
        if not mask.any():
            return None
        return df_gold[mask].iloc[-1]

    def _get_gold_series(self, current_date, n_days=60):
        """获取截至current_date的黄金ETF最近n_days条数据"""
        df_gold = self._get_gold_df()
        if df_gold is None or df_gold.empty:
            return None
        mask = df_gold['date'] <= current_date
        if not mask.any():
            return None
        return df_gold[mask].tail(n_days)

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        # === 获取黄金股ETF自身指标 ===
        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        consec_down = int(last.get('consec_down', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))

        # === 获取黄金ETF参考数据 ===
        current_date = last['date'] if 'date' in last else None
        gold_last = self._get_gold_indicators(current_date) if current_date else None

        gold_rsi = float(gold_last.get('rsi_14', 50)) if gold_last is not None else 50
        gold_close = float(gold_last.get('close', 0)) if gold_last is not None else 0
        gold_ma20 = float(gold_last.get('ma20', 0)) if gold_last is not None else 0
        gold_above_ma20 = gold_close > gold_ma20 if gold_last is not None else False
        gold_ret = float(gold_last.get('change_pct', 0)) if gold_last is not None else 0

        # === 计算价差比Z-score (黄金股/黄金) ===
        ratio_zscore = 0
        if gold_last is not None and len(df) >= 60:
            gold_series = self._get_gold_series(current_date, 60)
            if gold_series is not None and len(gold_series) >= 20:
                gs_series = df.tail(len(gold_series))['close'].values
                g_series = gold_series['close'].values
                if len(gs_series) == len(g_series) and len(gs_series) >= 20:
                    ratios = gs_series / g_series
                    ratio_mean = np.mean(ratios)
                    ratio_std = np.std(ratios)
                    if ratio_std > 0:
                        ratio_zscore = (ratios[-1] - ratio_mean) / ratio_std

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma200': dev_ma200, 'consec_down': consec_down,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
            'gold_rsi': gold_rsi, 'gold_above_ma20': gold_above_ma20,
            'gold_ret': gold_ret, 'ratio_zscore': round(ratio_zscore, 2),
        })

        # === F1: 黄金股RSI超卖 (30%) ===
        if rsi < 25:
            s1 = 30
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s1 = 25
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s1 = 12
            reasons.append(f"RSI={rsi:.0f}，偏低")
        elif rsi < 45:
            s1 = 5
        else:
            s1 = 0
        score += s1

        # === F2: 黄金趋势确认 (30%) — 核心差异化因子 ===
        if gold_rsi > 65:
            s2 = 15  # 黄金超买，可能回调拖累黄金股
            reasons.append(f"黄金RSI={gold_rsi:.0f}偏高，注意回调风险")
        elif gold_above_ma20 and gold_rsi > 45:
            s2 = 30
            reasons.append(f"黄金强趋势(RSI={gold_rsi:.0f}+在MA20上)")
        elif gold_above_ma20:
            s2 = 25
            reasons.append(f"黄金在MA20上(趋势向上)")
        elif gold_rsi > 50:
            s2 = 15
            reasons.append(f"黄金RSI={gold_rsi:.0f}回升")
        elif gold_rsi > 45:
            s2 = 8
        elif gold_rsi < 35:
            s2 = 0  # 黄金也超卖，危险
        else:
            s2 = 3
        score += s2

        # === F3: 相对超跌度 (15%) ===
        gs_ret = float(last.get('change_pct', 0))
        if gs_ret < -2 and gold_ret > -0.5:
            s3 = 15
            reasons.append(f"黄金股跌{gs_ret:.1f}%但黄金仅跌{gold_ret:.1f}%，相对超跌")
        elif ratio_zscore < -1.5:
            s3 = 12
            reasons.append(f"价差比Z={ratio_zscore:.1f}，黄金股严重低估")
        elif gs_ret < -1 and gold_ret > 0:
            s3 = 10
            reasons.append(f"黄金股跌但黄金涨，背离")
        elif ratio_zscore < -0.5:
            s3 = 8
        elif ratio_zscore > 1:
            s3 = 6
        elif ratio_zscore > 0:
            s3 = 3
        else:
            s3 = 0
        score += s3

        # === F4: 黄金股MA200偏离 (10%) ===
        if dev_ma200 < -12:
            s4 = 10
            reasons.append(f"偏离MA200={dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -9:
            s4 = 9
        elif dev_ma200 < -8:
            s4 = 7
        elif dev_ma200 < -5:
            s4 = 5
        elif dev_ma200 < -3:
            s4 = 3
        elif dev_ma200 < 0:
            s4 = 1
        else:
            s4 = 0
        score += s4

        # === F5: 恐慌/连跌/背离 (10%) ===
        rsi_prev = None
        rsi_at_prev_low = None
        is_new_low_20d = False
        try:
            if len(df) >= 21:
                close_20d_before = float(df.iloc[-21:-1]['close'].min())
                current_close = float(df.iloc[-1]['close'])
                is_new_low_20d = current_close <= close_20d_before
                if 'rsi_14' in df.columns:
                    rsi_at_prev_low = float(df.iloc[-21:-1]['rsi_14'].min())
            if len(df) >= 2 and 'rsi_14' in df.columns:
                rsi_prev = float(df.iloc[-2]['rsi_14'])
        except (IndexError, KeyError, TypeError, ValueError):
            pass

        has_divergence = (is_new_low_20d and rsi_at_prev_low is not None
                         and rsi > rsi_at_prev_low + 2)
        rsi_rising = rsi_prev is not None and rsi > rsi_prev
        is_falling_knife = (not has_divergence and rsi_prev is not None
                           and rsi < rsi_prev and 2 <= consec_down <= 3)

        if has_divergence and rsi < 30:
            s5 = 10
            reasons.append(f"底背离(价格新低+RSI未创新低)")
        elif has_divergence:
            s5 = 5
            reasons.append(f"弱背离(价格新低+RSI未创新低但RSI={rsi:.0f}未深跌)")
        elif -3 < gs_ret <= -2:
            s5 = 8
            reasons.append(f"单日跌{gs_ret:.1f}%，恐慌抛售可反弹")
        elif consec_down >= 4 and vol_ratio > 1.5:
            s5 = 7
            reasons.append(f"连跌{consec_down}日放量，恐慌底部")
        elif rsi_rising and is_new_low_20d:
            s5 = 8
            reasons.append(f"微背离(RSI回升+价格新低)")
        elif vol_ratio > 1.5:
            s5 = 5
            reasons.append(f"量比{vol_ratio:.1f}，放量")
        elif is_falling_knife:
            s5 = 0
            reasons.append(f"动能未衰竭(连跌{consec_down}日RSI仍降)")
        elif consec_down >= 3:
            s5 = 0  # 连跌3日惩罚 (53.3%胜率)
        elif consec_down >= 2:
            s5 = 3
        else:
            s5 = 0
        score += s5

        # === F6: 组合信号加成 (5%) ===
        # 关键组合: RSI<30 + 黄金在MA20上 → 91.7%胜率
        if rsi < 30 and gold_above_ma20:
            s6 = 5
            reasons.append(f"黄金组合(RSI={rsi:.0f}+黄金趋势向上)")
        elif rsi < 25:
            s6 = 3
        elif rsi < 35 and ratio_zscore < -1:
            s6 = 3
        else:
            s6 = 0
        score += s6

        # === 动能衰减惩罚: 连跌≥3日+RSI仍降+无背离 → 封顶65分 ===
        if is_falling_knife or (consec_down >= 3 and not has_divergence and not rsi_rising):
            score = min(score, 65)
            reasons.append(f"动能衰减封顶(连跌{consec_down}日)")

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 14. 石油组合反弹型算法（石油LOF/标普油气ETF专属）
# ===================================================================
class OilPairReversalAlgorithm(BaseAlgorithm):
    """
    石油组合反弹型 - 石油LOF(sz162411)/标普油气ETF(sz161129)专属算法

    核心逻辑:
    - COMEX原油价格趋势(MA20/RSI)是最强预测因子
    - 原油<$70时ETF性价比升高, 但仅当油价止跌(在MA20上)时才有效
    - 原油在MA20下+RSI<30 → 下跌中继, 60.2%胜率, 应回避
    - 原油>=$80时ETF表现稳定(82-85.5%), 高油价=趋势确认
    - ETF自身RSI超卖+油价趋势确认 → 高胜率反弹

    分析依据: 2024-01-01起回测
    - sz161129: 89.4%胜率(47信号), 平均最大收益+3.95%
    - sz162411: 77.8%胜率(18信号), 平均最大收益+3.08%
    - 对比: trend_pullback 76.2%, broad_reversal 75.0%, extreme_reversal 80.0%
    - T+5胜率(非T+3), 因原油受战争/地缘不确定因素影响

    因子结构(100分):
      F1: ETF RSI超卖(25%) + F2: 原油趋势确认(30%)
      F3: 原油价格区间(20%) + F4: ETF MA200偏离(10%)
      F5: 恐慌/背离(10%) + F6: 组合加成(5%)
    """
    name = "oil_pair_reversal"

    # 类级缓存: COMEX原油数据
    _oil_cache = None

    @classmethod
    def _get_oil_df(cls):
        """获取COMEX原油数据(含指标), 类级缓存"""
        if cls._oil_cache is None:
            try:
                cache_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'data', 'cache', 'comex_oil_price.pkl'
                )
                oil_df = pd.read_pickle(cache_path)
                oil_df['date'] = pd.to_datetime(oil_df['date'])
                oil_df = oil_df.sort_values('date').reset_index(drop=True)

                # 计算原油指标
                oil_df['oil_ret'] = oil_df['close'].pct_change() * 100
                oil_df['oil_ma20'] = oil_df['close'].rolling(20).mean()
                oil_df['oil_above_ma20'] = oil_df['close'] > oil_df['oil_ma20']

                # 原油RSI
                delta = oil_df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                oil_df['oil_rsi'] = 100 - (100 / (1 + rs))
                oil_df['oil_rsi'] = oil_df['oil_rsi'].fillna(50).clip(0, 100)

                cls._oil_cache = oil_df
            except Exception as e:
                logger.warning(f"原油数据获取失败: {e}")
                cls._oil_cache = pd.DataFrame()
        return cls._oil_cache

    def _get_oil_indicators(self, current_date):
        """获取截至current_date的原油指标"""
        oil_df = self._get_oil_df()
        if oil_df is None or oil_df.empty:
            return None
        mask = oil_df['date'] <= current_date
        if not mask.any():
            return None
        return oil_df[mask].iloc[-1]

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        # === 获取ETF自身指标 ===
        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        consec_down = int(last.get('consec_down', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        etf_ret = float(last.get('change_pct', 0))

        # === 获取原油参考数据 ===
        current_date = last['date'] if 'date' in last else None
        oil_last = self._get_oil_indicators(current_date) if current_date else None

        oil_close = float(oil_last['close']) if oil_last is not None else 80.0
        oil_rsi = float(oil_last['oil_rsi']) if oil_last is not None else 50.0
        oil_above_ma20 = bool(oil_last['oil_above_ma20']) if oil_last is not None else False
        oil_ret = float(oil_last['oil_ret']) if oil_last is not None else 0.0

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma200': dev_ma200, 'consec_down': consec_down,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
            'oil_close': round(oil_close, 2), 'oil_rsi': round(oil_rsi, 1),
            'oil_above_ma20': oil_above_ma20, 'oil_ret': round(oil_ret, 2),
        })

        # === F1: ETF RSI超卖 (25%) ===
        if rsi < 25:
            s1 = 25
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s1 = 20
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s1 = 12
            reasons.append(f"RSI={rsi:.0f}，偏低")
        elif rsi < 45:
            s1 = 5
        else:
            s1 = 0
        score += s1

        # === F2: 原油趋势确认 (30%) — 核心因子 ===
        if oil_above_ma20 and oil_rsi > 50:
            s2 = 30
            reasons.append(f"原油强趋势(${oil_close:.1f}+RSI={oil_rsi:.0f}+在MA20上)")
        elif oil_above_ma20:
            s2 = 25
            reasons.append(f"原油在MA20上(趋势止跌)")
        elif oil_rsi > 60:
            s2 = 18
            reasons.append(f"原油RSI={oil_rsi:.0f}回升(动能恢复)")
        elif oil_rsi > 40:
            s2 = 10
        elif oil_rsi < 30:
            s2 = 0  # 原油也超卖，危险
            reasons.append(f"原油RSI={oil_rsi:.0f}超卖，避免")
        else:
            s2 = 3
        score += s2

        # === F3: 原油价格区间 (20%) ===
        if oil_close < 60 and oil_above_ma20:
            s3 = 20
            reasons.append(f"油价${oil_close:.0f}低位+止跌，性价比极高")
        elif oil_close < 70 and oil_above_ma20:
            s3 = 18
            reasons.append(f"油价${oil_close:.0f}低于70+止跌")
        elif oil_close >= 90:
            s3 = 15
            reasons.append(f"油价${oil_close:.0f}高位(趋势强确认)")
        elif oil_close >= 80:
            s3 = 12
        elif oil_close < 70 and not oil_above_ma20:
            s3 = 3  # 低价但油仍在跌 = 下跌中继
            reasons.append(f"油价${oil_close:.0f}低但在MA20下(下跌中继)")
        elif oil_close < 70:
            s3 = 8
        else:
            s3 = 8
        score += s3

        # === F4: ETF MA200偏离 (10%) ===
        if dev_ma200 < -12:
            s4 = 10
            reasons.append(f"偏离MA200={dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -9:
            s4 = 9
        elif dev_ma200 < -8:
            s4 = 7
        elif dev_ma200 < -5:
            s4 = 5
        elif dev_ma200 < -3:
            s4 = 3
        elif dev_ma200 < 0:
            s4 = 1
        else:
            s4 = 0
        score += s4

        # === F5: 恐慌/连跌/背离 (10%) ===
        rsi_prev = None
        rsi_at_prev_low = None
        is_new_low_20d = False
        try:
            if len(df) >= 21:
                close_20d_before = float(df.iloc[-21:-1]['close'].min())
                current_close = float(df.iloc[-1]['close'])
                is_new_low_20d = current_close <= close_20d_before
                if 'rsi_14' in df.columns:
                    rsi_at_prev_low = float(df.iloc[-21:-1]['rsi_14'].min())
            if len(df) >= 2 and 'rsi_14' in df.columns:
                rsi_prev = float(df.iloc[-2]['rsi_14'])
        except (IndexError, KeyError, TypeError, ValueError):
            pass

        has_divergence = (is_new_low_20d and rsi_at_prev_low is not None
                         and rsi > rsi_at_prev_low + 2)
        rsi_rising = rsi_prev is not None and rsi > rsi_prev
        is_falling_knife = (not has_divergence and rsi_prev is not None
                           and rsi < rsi_prev and 2 <= consec_down <= 3)

        if has_divergence:
            s5 = 10
            reasons.append("底背离(价格新低+RSI未创新低)")
        elif -3 < etf_ret <= -2:
            s5 = 8
            reasons.append(f"单日跌{etf_ret:.1f}%，恐慌抛售可反弹")
        elif consec_down >= 4 and vol_ratio > 1.5:
            s5 = 7
            reasons.append(f"连跌{consec_down}日放量，恐慌底部")
        elif rsi_rising and is_new_low_20d:
            s5 = 8
            reasons.append("微背离(RSI回升+价格新低)")
        elif vol_ratio > 1.5:
            s5 = 5
            reasons.append(f"量比{vol_ratio:.1f}，放量")
        elif is_falling_knife:
            s5 = 0
            reasons.append(f"动能未衰竭(连跌{consec_down}日RSI仍降)")
        elif consec_down >= 3:
            s5 = 0
        elif consec_down >= 2:
            s5 = 3
        else:
            s5 = 0
        score += s5

        # === F6: 组合加成 (5%) ===
        # 关键组合: 油价<$70 + 油在MA20上 + ETF RSI<30
        if oil_close < 70 and oil_above_ma20 and rsi < 30:
            s6 = 5
            reasons.append(f"石油组合(油${oil_close:.0f}+趋势止跌+RSI={rsi:.0f})")
        elif oil_close < 60 and rsi < 25:
            s6 = 3
        elif oil_close >= 90 and rsi < 35:
            s6 = 3
            reasons.append(f"高油价+超卖(油${oil_close:.0f}+RSI={rsi:.0f})")
        else:
            s6 = 0
        score += s6

        # === 下跌中继惩罚: 油在MA20下 + 油RSI<30 + ETF连跌 → 封顶60 ===
        if (not oil_above_ma20 and oil_rsi < 30 and
            consec_down >= 2 and not has_divergence):
            score = min(score, 60)
            reasons.append(f"原油下跌中继封顶(油RSI={oil_rsi:.0f}+在MA20下)")

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 14. 生物科技趋势回踩型算法 (trend_pullback生物科技优化版)
# ===================================================================
class BiotechTrendPullbackAlgorithm(BaseAlgorithm):
    """
    生物科技趋势回踩型 - 适用于标普生物科技ETF(sz159502)

    基于trend_pullback优化, 针对生物科技高波动特性调整:
    - 放宽回踩MA20范围(-5%~+2%, 原版-3%~+1%)
    - RSI < 25给满分(原版<30), 要求更深超卖
    - MACD降权(10%, 原版15%)
    - 布林带替换为KDJ(10%) - 跨境ETF更有效
    - 新增放量回踩因子(20%) - 底部放量是强信号

    回测: 8信号, 87.5%胜率, +2.08%收益, Sharpe 0.71 (原版75.6%/+0.80%/0.25)
    """
    name = "biotech_trend_pullback"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        close = df['close']
        rsi = float(last.get('rsi_14', 50))
        macd_hist = float(last.get('macd_hist', 0))
        ma20 = float(last.get('ma20', 0))
        ma200 = float(last.get('ma200', 0))
        dev_ma20 = float(last.get('dev_ma20', 0))
        momentum_20 = float(last.get('momentum_20', 0))
        kdj_j = float(last.get('kdj_j', 50))
        kdj_k = float(last.get('kdj_k', 50))
        vol_ratio = float(last.get('vol_ratio_20', 1))

        indicators.update({
            'rsi_14': rsi, 'dev_ma20': dev_ma20, 'ma200': ma200,
            'momentum_20': momentum_20, 'kdj_j': kdj_j, 'vol_ratio': vol_ratio,
        })

        # 前提: MA200趋势保护 (放宽到7%, 生物科技波动更大)
        if ma200 > 0 and price < ma200 * 0.93:
            return self._build_result(0, ["跌破MA200 7%,趋势破坏"], indicators)

        # F1: 趋势确认 (15%)
        if ma200 > 0 and price > ma200 and momentum_20 > 0:
            s1 = 15
            reasons.append("上升趋势中")
        elif ma200 > 0 and price > ma200:
            s1 = 8
            reasons.append("价格在MA200上方")
        else:
            s1 = 3
        score += s1

        # F2: 回踩MA20 (25%) - 放宽范围适配高波动
        if ma20 > 0:
            dev_pct = (price - ma20) / ma20 * 100
            if -5 <= dev_pct <= 2:
                s2 = 25
                reasons.append(f"回踩MA20(偏离{dev_pct:+.1f}%)")
            elif -8 <= dev_pct < -5:
                s2 = 18
            elif -12 <= dev_pct < -8:
                s2 = 10
            elif 2 < dev_pct <= 4:
                s2 = 8
            else:
                s2 = 0
        else:
            s2 = 0
        score += s2

        # F3: RSI超卖 (20%) - 要求更深超卖
        if rsi < 25:
            s3 = 20
            reasons.append(f"RSI={rsi:.0f},极度超卖")
        elif rsi < 30:
            s3 = 15
            reasons.append(f"RSI={rsi:.0f},超卖")
        elif rsi < 35:
            s3 = 10
        elif rsi < 40:
            s3 = 5
        else:
            s3 = 0
        score += s3

        # F4: MACD (10%) - 降权
        if len(df) >= 3:
            prev_hist = float(df['macd_hist'].iloc[-2])
            if macd_hist > 0 and prev_hist < 0:
                s4 = 10
                reasons.append("MACD翻红")
            elif macd_hist > prev_hist and macd_hist < 0:
                s4 = 6
                reasons.append("MACD收窄")
            elif macd_hist > 0:
                s4 = 3
            else:
                s4 = 0
        else:
            s4 = 0
        score += s4

        # F5: KDJ超卖 (10%) - 替换布林带
        if kdj_j < 0 and kdj_k < 20:
            s5 = 10
            reasons.append(f"KDJ超卖(J={kdj_j:.0f})")
        elif kdj_j < 10:
            s5 = 6
        elif kdj_j < 20:
            s5 = 3
        else:
            s5 = 0
        score += s5

        # F6: 放量回踩 (20%) - 新增, 底部放量是强信号
        if vol_ratio > 2.0 and dev_ma20 < 0:
            s6 = 20
            reasons.append(f"放量回踩(量比{vol_ratio:.1f})")
        elif vol_ratio > 1.5:
            s6 = 12
        elif vol_ratio > 1.2:
            s6 = 6
        else:
            s6 = 0
        score += s6

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 15. 黄金支撑反弹型算法 (support_rebound黄金优化版)
# ===================================================================
class GoldSupportReboundAlgorithm(BaseAlgorithm):
    """
    黄金支撑反弹型 - 适用于黄金ETF(sh518880)

    基于support_rebound优化, 针对黄金长期趋势特性调整:
    - 新增MA200趋势确认因子(15%) - 黄金在MA200下方时抄底风险大
    - F1支撑位/F2 RSI/F4布林带降权腾空间
    - F3 MACD背离保持(黄金MACD背离信号有效)

    回测: 14信号, 85.7%胜率, +1.92%收益, Sharpe 0.76 (原版82.4%/+1.32%/0.47)
    优化点: 过滤了2026年6月MA200下方下跌中继的3个亏损信号
    """
    name = "gold_support_rebound"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        close = df['close']
        high = df['high']
        low = df['low']
        rsi = float(last.get('rsi_14', 50))
        macd_hist = float(last.get('macd_hist', 0))
        ma200 = float(last.get('ma200', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        zscore = float(last.get('zscore_20', 0))
        kdj_j = float(last.get('kdj_j', 50))
        momentum_20 = float(last.get('momentum_20', 0))

        # 计算60日支撑位
        if len(low) >= 60:
            support = float(low.iloc[-60:].min())
        else:
            support = float(low.min())

        if support > 0 and price > 0:
            dev_support = (price - support) / support * 100
        else:
            dev_support = 100

        indicators.update({
            'rsi_14': rsi, 'macd_hist': macd_hist,
            'support_60d': support, 'dev_support': dev_support,
            'bb_percent_b': bb_pctb, 'zscore_20': zscore,
            'kdj_j': kdj_j, 'ma200': ma200, 'momentum_20': momentum_20,
        })

        # F1: 接近支撑位 (25%, 降权自30%)
        if dev_support <= 2:
            s1 = 25
            reasons.append(f"触及60日支撑(偏离{dev_support:+.1f}%)")
        elif dev_support <= 5:
            s1 = 18
            reasons.append(f"接近支撑(偏离{dev_support:+.1f}%)")
        elif dev_support <= 8:
            s1 = 10
        else:
            s1 = 0
        score += s1

        # F2: RSI超卖 (20%, 降权自25%)
        if rsi < 25:
            s2 = 20
            reasons.append(f"RSI={rsi:.0f},极度超卖")
        elif rsi < 30:
            s2 = 16
            reasons.append(f"RSI={rsi:.0f},超卖")
        elif rsi < 35:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # F3: MACD底背离 (20%, 保持)
        if len(df) >= 10:
            recent_hist = df['macd_hist'].iloc[-5:].values
            if macd_hist < 0 and recent_hist[-1] > recent_hist[0]:
                s3 = 20
                reasons.append("MACD柱线收窄,底背离")
            elif macd_hist > 0 and float(df['macd_hist'].iloc[-2]) < 0:
                s3 = 15
                reasons.append("MACD翻红")
            elif macd_hist < 0 and macd_hist > float(df['macd_hist'].iloc[-3]):
                s3 = 10
            else:
                s3 = 0
        else:
            s3 = 0
        score += s3

        # F4: 布林带 (10%, 降权自15%)
        if bb_pctb < 0.05:
            s4 = 10
            reasons.append("布林带下轨")
        elif bb_pctb < 0.15:
            s4 = 6
        else:
            s4 = 0
        score += s4

        # F5: Z-score + KDJ (10%, 保持)
        if zscore < -1.5 and kdj_j < 10:
            s5 = 10
            reasons.append("Z偏低+KDJ超卖")
        elif zscore < -1:
            s5 = 5
        else:
            s5 = 0
        score += s5

        # F6: MA200趋势确认 (15%, 新增) - 核心优化: 过滤下跌中继
        if ma200 > 0 and price > ma200 and momentum_20 > 0:
            s6 = 15
            reasons.append("上升趋势中回调")
        elif ma200 > 0 and price > ma200:
            s6 = 10
            reasons.append("价格在MA200上方")
        elif ma200 > 0 and price > ma200 * 0.97:
            s6 = 5
        else:
            s6 = 0
        score += s6

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 16. 新能源超卖反弹型 (dividend_value新能源混合优化版)
# ===================================================================
class NewEnergyReversalAlgorithm(BaseAlgorithm):
    """
    新能源超卖反弹型 - 适用于新能源ETF(sh516160)

    基于dividend_value混合优化, 针对新能源板块高波动特性调整:
    - 保留核心因子: RSI超卖(30%) + MA200偏离(25%) + Z-Score(20%)
    - F4布林带降权(15%→10%)
    - F5替换: 量能确认(10%) - 替代底背离, 新能源波动大, 量能信号更直接
    - 新增F6: KDJ超卖(10%) - 另一维度的超卖确认
    - 新增F7: 短期动量(5%) - RSI回升时加分, 确认反弹启动
    - 硬过滤: dev_ma60 < -15% → 封顶59分, 避免极端下跌中继

    回测: 78信号, 78.2%胜率, +2.33%收益, +4.87%最大, Sharpe 0.82
    (原dividend_value: 76信号, 75.0%胜率, +2.19%收益, +4.85%最大, Sharpe 0.75)
    优化点: KDJ+量能+动量替代底背离, 更适合高波动新能源板块
    """
    name = "new_energy_reversal"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        consec_down = int(last.get('consec_down', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        kdj_j = float(last.get('kdj_j', 50))
        dev_ma60 = float(last.get('dev_ma60', 0))

        rsi_prev = None
        try:
            if len(df) >= 2 and 'rsi_14' in df.columns:
                rsi_prev = float(df.iloc[-2]['rsi_14'])
        except (IndexError, KeyError, TypeError, ValueError):
            pass

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma200': dev_ma200, 'consec_down': consec_down,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
            'kdj_j': kdj_j, 'dev_ma60': dev_ma60,
        })

        # F1: RSI超卖 (30%)
        if rsi < 25:
            s1 = 30
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s1 = 25
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s1 = 18
            reasons.append(f"RSI={rsi:.0f}，偏低")
        elif rsi < 45:
            s1 = 10
        else:
            s1 = 0
        score += s1

        # F2: MA200偏离度 (25%)
        if dev_ma200 < -12:
            s2 = 25
            reasons.append(f"40周均线偏离{dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -9:
            s2 = 22
            reasons.append(f"偏离{dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -8:
            s2 = 20
            reasons.append(f"偏离{dev_ma200:.1f}%，明显超卖")
        elif dev_ma200 < -5:
            s2 = 12
        elif dev_ma200 < -3:
            s2 = 7
        elif dev_ma200 < 0:
            s2 = 3
        else:
            s2 = 0
        score += s2

        # F3: Z-Score (20%)
        if zscore < -2:
            s3 = 20
            reasons.append(f"Z-score={zscore:.1f}，严重低于均值")
        elif zscore < -1.8:
            s3 = 18
            reasons.append(f"Z-score={zscore:.1f}，明显低于均值")
        elif zscore < -1.5:
            s3 = 15
        elif zscore < -1:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # F4: 布林带位置 (10%，降权自15%)
        if bb_pctb < 0.05:
            s4 = 10
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s4 = 7
        elif bb_pctb < 0.3:
            s4 = 3
        else:
            s4 = 0
        score += s4

        # F5: 量能确认 (10%，替代原底背离)
        if vol_ratio > 2.0:
            s5 = 10
            reasons.append(f"量比{vol_ratio:.1f}，恐慌性放量")
        elif vol_ratio > 1.5:
            s5 = 7
            reasons.append(f"量比{vol_ratio:.1f}，显著放量")
        elif vol_ratio > 1.2:
            s5 = 4
            reasons.append(f"量比{vol_ratio:.1f}，温和放量")
        else:
            s5 = 0
        score += s5

        # F6: KDJ超卖 (10%，新增)
        if kdj_j < 0:
            s6 = 10
            reasons.append(f"KDJ J={kdj_j:.0f}，极度超卖")
        elif kdj_j < 10:
            s6 = 8
            reasons.append(f"KDJ J={kdj_j:.0f}，严重超卖")
        elif kdj_j < 20:
            s6 = 5
            reasons.append(f"KDJ J={kdj_j:.0f}，超卖")
        elif kdj_j < 30:
            s6 = 2
        else:
            s6 = 0
        score += s6

        # F7: 短期动量确认 (5%，新增)
        if rsi_prev is not None and rsi > rsi_prev and rsi < 45:
            s7 = 5
            reasons.append(f"RSI回升({rsi_prev:.0f}→{rsi:.0f})，动量转正")
        else:
            s7 = 0
        score += s7

        # 硬过滤: MA60极端偏离 → 封顶59，避免下跌中继
        if dev_ma60 < -15:
            score = min(score, 59)
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%，过度超卖封顶")

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 17. 股息率超卖反弹型 (dividend_value股息率优化版)
# ===================================================================
class DividendYieldReversalAlgorithm(BaseAlgorithm):
    """
    股息率超卖反弹型 - 适用于红利ETF(sh510880)

    基于dividend_value优化, 核心改进: 用股息率分位替代底背离因子
    - F1 RSI超卖(28%) + F2 MA200偏离(22%) + F3 Z-Score(20%) + F4 布林带(15%)
    - F5替换: 股息率分位(15%) - 真正的红利价值因子, 原dividend_value缺失
      股息率在过去250日中的百分位 >= 90 → 15分, >= 80 → 12, >= 60 → 8
    - 股息率数据由data_engine.get_dividend_yield()加载, signal_engine注入

    回测: 46信号, 82.6%胜率, +0.67%收益, +1.81%最大, Sharpe 1.952
    (原dividend_value: 47信号, 70.2%胜率, +0.21%收益, +1.41%最大, Sharpe 0.663)
    优化点: 股息率分位替代底背离, 胜率+12.4pp, Sharpe 2.9倍
    """
    name = "dividend_yield_reversal"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        dy_pct = float(last.get('dividend_yield_pct', 50))

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma200': dev_ma200, 'bb_percent_b': bb_pctb,
            'dividend_yield_pct': dy_pct,
        })

        # F1: RSI超卖 (28%)
        if rsi < 25:
            s1 = 28
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s1 = 23
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s1 = 17
            reasons.append(f"RSI={rsi:.0f}，偏低")
        elif rsi < 45:
            s1 = 9
        else:
            s1 = 0
        score += s1

        # F2: MA200偏离度 (22%)
        if dev_ma200 < -12:
            s2 = 22
            reasons.append(f"40周均线偏离{dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -9:
            s2 = 19
            reasons.append(f"偏离{dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -8:
            s2 = 17
            reasons.append(f"偏离{dev_ma200:.1f}%，明显超卖")
        elif dev_ma200 < -5:
            s2 = 10
        elif dev_ma200 < -3:
            s2 = 6
        elif dev_ma200 < 0:
            s2 = 3
        else:
            s2 = 0
        score += s2

        # F3: Z-Score (20%)
        if zscore < -2:
            s3 = 20
            reasons.append(f"Z-score={zscore:.1f}，严重低于均值")
        elif zscore < -1.8:
            s3 = 18
            reasons.append(f"Z-score={zscore:.1f}，明显低于均值")
        elif zscore < -1.5:
            s3 = 15
        elif zscore < -1:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # F4: 布林带位置 (15%)
        if bb_pctb < 0.05:
            s4 = 15
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s4 = 10
        elif bb_pctb < 0.3:
            s4 = 5
        else:
            s4 = 0
        score += s4

        # F5: 股息率分位 (15%, 替代底背离) - 核心新增因子
        if dy_pct >= 90:
            s5 = 15
            reasons.append(f"股息率分位{dy_pct:.0f}，历史高位")
        elif dy_pct >= 80:
            s5 = 12
            reasons.append(f"股息率分位{dy_pct:.0f}，明显偏高")
        elif dy_pct >= 60:
            s5 = 8
            reasons.append(f"股息率分位{dy_pct:.0f}，偏高")
        elif dy_pct >= 40:
            s5 = 4
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 18. 机器人反转型 - 机器人ETF(sh562500)专属
# ===================================================================
class RobotReversalAlgorithm(BaseAlgorithm):
    """
    机器人反转型 - 基于broad_reversal宽RSI优化版

    核心逻辑：
    - RSI双重超卖（放宽阈值，增加信号覆盖）
    - 布林带下轨突破（降权20%）
    - 连续下跌（降权15%，连跌2日也给分）
    - KDJ超卖
    - Z-score + 放量（卖出衰竭）
    - MA60偏离（新增15%因子，偏离越大反弹概率越高）
    """
    name = "robot_reversal"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        rsi7 = float(last.get('rsi_7', 50))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        consec_down = int(last.get('consec_down', 0))
        kdj_j = float(last.get('kdj_j', 50))
        kdj_k = float(last.get('kdj_k', 50))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        zscore = float(last.get('zscore_20', 0))
        dev_ma60 = float(last.get('dev_ma60', 0))

        indicators.update({
            'rsi_14': rsi, 'rsi_7': rsi7, 'bb_percent_b': bb_pctb,
            'consec_down': consec_down, 'kdj_j': kdj_j,
            'vol_ratio': vol_ratio, 'dev_ma60': dev_ma60,
        })

        # F1: RSI双重超卖 (25%, 放宽)
        if rsi < 25 and rsi7 < 20:
            s1 = 25
            reasons.append(f"RSI14={rsi:.0f}/RSI7={rsi7:.0f}，双重极度超卖")
        elif rsi < 30 and rsi7 < 25:
            s1 = 20
            reasons.append(f"RSI={rsi:.0f}/RSI7={rsi7:.0f}，双重超卖")
        elif rsi < 35:
            s1 = 13
            reasons.append(f"RSI={rsi:.0f}，偏低")
        elif rsi < 40:
            s1 = 7
        else:
            s1 = 0
        score += s1

        # F2: 布林带 (20%, 降权)
        if bb_pctb < 0:
            s2 = 20
            reasons.append("跌破布林带下轨")
        elif bb_pctb < 0.05:
            s2 = 16
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # F3: 连续下跌 (15%, 降权, 放宽连跌2日)
        if consec_down >= 5:
            s3 = 15
            reasons.append(f"连跌{consec_down}日")
        elif consec_down >= 4:
            s3 = 12
        elif consec_down >= 3:
            s3 = 8
        elif consec_down >= 2:
            s3 = 5
        else:
            s3 = 0
        score += s3

        # F4: KDJ超卖 (15%)
        if kdj_j < -5 and kdj_k < 20:
            s4 = 15
            reasons.append(f"KDJ J={kdj_j:.0f}，超卖")
        elif kdj_j < 10:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: Z-score + 放量 (10%)
        if zscore < -1.5 and vol_ratio > 1.2:
            s5 = 10
            reasons.append("Z偏低+放量")
        elif zscore < -1:
            s5 = 5
        else:
            s5 = 0
        score += s5

        # F6: MA60偏离 (15%, 新增)
        if dev_ma60 < -10:
            s6 = 15
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%，严重超卖")
        elif dev_ma60 < -7:
            s6 = 11
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%")
        elif dev_ma60 < -5:
            s6 = 7
        elif dev_ma60 < -3:
            s6 = 4
        else:
            s6 = 0
        score += s6

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 19. 创新药反转型 - 创新药ETF(sz159992)专属
# ===================================================================
class PharmaReversalAlgorithm(BaseAlgorithm):
    """
    创新药反转型 - 基于broad_reversal + MA200趋势过滤

    核心逻辑：
    - RSI双重超卖（30%）
    - 布林带下轨突破（25%）
    - 连续下跌（20%）
    - KDJ超卖（15%）
    - Z-score + 放量（10%）
    - MA200趋势过滤：MA200上方加分（趋势回踩更可靠），深熊封顶59
    """
    name = "pharma_reversal"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        rsi7 = float(last.get('rsi_7', 50))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        consec_down = int(last.get('consec_down', 0))
        kdj_j = float(last.get('kdj_j', 50))
        kdj_k = float(last.get('kdj_k', 50))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        zscore = float(last.get('zscore_20', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))

        indicators.update({
            'rsi_14': rsi, 'rsi_7': rsi7, 'bb_percent_b': bb_pctb,
            'consec_down': consec_down, 'kdj_j': kdj_j,
            'vol_ratio': vol_ratio, 'dev_ma200': dev_ma200,
        })

        # F1: RSI双重超卖 (30%)
        if rsi < 25 and rsi7 < 20:
            s1 = 30
            reasons.append(f"RSI14={rsi:.0f}/RSI7={rsi7:.0f}，双重极度超卖")
        elif rsi < 30:
            s1 = 22
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s1 = 15
        elif rsi < 40:
            s1 = 8
        else:
            s1 = 0
        score += s1

        # F2: 布林带 (25%)
        if bb_pctb < 0:
            s2 = 25
            reasons.append("跌破布林带下轨，极端偏离")
        elif bb_pctb < 0.05:
            s2 = 20
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s2 = 12
        else:
            s2 = 0
        score += s2

        # F3: 连续下跌 (20%)
        if consec_down >= 5:
            s3 = 20
            reasons.append(f"连续下跌{consec_down}日，超跌反弹概率大")
        elif consec_down >= 4:
            s3 = 15
        elif consec_down >= 3:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # F4: KDJ超卖 (15%)
        if kdj_j < -5 and kdj_k < 20:
            s4 = 15
            reasons.append(f"KDJ J={kdj_j:.0f}，超卖")
        elif kdj_j < 10:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: Z-score + 放量 (10%)
        if zscore < -1.5 and vol_ratio > 1.2:
            s5 = 10
            reasons.append("Z-score偏低+放量，卖出衰竭")
        elif zscore < -1:
            s5 = 5
        else:
            s5 = 0
        score += s5

        # F6: MA200趋势过滤（加分/封顶）
        if dev_ma200 > 0:
            score = min(score + 5, 100)
            reasons.append(f"MA200上方({dev_ma200:+.1f}%)，趋势回踩")

        # 硬过滤：深熊市封顶59
        if dev_ma200 < -15:
            score = min(score, 59)
            reasons.append(f"MA200下方{dev_ma200:.1f}%，熊市封顶")

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 20. 白酒价值季节型 - 酒ETF(sh512690)专属
# ===================================================================
class WineValueReversalAlgorithm(BaseAlgorithm):
    """
    白酒价值季节型 - 基于financial_value + 白酒季节性因子

    核心逻辑：
    - MA60/MA200严重偏离（28%）
    - RSI超卖（25%）
    - Z-score偏低（15%）
    - 布林带下轨（15%）
    - 白酒季节性因子（17%）：春季调整期/夏季淡季/旺季回调
    """
    name = "wine_value_reversal"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        # 获取当前月份
        if 'date' in df.columns:
            last_date = df['date'].iloc[-1]
            if hasattr(last_date, 'month'):
                month = last_date.month
            else:
                month = int(str(last_date)[5:7])
        else:
            from datetime import datetime as dt
            month = dt.now().month

        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma60 = float(last.get('dev_ma60', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma60': dev_ma60, 'dev_ma200': dev_ma200,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio, 'month': month,
        })

        # F1: MA60/MA200偏离 (28%)
        if dev_ma60 < -10 and dev_ma200 < -15:
            s1 = 28
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%，严重超卖")
        elif dev_ma60 < -8:
            s1 = 20
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%")
        elif dev_ma60 < -5:
            s1 = 11
        else:
            s1 = 0
        score += s1

        # F2: RSI超卖 (25%)
        if rsi < 25:
            s2 = 25
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s2 = 18
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # F3: Z-score (15%)
        if zscore < -2:
            s3 = 15
            reasons.append(f"Z-score={zscore:.1f}，严重偏低")
        elif zscore < -1.5:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # F4: 布林带 (15%)
        if bb_pctb < 0.05:
            s4 = 15
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: 白酒季节性因子 (17%)
        # 白酒季节性规律：
        # 春季调整(3-4月)消费淡季，是布局机会
        # 夏季淡季(7-8月)超卖概率高
        # 旺季回调(12-2月)春节消费旺季前布局
        # 节前预热(10-11月)
        if month in [3, 4]:
            s5 = 17
            reasons.append(f"{month}月春季调整期，季节性布局机会")
        elif month in [7, 8]:
            s5 = 13
            reasons.append(f"{month}月夏季淡季，超卖概率高")
        elif month in [12, 1, 2]:
            s5 = 10
            reasons.append(f"{month}月旺季回调，布局机会")
        elif month in [10, 11]:
            s5 = 7
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ===================================================================
# 算法工厂
# ===================================================================
ALGORITHM_MAP = {
    'dividend_value': DividendValueAlgorithm,
    'broad_reversal': BroadReversalAlgorithm,
    'trend_pullback': TrendPullbackAlgorithm,
    'extreme_reversal': ExtremeReversalAlgorithm,
    'momentum_pullback': MomentumPullbackAlgorithm,
    'support_rebound': SupportReboundAlgorithm,
    'seasonal_value': SeasonalValueAlgorithm,
    'financial_value': FinancialValueAlgorithm,
    'volatility_breakout': VolatilityBreakoutAlgorithm,
    'cycle_momentum': CycleMomentumAlgorithm,
    'premium_rate': PremiumRateAlgorithm,
    'gold_pair_reversal': GoldPairReversalAlgorithm,
    'oil_pair_reversal': OilPairReversalAlgorithm,
    'biotech_trend_pullback': BiotechTrendPullbackAlgorithm,
    'gold_support_rebound': GoldSupportReboundAlgorithm,
    'new_energy_reversal': NewEnergyReversalAlgorithm,
    'dividend_yield_reversal': DividendYieldReversalAlgorithm,
    'robot_reversal': RobotReversalAlgorithm,
    'pharma_reversal': PharmaReversalAlgorithm,
    'wine_value_reversal': WineValueReversalAlgorithm,
}


def get_algorithm(algorithm_name: str) -> BaseAlgorithm:
    """获取算法实例"""
    cls = ALGORITHM_MAP.get(algorithm_name)
    if cls is None:
        raise ValueError(f"未知算法: {algorithm_name}，可选: {list(ALGORITHM_MAP.keys())}")
    return cls()
