# -*- coding: utf-8 -*-
"""
新能源ETF - dividend_value 信号深度分析 + 优化变体对比
======================================================
当前算法: volatility_breakout (72.1%胜率, -0.83%收益, 43信号)
基准算法: dividend_value (75.0%胜率, +2.19%收益, 76信号)

设计3个优化变体:
  A. 趋势过滤版 - 加MA60极端偏离过滤+短期反弹确认
  B. 宽阈值版 - 放宽RSI/MA200阈值+加KDJ因子,增加信号
  C. 混合版 - 保留核心因子+替换F5为量能+KDJ+动量确认
"""
import sys, os, json, logging
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import ALGORITHM_MAP, BaseAlgorithm, DividendValueAlgorithm
from indicators import calc_all_indicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'algo_optimization')
ETF_CODE = 'sh516160'
ETF_NAME = '新能源ETF'
START_DATE = '2024-01-01'
THRESHOLD = 60


# ======================================================================
# 变体A: 趋势过滤版 - MA60极端偏离过滤+短期反弹确认
# ======================================================================
class DividendValueEnhanced(BaseAlgorithm):
    """
    dividend_value 新能源增强版
    - 保留原5因子不变
    - 新增F6: MA60极端偏离过滤(15%) - 新能源板块波动大,
      偏离MA60超-15%时是下跌中继非底部,硬封顶59分不发信号
    - 新增F7: 短期反弹确认(5%) - RSI从超卖区回升时加分
    """
    name = "dividend_value_enhanced"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        consec_down = int(last.get('consec_down', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        dev_ma60 = float(last.get('dev_ma60', 0))
        ma60 = float(last.get('ma60', 0))

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma200': dev_ma200, 'consec_down': consec_down,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
            'dev_ma60': dev_ma60,
        })

        # F1: RSI超卖 (30%)
        if rsi < 25:
            s1 = 30
            reasons.append(f"RSI={rsi:.0f}, 极度超卖")
        elif rsi < 30:
            s1 = 25
            reasons.append(f"RSI={rsi:.0f}, 超卖")
        elif rsi < 35:
            s1 = 18
            reasons.append(f"RSI={rsi:.0f}, 偏低")
        elif rsi < 45:
            s1 = 10
        else:
            s1 = 0
        score += s1

        # F2: MA200偏离度 (25%)
        if dev_ma200 < -12:
            s2 = 25
            reasons.append(f"40周均线偏离{dev_ma200:.1f}%, 严重超卖")
        elif dev_ma200 < -9:
            s2 = 22
            reasons.append(f"偏离{dev_ma200:.1f}%, 严重超卖")
        elif dev_ma200 < -8:
            s2 = 20
            reasons.append(f"偏离{dev_ma200:.1f}%, 明显超卖")
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
            reasons.append(f"Z-score={zscore:.1f}, 严重低于均值")
        elif zscore < -1.8:
            s3 = 18
            reasons.append(f"Z-score={zscore:.1f}, 明显低于均值")
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

        # F5: 底背离/微背离/动能衰减 (10%) - 与原版相同
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
        is_falling_knife = (not has_divergence and rsi_prev is not None
                           and rsi < rsi_prev and 2 <= consec_down <= 3)

        if has_divergence:
            s5 = 10
            reasons.append(f"底背离: 价格新低但RSI未创新低")
        elif rsi_rising and is_new_low_20d:
            s5 = 8
            reasons.append(f"微背离: RSI回升但价格新低")
        elif consec_down >= 4 and vol_ratio > 1.5:
            s5 = 10
            reasons.append(f"连跌{consec_down}日且放量, 恐慌性抛售")
        elif is_falling_knife and is_new_low_20d:
            s5 = 0
        elif is_new_low_20d and consec_down >= 3:
            s5 = 7
            reasons.append(f"20日新低且连跌{consec_down}日")
        elif is_new_low_20d:
            s5 = 5
            reasons.append("创20日新低")
        elif is_falling_knife:
            s5 = 0
        elif consec_down >= 3:
            s5 = 5
            reasons.append(f"连跌{consec_down}日")
        elif vol_ratio > 1.2:
            s5 = 3
            reasons.append(f"量比{vol_ratio:.1f}, 放量信号")
        else:
            s5 = 0
        score += s5

        # F6: MA60极端偏离过滤 (新增)
        if dev_ma60 < -15:
            # 偏离MA60超15%, 下跌中继风险极大, 硬封顶59
            score = min(score, 59)
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%, 过度超卖封顶(下跌中继风险)")
        elif dev_ma60 < -10:
            # 偏离10-15%, 降权(扣5分)
            score = max(score - 5, 0)
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%, 严重超卖降权")

        # F7: 短期反弹确认 (新增, 5分bonus)
        if rsi_prev is not None and rsi > rsi_prev and rsi < 40:
            score += 5
            reasons.append(f"RSI回升({rsi_prev:.0f}→{rsi:.0f}), 短期反弹确认")

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 变体B: 宽阈值版 - 放宽RSI/MA200阈值+加KDJ因子
# ======================================================================
class DividendValueLoose(BaseAlgorithm):
    """
    dividend_value 新能源宽阈值版
    - F1 RSI放宽: <50给5分(原版<45给10分,此处更宽但更低分)
    - F2 MA200放宽: <-1%给5分(原版<-3%给7分)
    - 新增F6: KDJ超卖(15%) - J<10→15, J<20→10, J<30→5
    - F3-F4保持, F5保持
    - 目标: 增加信号数量, 捕捉更多中等超卖机会
    """
    name = "dividend_value_loose"

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

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma200': dev_ma200, 'consec_down': consec_down,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
            'kdj_j': kdj_j,
        })

        # F1: RSI超卖 (25%, 降权腾KDJ空间)
        if rsi < 25:
            s1 = 25
            reasons.append(f"RSI={rsi:.0f}, 极度超卖")
        elif rsi < 30:
            s1 = 20
            reasons.append(f"RSI={rsi:.0f}, 超卖")
        elif rsi < 35:
            s1 = 15
            reasons.append(f"RSI={rsi:.0f}, 偏低")
        elif rsi < 45:
            s1 = 8
        elif rsi < 50:
            s1 = 5  # 新增: 轻度超卖也给分
            reasons.append(f"RSI={rsi:.0f}, 轻度偏低")
        else:
            s1 = 0
        score += s1

        # F2: MA200偏离度 (20%, 降权)
        if dev_ma200 < -12:
            s2 = 20
            reasons.append(f"40周均线偏离{dev_ma200:.1f}%, 严重超卖")
        elif dev_ma200 < -9:
            s2 = 18
            reasons.append(f"偏离{dev_ma200:.1f}%, 严重超卖")
        elif dev_ma200 < -8:
            s2 = 15
            reasons.append(f"偏离{dev_ma200:.1f}%, 明显超卖")
        elif dev_ma200 < -5:
            s2 = 10
        elif dev_ma200 < -3:
            s2 = 5
        elif dev_ma200 < -1:
            s2 = 5  # 新增: 轻度偏离也给分
            reasons.append(f"偏离{dev_ma200:.1f}%, 轻度超卖")
        else:
            s2 = 0
        score += s2

        # F3: Z-Score (20%)
        if zscore < -2:
            s3 = 20
            reasons.append(f"Z-score={zscore:.1f}, 严重低于均值")
        elif zscore < -1.8:
            s3 = 18
            reasons.append(f"Z-score={zscore:.1f}, 明显低于均值")
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

        # F5: 底背离/微背离/动能衰减 (5%, 降权)
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
        is_falling_knife = (not has_divergence and rsi_prev is not None
                           and rsi < rsi_prev and 2 <= consec_down <= 3)

        if has_divergence:
            s5 = 5
            reasons.append("底背离")
        elif rsi_rising and is_new_low_20d:
            s5 = 4
            reasons.append("微背离")
        elif consec_down >= 4 and vol_ratio > 1.5:
            s5 = 5
            reasons.append(f"连跌{consec_down}日且放量, 恐慌抛售")
        elif is_falling_knife and is_new_low_20d:
            s5 = 0
        elif is_new_low_20d and consec_down >= 3:
            s5 = 3
        elif is_new_low_20d:
            s5 = 2
        elif is_falling_knife:
            s5 = 0
        elif consec_down >= 3:
            s5 = 2
        elif vol_ratio > 1.2:
            s5 = 1
        else:
            s5 = 0
        score += s5

        # F6: KDJ超卖 (15%, 新增)
        if kdj_j < 0:
            s6 = 15
            reasons.append(f"KDJ J={kdj_j:.0f}, 极度超卖")
        elif kdj_j < 10:
            s6 = 12
            reasons.append(f"KDJ J={kdj_j:.0f}, 严重超卖")
        elif kdj_j < 20:
            s6 = 8
            reasons.append(f"KDJ J={kdj_j:.0f}, 超卖")
        elif kdj_j < 30:
            s6 = 4
        else:
            s6 = 0
        score += s6

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 变体C: 混合版 - 核心因子+量能/KDJ/动量确认替换F5
# ======================================================================
class DividendValueMixed(BaseAlgorithm):
    """
    dividend_value 新能源混合版
    - 保留F1(RSI 30%)+F2(MA200 25%)+F3(Z-score 20%)+F4(BB 10%, 降权)
    - F5替换为: 量能确认(10%)+KDJ超卖(10%)+短期动量(5%)
    - 新增硬过滤: dev_ma60 < -15% → 封顶59(同增强版)
    - 目标: 用量能+KDJ+动量替代底背离, 更适合波动大的新能源
    """
    name = "dividend_value_mixed"

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
            reasons.append(f"RSI={rsi:.0f}, 极度超卖")
        elif rsi < 30:
            s1 = 25
            reasons.append(f"RSI={rsi:.0f}, 超卖")
        elif rsi < 35:
            s1 = 18
            reasons.append(f"RSI={rsi:.0f}, 偏低")
        elif rsi < 45:
            s1 = 10
        else:
            s1 = 0
        score += s1

        # F2: MA200偏离度 (25%)
        if dev_ma200 < -12:
            s2 = 25
            reasons.append(f"40周均线偏离{dev_ma200:.1f}%, 严重超卖")
        elif dev_ma200 < -9:
            s2 = 22
            reasons.append(f"偏离{dev_ma200:.1f}%, 严重超卖")
        elif dev_ma200 < -8:
            s2 = 20
            reasons.append(f"偏离{dev_ma200:.1f}%, 明显超卖")
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
            reasons.append(f"Z-score={zscore:.1f}, 严重低于均值")
        elif zscore < -1.8:
            s3 = 18
            reasons.append(f"Z-score={zscore:.1f}, 明显低于均值")
        elif zscore < -1.5:
            s3 = 15
        elif zscore < -1:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # F4: 布林带位置 (10%, 降权)
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

        # F5: 量能确认 (10%, 替换原F5)
        if vol_ratio > 2.0:
            s5 = 10
            reasons.append(f"量比{vol_ratio:.1f}, 恐慌性放量")
        elif vol_ratio > 1.5:
            s5 = 7
            reasons.append(f"量比{vol_ratio:.1f}, 显著放量")
        elif vol_ratio > 1.2:
            s5 = 4
            reasons.append(f"量比{vol_ratio:.1f}, 温和放量")
        else:
            s5 = 0
        score += s5

        # F6: KDJ超卖 (10%, 新增)
        if kdj_j < 0:
            s6 = 10
            reasons.append(f"KDJ J={kdj_j:.0f}, 极度超卖")
        elif kdj_j < 10:
            s6 = 8
            reasons.append(f"KDJ J={kdj_j:.0f}, 严重超卖")
        elif kdj_j < 20:
            s6 = 5
            reasons.append(f"KDJ J={kdj_j:.0f}, 超卖")
        elif kdj_j < 30:
            s6 = 2
        else:
            s6 = 0
        score += s6

        # F7: 短期动量确认 (5%, 新增)
        if rsi_prev is not None and rsi > rsi_prev and rsi < 45:
            s7 = 5
            reasons.append(f"RSI回升({rsi_prev:.0f}>{rsi:.0f}), 动量转正")
        elif rsi_prev is not None and rsi < rsi_prev and consec_down >= 2:
            s7 = 0
            # RSI仍在降+连跌, 不加分(但不惩罚, 让其他因子决定)
        else:
            s7 = 0
        score += s7

        # 硬过滤: MA60极端偏离
        if dev_ma60 < -15:
            score = min(score, 59)
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%, 过度超卖封顶")

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 回测函数
# ======================================================================
def run_backtest(df, algo, threshold=THRESHOLD):
    signals = []
    for i in range(60, len(df) - 3):
        date_str = df.iloc[i]['date'].strftime('%Y-%m-%d')
        if date_str < START_DATE:
            continue
        df_slice = df.iloc[:i + 1]
        try:
            signal = algo.calculate(df_slice)
        except Exception:
            continue
        if signal.score < threshold:
            continue

        buy_price = float(df.iloc[i]['close'])
        future = df.iloc[i + 1: i + 1 + 3]
        if len(future) < 1:
            continue

        future_high = float(future['high'].max())
        future_close = float(future.iloc[-1]['close'])
        max_ret = (future_high / buy_price - 1) * 100
        close_ret = (future_close / buy_price - 1) * 100
        is_win = max_ret > 0.5

        days_to_win = 0
        if is_win:
            for j in range(len(future)):
                if (float(future.iloc[j]['high']) / buy_price - 1) * 100 > 0.5:
                    days_to_win = j + 1
                    break

        signals.append({
            'date': date_str, 'score': round(signal.score, 1),
            'level': signal.level, 'buy_price': round(buy_price, 4),
            'max_return': round(max_ret, 2), 'close_return': round(close_ret, 2),
            'is_win': is_win, 'days_to_win': days_to_win,
            'reasons': signal.reasons[:3],
        })

    n = len(signals)
    if n == 0:
        return {'signals': 0, 'wins': 0, 'win_rate': 0, 'avg_return': 0,
                'avg_max_return': 0, 'avg_score': 0, 'best': 0, 'worst': 0,
                'sharpe': 0, 'std': 0, 'signal_list': []}

    wins = sum(1 for s in signals if s['is_win'])
    rets = [s['close_return'] for s in signals]
    max_rets = [s['max_return'] for s in signals]
    scores = [s['score'] for s in signals]
    std = np.std(rets) if len(rets) > 1 else 0
    sharpe = (np.mean(rets) / std * np.sqrt(len(rets))) if std > 0 else 0

    return {
        'signals': n, 'wins': wins, 'win_rate': round(wins / n * 100, 1),
        'avg_return': round(np.mean(rets), 2),
        'avg_max_return': round(np.mean(max_rets), 2),
        'avg_score': round(np.mean(scores), 1),
        'best': round(max(max_rets), 2), 'worst': round(min(rets), 2),
        'sharpe': round(sharpe, 3), 'std': round(std, 2),
        'signal_list': signals,
    }


def generate_html(results, df_info):
    sorted_results = sorted(results.items(), key=lambda x: (-x[1]['win_rate'], -x[1]['signals']))

    # 摘要表
    summary_rows = ""
    for name, r in sorted_results:
        wr = r['win_rate']
        ret = r['avg_return']
        maxret = r['avg_max_return']
        wr_c = '#27ae60' if wr >= 75 else ('#e67e22' if wr >= 60 else '#e74c3c')
        ret_c = '#27ae60' if ret >= 0 else '#e74c3c'
        max_c = '#27ae60' if maxret >= 0 else '#e74c3c'
        marker = ''
        if name == 'dividend_value (原版)':
            marker = ' [基准]'
        best_mark = ''
        candidates = [(n, v) for n, v in results.items() if v['signals'] >= 5]
        if candidates:
            best_name = max(candidates, key=lambda x: (x[1]['win_rate'], x[1]['avg_max_return']))[0]
            if name == best_name and name != 'dividend_value (原版)':
                best_mark = ' [最优]'
        summary_rows += f"""
            <tr>
                <td><strong>{name}</strong>{marker}{best_mark}</td>
                <td>{r['signals']}</td>
                <td>{r['wins']}</td>
                <td style="color:{wr_c};font-weight:bold;">{wr:.1f}%</td>
                <td style="color:{ret_c};">{ret:+.2f}%</td>
                <td style="color:{max_c};font-weight:bold;">{maxret:+.2f}%</td>
                <td>{r['avg_score']:.1f}</td>
                <td>{r['sharpe']:.3f}</td>
                <td style="color:#27ae60;">{r['best']:+.2f}%</td>
                <td style="color:#e74c3c;">{r['worst']:+.2f}%</td>
            </tr>"""

    # 逐年对比
    yearly_data = {}
    for name, r in results.items():
        for s in r['signal_list']:
            year = s['date'][:4]
            key = f"{name}_{year}"
            if key not in yearly_data:
                yearly_data[key] = {'algo': name, 'year': year, 'signals': 0, 'wins': 0, 'rets': []}
            yearly_data[key]['signals'] += 1
            if s['is_win']:
                yearly_data[key]['wins'] += 1
            yearly_data[key]['rets'].append(s['close_return'])

    yearly_rows = ""
    years = sorted(set(s['date'][:4] for r in results.values() for s in r['signal_list']))
    algos = [name for name, _ in sorted_results]
    for algo_name in algos:
        for year in years:
            key = f"{algo_name}_{year}"
            d = yearly_data.get(key)
            if d and d['signals'] > 0:
                wr = d['wins'] / d['signals'] * 100
                avg_ret = np.mean(d['rets'])
                wr_c = '#27ae60' if wr >= 75 else ('#e67e22' if wr >= 60 else '#e74c3c')
                yearly_rows += f"""
                <tr>
                    <td>{algo_name}</td><td>{year}</td><td>{d['signals']}</td>
                    <td style="color:{wr_c};">{wr:.0f}%</td>
                    <td>{avg_ret:+.2f}%</td>
                </tr>"""

    # 信号明细(原版 vs 最优变体)
    candidates = [(n, v) for n, v in results.items() if v['signals'] >= 5 and n != 'dividend_value (原版)']
    best_name = max(candidates, key=lambda x: (x[1]['win_rate'], x[1]['avg_max_return']))[0] if candidates else 'dividend_value (原版)'

    detail_rows = ""
    for algo_name in ['dividend_value (原版)', best_name]:
        r = results.get(algo_name, {})
        for s in r.get('signal_list', []):
            win_class = 'win' if s['is_win'] else 'loss'
            ret_c = '#27ae60' if s['close_return'] >= 0 else '#e74c3c'
            max_c = '#27ae60' if s['max_return'] >= 0 else '#e74c3c'
            reasons = '; '.join(s.get('reasons', [])) if s.get('reasons') else '-'
            detail_rows += f"""
            <tr class="{win_class}">
                <td>{algo_name}</td><td>{s['date']}</td><td>{s['score']:.1f}</td>
                <td>{s['buy_price']:.3f}</td>
                <td class="win-cell">{'Y' if s['is_win'] else 'N'}</td>
                <td style="color:{max_c};font-weight:bold;">{s['max_return']:+.2f}%</td>
                <td style="color:{ret_c};">{s['close_return']:+.2f}%</td>
                <td>{s['days_to_win']}d</td>
                <td class="reasons-cell">{reasons}</td>
            </tr>"""

    # 结论
    orig = results.get('dividend_value (原版)', {})
    if candidates:
        best_r = results[best_name]
        if best_r['win_rate'] > orig.get('win_rate', 0) + 2:
            conclusion = f"最优变体: {best_name} (胜率{best_r['win_rate']:.1f}% vs 原版{orig.get('win_rate',0):.1f}%), 建议采用"
            conclusion_color = '#27ae60'
        elif best_r['win_rate'] >= orig.get('win_rate', 0):
            conclusion = f"变体{best_name}胜率{best_r['win_rate']:.1f}%接近原版{orig.get('win_rate',0):.1f}%, 收益{best_r['avg_return']:+.2f}% vs 原版{orig.get('avg_return',0):+.2f}%, 综合评估"
            conclusion_color = '#e67e22'
        else:
            conclusion = f"原版dividend_value最优(胜率{orig.get('win_rate',0):.1f}%), 所有变体均未超越, 建议采用原版"
            conclusion_color = '#3498db'
    else:
        conclusion = "无有效变体对比, 建议采用原版dividend_value"
        conclusion_color = '#3498db'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>新能源ETF dividend_value 优化对比</title>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif; background:#f5f6fa; color:#2c3e50; padding:20px; line-height:1.6; }}
        .header {{ background:linear-gradient(135deg,#1a1a2e,#16213e); color:white; padding:25px; border-radius:12px; margin-bottom:20px; }}
        .header h1 {{ font-size:22px; margin-bottom:8px; }}
        .header .meta {{ opacity:0.8; font-size:13px; }}
        .conclusion {{ background:{conclusion_color}; color:white; padding:15px 20px; border-radius:10px; margin-bottom:20px; font-size:15px; }}
        .section {{ background:white; border-radius:12px; padding:20px; margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,0.08); overflow-x:auto; }}
        .section h2 {{ font-size:17px; margin-bottom:15px; color:#2c3e50; border-bottom:2px solid #ecf0f1; padding-bottom:10px; }}
        table {{ width:100%; border-collapse:collapse; font-size:13px; }}
        th {{ background:#f8f9fa; padding:10px; text-align:left; border-bottom:2px solid #dee2e6; color:#495057; white-space:nowrap; }}
        td {{ padding:8px 10px; border-bottom:1px solid #ecf0f1; }}
        tr.win {{ background:#f0fff4; }}
        tr.loss {{ background:#fff5f5; }}
        .win-cell {{ font-weight:bold; text-align:center; }}
        tr.win .win-cell {{ color:#27ae60; }}
        tr.loss .win-cell {{ color:#e74c3c; }}
        .reasons-cell {{ font-size:11px; color:#555; max-width:300px; }}
        .footer {{ text-align:center; padding:20px; color:#95a5a6; font-size:12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>新能源ETF dividend_value 优化对比</h1>
        <div class="meta">
            ETF: {ETF_CODE} {ETF_NAME} | 数据: {df_info} |
            回测: {START_DATE}~最新 | 信号阈值: &ge;{THRESHOLD}分 |
            胜率定义: T+3最高价收益 &gt; 0.5%
        </div>
    </div>
    <div class="conclusion">{conclusion}</div>
    <div class="section">
        <h2>算法对比摘要 (按胜率降序)</h2>
        <table>
            <tr><th>算法</th><th>信号</th><th>胜利</th><th>胜率</th>
            <th>均收益</th><th>均最大</th><th>均分</th><th>Sharpe</th>
            <th>最佳</th><th>最差</th></tr>
            {summary_rows}
        </table>
    </div>
    <div class="section">
        <h2>逐年对比</h2>
        <table>
            <tr><th>算法</th><th>年份</th><th>信号</th><th>胜率</th><th>均收益</th></tr>
            {yearly_rows}
        </table>
    </div>
    <div class="section">
        <h2>原版 vs 最优变体 - 信号明细</h2>
        <table>
            <tr><th>算法</th><th>日期</th><th>信号分</th><th>买入价</th>
            <th>胜</th><th>最大收益</th><th>收盘收益</th><th>达标天数</th><th>理由</th></tr>
            {detail_rows}
        </table>
    </div>
    <div class="footer">
        <p>新能源ETF算法优化 | dividend_value 3变体对比 | T+3胜率回测</p>
        <p>本报告仅供参考, 不构成投资建议。</p>
    </div>
</body>
</html>"""
    return html


def main():
    logger.info(f"开始 {ETF_NAME} dividend_value 优化对比...")

    engine = DataEngine()
    df = engine.get_history_kline(ETF_CODE)
    if df is None:
        logger.error(f"无法获取 {ETF_CODE} 数据")
        return

    df_info = f"{len(df)}条, {df['date'].iloc[0].strftime('%Y-%m-%d')}~{df['date'].iloc[-1].strftime('%Y-%m-%d')}"
    logger.info(f"{ETF_NAME} 数据: {df_info}")
    df = calc_all_indicators(df)

    # 运行回测
    algorithms = {
        'dividend_value (原版)': DividendValueAlgorithm(),
        'A. 趋势过滤版': DividendValueEnhanced(),
        'B. 宽阈值版': DividendValueLoose(),
        'C. 混合版': DividendValueMixed(),
    }

    results = {}
    for name, algo in algorithms.items():
        logger.info(f"  回测 {name}...")
        r = run_backtest(df, algo)
        results[name] = r
        if r['signals'] > 0:
            logger.info(f"  {name:25s} | 信号:{r['signals']:3d} | 胜率:{r['win_rate']:.1f}% | 收益:{r['avg_return']:+.2f}% | 最大:{r['avg_max_return']:+.2f}% | Sharpe:{r['sharpe']:.3f}")
        else:
            logger.info(f"  {name:25s} | 信号:  0")

    # 保存JSON
    json_data = {k: {kk: vv for kk, vv in v.items() if kk != 'signal_list'} for k, v in results.items()}
    json_path = os.path.join(OUTPUT_DIR, '新能源ETF_dividend_value优化.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON: {json_path}")

    # 生成HTML
    html = generate_html(results, df_info)
    html_path = os.path.join(OUTPUT_DIR, '新能源ETF_dividend_value优化.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML: {html_path}")

    # 打印摘要
    print(f"\n{'=' * 90}")
    print(f"新能源ETF dividend_value 优化对比 (按胜率降序)")
    print(f"{'=' * 90}")
    print(f"{'算法':<28s} {'信号':>4s} {'胜率':>6s} {'均收益':>7s} {'均最大':>7s} {'Sharpe':>7s}")
    print("-" * 90)
    for name, r in sorted(results.items(), key=lambda x: (-x[1]['win_rate'], -x[1]['signals'])):
        marker = ''
        if '原版' in name:
            marker = ' <- 基准'
        print(f"{name:<28s} {r['signals']:4d} {r['win_rate']:5.1f}% {r['avg_return']:+6.2f}% {r['avg_max_return']:+6.2f}% {r['sharpe']:6.3f}{marker}")


if __name__ == '__main__':
    main()
