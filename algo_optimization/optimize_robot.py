# -*- coding: utf-8 -*-
"""
机器人ETF - broad_reversal/extreme_reversal 优化对比
====================================================
当前算法: volatility_breakout (83.7%胜率, +0.41%收益, 43信号)
基准算法: broad_reversal (85.3%胜率, +0.72%收益, 34信号)
         extreme_reversal (100%胜率, +3.36%收益, 11信号)

设计3个优化变体:
  A. 宽RSI+MA60版 - broad_reversal放宽RSI+新增MA60偏离(15%)
  B. 极端放宽+KDJ版 - extreme_reversal放宽RSI/Z阈值+加KDJ
  C. 混合趋势版 - broad_reversal核心+extreme的Z-score+MA60过滤
"""
import sys, os, json, logging
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import (ALGORITHM_MAP, BaseAlgorithm,
                        BroadReversalAlgorithm, ExtremeReversalAlgorithm,
                        VolatilityBreakoutAlgorithm)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'algo_optimization')
ETF_CODE = 'sh562500'
ETF_NAME = '机器人ETF'
START_DATE = '2024-01-01'
THRESHOLD = 60


# ======================================================================
# 变体A: 宽RSI+MA60版 - broad_reversal放宽RSI+新增MA60偏离
# ======================================================================
class BroadReversalWideRSI(BaseAlgorithm):
    """
    broad_reversal 机器人宽RSI版
    - F1 RSI放宽(25%): RSI<30且RSI7<25→25(原<25且<20), 增加信号
    - F2 布林带(20%, 降权)
    - F3 连跌(15%, 降权): 连跌2日也给5分
    - F4 KDJ(15%): 保持
    - F5 Z+放量(10%): 保持
    - F6 MA60偏离(15%, 新增): 偏离越大反弹概率越高
    """
    name = "broad_reversal_wide_rsi"

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

        # F4: KDJ超卖 (15%, 保持)
        if kdj_j < -5 and kdj_k < 20:
            s4 = 15
            reasons.append(f"KDJ J={kdj_j:.0f}，超卖")
        elif kdj_j < 10:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: Z-score+放量 (10%, 保持)
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


# ======================================================================
# 变体B: 极端放宽+KDJ版 - extreme_reversal放宽阈值+加KDJ
# ======================================================================
class ExtremeReversalRelaxed(BaseAlgorithm):
    """
    extreme_reversal 机器人放宽版
    - F1 RSI(25%): 放宽 RSI<20→25(原<15→30), <25→20, <30→15, <35→8
    - F2 Z-Score(20%): 放宽 Z<-2→20(原<-2.5→25), <-1.5→12
    - F3 布林带(15%): 保持
    - F4 KDJ(15%, 新增): J<0→15, J<10→10, J<20→5
    - F5 恐慌放量(10%): 降权
    - F6 连跌+MA60(15%): 保持
    目标: 在extreme_reversal 100%胜率基础上增加信号数
    """
    name = "extreme_reversal_relaxed"

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
        kdj_k = float(last.get('kdj_k', 50))

        indicators.update({
            'rsi_14': rsi, 'rsi_7': rsi7, 'zscore_20': zscore,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
            'consec_down': consec_down, 'dev_ma60': dev_ma60,
            'kdj_j': kdj_j,
        })

        # F1: 极端RSI (25%, 放宽)
        if rsi < 15:
            s1 = 25
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 20:
            s1 = 22
            reasons.append(f"RSI={rsi:.0f}，极端超卖")
        elif rsi < 25:
            s1 = 16
            reasons.append(f"RSI={rsi:.0f}，严重超卖")
        elif rsi < 30:
            s1 = 8
        else:
            s1 = 0
        score += s1

        # F2: Z-Score (20%, 放宽)
        if zscore < -2.5:
            s2 = 20
            reasons.append(f"Z-score={zscore:.1f}，极端偏离")
        elif zscore < -2:
            s2 = 16
            reasons.append(f"Z-score={zscore:.1f}，严重偏离")
        elif zscore < -1.5:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # F3: 布林带 (15%, 保持)
        if bb_pctb < 0:
            s3 = 15
            reasons.append("跌破布林带下轨")
        elif bb_pctb < 0.05:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # F4: KDJ超卖 (15%, 新增)
        if kdj_j < 0 and kdj_k < 20:
            s4 = 15
            reasons.append(f"KDJ J={kdj_j:.0f}，极度超卖")
        elif kdj_j < 10:
            s4 = 10
            reasons.append(f"KDJ J={kdj_j:.0f}，超卖")
        elif kdj_j < 20:
            s4 = 5
        else:
            s4 = 0
        score += s4

        # F5: 恐慌放量 (10%, 降权)
        if vol_ratio > 2.0:
            s5 = 10
            reasons.append(f"量比{vol_ratio:.1f}，恐慌放量")
        elif vol_ratio > 1.5:
            s5 = 7
        elif vol_ratio > 1.2:
            s5 = 4
        else:
            s5 = 0
        score += s5

        # F6: 连跌+MA60偏离 (15%, 保持)
        if consec_down >= 4 and dev_ma60 < -10:
            s6 = 15
            reasons.append(f"连跌{consec_down}日+偏离MA60 {dev_ma60:.1f}%")
        elif consec_down >= 3 and dev_ma60 < -5:
            s6 = 10
        elif dev_ma60 < -10:
            s6 = 8
        else:
            s6 = 0
        score += s6

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 变体C: 混合趋势版 - broad_reversal核心+extreme的Z-score+MA60过滤
# ======================================================================
class BroadExtremeMixed(BaseAlgorithm):
    """
    broad_reversal + extreme_reversal 混合版
    - F1: RSI双重超卖 (25%, broad_reversal核心)
    - F2: 布林带 (20%, broad_reversal)
    - F3: Z-Score极端 (20%, from extreme_reversal, 加权)
      Z<-3→20, Z<-2→16, Z<-1.5→12, Z<-1→6 (替代原连跌因子)
    - F4: KDJ超卖 (15%, broad_reversal)
    - F5: 量能+连跌 (10%, 混合)
    - F6: MA60趋势过滤 (10%, 新增)
      偏离MA60<-8%→10(超卖), <-5%→7, <-3%→4, >5%→0(封顶59)
    硬过滤: dev_ma60 > 5% → 封顶59 (超买不抄底)
    """
    name = "broad_extreme_mixed"

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
            'zscore_20': zscore,
        })

        # F1: RSI双重超卖 (25%)
        if rsi < 25 and rsi7 < 20:
            s1 = 25
            reasons.append(f"RSI14={rsi:.0f}/RSI7={rsi7:.0f}，双重极度超卖")
        elif rsi < 30:
            s1 = 20
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s1 = 12
        elif rsi < 40:
            s1 = 6
        else:
            s1 = 0
        score += s1

        # F2: 布林带 (20%)
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

        # F3: Z-Score极端 (20%, from extreme_reversal加权)
        if zscore < -3:
            s3 = 20
            reasons.append(f"Z-score={zscore:.1f}，极端偏离(3σ)")
        elif zscore < -2:
            s3 = 16
            reasons.append(f"Z-score={zscore:.1f}，严重偏离")
        elif zscore < -1.5:
            s3 = 12
        elif zscore < -1:
            s3 = 6
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

        # F5: 量能+连跌 (10%, 混合)
        if vol_ratio > 1.5 and consec_down >= 3:
            s5 = 10
            reasons.append(f"连跌{consec_down}日+放量{vol_ratio:.1f}")
        elif vol_ratio > 1.2 and consec_down >= 2:
            s5 = 6
        elif consec_down >= 4:
            s5 = 5
        else:
            s5 = 0
        score += s5

        # F6: MA60趋势 (10%, 新增)
        if dev_ma60 < -8:
            s6 = 10
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%，严重超卖")
        elif dev_ma60 < -5:
            s6 = 7
        elif dev_ma60 < -3:
            s6 = 4
        else:
            s6 = 0
        score += s6

        # 硬过滤: MA60超买(偏离>5%) → 封顶59
        if dev_ma60 > 5:
            score = min(score, 59)
            reasons.append(f"偏离MA60 +{dev_ma60:.1f}%，超买封顶")

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


def main():
    logger.info(f"开始 {ETF_NAME} 优化对比...")

    engine = DataEngine()
    df = engine.get_history_kline(ETF_CODE)
    if df is None:
        logger.error(f"无法获取 {ETF_CODE} 数据")
        return
    df_info = f"{len(df)}条, {df['date'].iloc[0].strftime('%Y-%m-%d')}~{df['date'].iloc[-1].strftime('%Y-%m-%d')}"
    logger.info(f"{ETF_NAME} 数据: {df_info}")
    df = calc_all_indicators(df)

    algorithms = {
        'volatility_breakout (当前)': VolatilityBreakoutAlgorithm(),
        'broad_reversal (基准1)': BroadReversalAlgorithm(),
        'extreme_reversal (基准2)': ExtremeReversalAlgorithm(),
        'A. 宽RSI+MA60版': BroadReversalWideRSI(),
        'B. 极端放宽+KDJ版': ExtremeReversalRelaxed(),
        'C. 混合趋势版': BroadExtremeMixed(),
    }

    results = {}
    for name, algo in algorithms.items():
        logger.info(f"  回测 {name}...")
        r = run_backtest(df, algo)
        results[name] = r
        if r['signals'] > 0:
            logger.info(f"  {name:30s} | 信号:{r['signals']:3d} | 胜率:{r['win_rate']:.1f}% | 收益:{r['avg_return']:+.2f}% | 最大:{r['avg_max_return']:+.2f}% | Sharpe:{r['sharpe']:.3f}")
        else:
            logger.info(f"  {name:30s} | 信号:  0")

    # 保存JSON
    json_data = {k: {kk: vv for kk, vv in v.items() if kk != 'signal_list'} for k, v in results.items()}
    json_path = os.path.join(OUTPUT_DIR, '机器人ETF_算法优化.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON: {json_path}")

    # 打印摘要
    print(f"\n{'=' * 95}")
    print(f"机器人ETF 优化对比 (按胜率降序)")
    print(f"{'=' * 95}")
    print(f"{'算法':<32s} {'信号':>4s} {'胜率':>6s} {'均收益':>7s} {'均最大':>7s} {'Sharpe':>7s}")
    print("-" * 95)
    for name, r in sorted(results.items(), key=lambda x: (-x[1]['win_rate'], -x[1]['signals'])):
        marker = ''
        if '当前' in name:
            marker = ' <- 当前'
        elif '基准' in name:
            marker = ' <- 基准'
        print(f"{name:<32s} {r['signals']:4d} {r['win_rate']:5.1f}% {r['avg_return']:+6.2f}% {r['avg_max_return']:+6.2f}% {r['sharpe']:6.3f}{marker}")


if __name__ == '__main__':
    main()
