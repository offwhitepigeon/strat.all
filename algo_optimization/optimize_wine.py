# -*- coding: utf-8 -*-
"""
Wine ETF (sh512690) - financial_value/seasonal_value optimization
=================================================================
Current algorithm: seasonal_value (75 signals, 77.3% win, +0.05% return)
Best baseline:
  financial_value: 72 signals, 84.7% win, +0.20% return, +2.38% max
  extreme_reversal: 22 signals, 81.8% win, +0.20% return, +2.57% max

financial_value factors: MA60/MA200 dev(30%) + RSI(25%) + Z-score(15%) + BB(15%) + Vol+KDJ(15%)
seasonal_value factors: Seasonal(25%) + RSI(25%) + KDJ(20%) + MA20+BB(15%) + ConsecDown+momentum(15%)

Design 3 variants based on financial_value:
  A. financial_value + MA200 trend filter (proven pattern from gold/pharma/innovation)
  B. financial_value + seasonal month factor (blend with seasonal_value, replace F5)
  C. financial_value + momentum narrowing (blend with seasonal_value F5, replace F5)
"""
import sys, os, json, logging
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import (BaseAlgorithm, FinancialValueAlgorithm,
                        SeasonalValueAlgorithm, ExtremeReversalAlgorithm)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'algo_optimization')
ETF_CODE = 'sh512690'
ETF_NAME = '酒ETF'
START_DATE = '2024-01-01'
THRESHOLD = 60


# ======================================================================
# Variant A: financial_value + MA200 trend filter
# ======================================================================
class FinValueMA200(BaseAlgorithm):
    """
    financial_value + MA200 trend filter
    - Same 5 factors as financial_value
    - F6: MA200 trend bonus (above MA200 +5, deep bear cap 59)
    """
    name = "fin_value_ma200"

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

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma60': dev_ma60, 'dev_ma200': dev_ma200,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
        })

        # F1: MA60/MA200 dev (30%)
        if dev_ma60 < -10 and dev_ma200 < -15:
            s1 = 30
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%")
        elif dev_ma60 < -8:
            s1 = 22
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%")
        elif dev_ma60 < -5:
            s1 = 12
        else:
            s1 = 0
        score += s1

        # F2: RSI (25%)
        if rsi < 25:
            s2 = 25
            reasons.append(f"RSI={rsi:.0f}")
        elif rsi < 30:
            s2 = 18
            reasons.append(f"RSI={rsi:.0f}")
        elif rsi < 35:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # F3: Z-score (15%)
        if zscore < -2:
            s3 = 15
            reasons.append(f"Z={zscore:.1f}")
        elif zscore < -1.5:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # F4: BB (15%)
        if bb_pctb < 0.05:
            s4 = 15
            reasons.append("BB下轨")
        elif bb_pctb < 0.15:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: Vol+KDJ (15%)
        if vol_ratio > 1.5 and kdj_j < 10:
            s5 = 15
            reasons.append("放量+KDJ超卖")
        elif vol_ratio > 1.2:
            s5 = 8
        elif kdj_j < 10:
            s5 = 5
        else:
            s5 = 0
        score += s5

        # F6: MA200 trend filter
        if dev_ma200 > 0:
            score = min(score + 5, 100)
            reasons.append(f"MA200上方({dev_ma200:+.1f}%)")

        if dev_ma200 < -15:
            score = min(score, 59)
            reasons.append(f"MA200下方{dev_ma200:.1f}%，熊市封顶")

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# Variant B: financial_value + seasonal month factor (replace F5)
# ======================================================================
class FinValueSeasonal(BaseAlgorithm):
    """
    financial_value + seasonal month (replace F5 vol+KDJ with seasonal)
    - F1: MA60/MA200 dev (28%)
    - F2: RSI (25%)
    - F3: Z-score (15%)
    - F4: BB (15%)
    - F5: Seasonal month (17%) - from seasonal_value
    """
    name = "fin_value_seasonal"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        if 'date' in df.columns:
            last_date = df['date'].iloc[-1]
            if hasattr(last_date, 'month'):
                month = last_date.month
            else:
                month = int(str(last_date)[5:7])
        else:
            month = datetime.now().month

        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma60 = float(last.get('dev_ma60', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        kdj_j = float(last.get('kdj_j', 50))

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma60': dev_ma60, 'dev_ma200': dev_ma200,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio, 'month': month,
        })

        # F1: MA60/MA200 dev (28%)
        if dev_ma60 < -10 and dev_ma200 < -15:
            s1 = 28
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%")
        elif dev_ma60 < -8:
            s1 = 20
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%")
        elif dev_ma60 < -5:
            s1 = 11
        else:
            s1 = 0
        score += s1

        # F2: RSI (25%)
        if rsi < 25:
            s2 = 25
            reasons.append(f"RSI={rsi:.0f}")
        elif rsi < 30:
            s2 = 18
            reasons.append(f"RSI={rsi:.0f}")
        elif rsi < 35:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # F3: Z-score (15%)
        if zscore < -2:
            s3 = 15
            reasons.append(f"Z={zscore:.1f}")
        elif zscore < -1.5:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # F4: BB (15%)
        if bb_pctb < 0.05:
            s4 = 15
            reasons.append("BB下轨")
        elif bb_pctb < 0.15:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: Seasonal month (17%)
        # 白酒季节性: 春季调整(3-4月)布局, 夏季淡季(7-8月)超卖, 旺季回调(12-2月)
        if month in [3, 4]:
            s5 = 17
            reasons.append(f"{month}月春季调整期")
        elif month in [7, 8]:
            s5 = 13
            reasons.append(f"{month}月夏季淡季")
        elif month in [12, 1, 2]:
            s5 = 10
            reasons.append(f"{month}月旺季回调")
        elif month in [10, 11]:
            s5 = 7
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# Variant C: financial_value + momentum narrowing (replace F5)
# ======================================================================
class FinValueMomentum(BaseAlgorithm):
    """
    financial_value + consec_down momentum narrowing (from seasonal_value F5)
    - F1: MA60/MA200 dev (28%)
    - F2: RSI (25%)
    - F3: Z-score (15%)
    - F4: BB (15%)
    - F5: Consec down + momentum narrowing (17%) - momentum_5 > momentum_20 = slowing decline
    """
    name = "fin_value_momentum"

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
        momentum_5 = float(last.get('momentum_5', 0))
        momentum_20 = float(last.get('momentum_20', 0))

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma60': dev_ma60, 'dev_ma200': dev_ma200,
            'bb_percent_b': bb_pctb, 'vol_ratio': vol_ratio,
            'consec_down': consec_down,
        })

        # F1: MA60/MA200 dev (28%)
        if dev_ma60 < -10 and dev_ma200 < -15:
            s1 = 28
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%")
        elif dev_ma60 < -8:
            s1 = 20
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%")
        elif dev_ma60 < -5:
            s1 = 11
        else:
            s1 = 0
        score += s1

        # F2: RSI (25%)
        if rsi < 25:
            s2 = 25
            reasons.append(f"RSI={rsi:.0f}")
        elif rsi < 30:
            s2 = 18
            reasons.append(f"RSI={rsi:.0f}")
        elif rsi < 35:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # F3: Z-score (15%)
        if zscore < -2:
            s3 = 15
            reasons.append(f"Z={zscore:.1f}")
        elif zscore < -1.5:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # F4: BB (15%)
        if bb_pctb < 0.05:
            s4 = 15
            reasons.append("BB下轨")
        elif bb_pctb < 0.15:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: Consec down + momentum narrowing (17%)
        if consec_down >= 3 and momentum_5 > momentum_20:
            s5 = 17
            reasons.append(f"连跌{consec_down}日+跌势放缓")
        elif consec_down >= 4:
            s5 = 10
            reasons.append(f"连跌{consec_down}日")
        elif vol_ratio > 1.5 and kdj_j < 10:
            s5 = 10
            reasons.append("放量+KDJ超卖")
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# Backtest
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


def deep_compare(results, names):
    """Print detailed comparison"""
    print(f"\n{'='*100}")
    print(f"Deep Comparison - {ETF_NAME}")
    print(f"{'='*100}")

    for name in names:
        r = results[name]
        sigs = r['signal_list']
        if not sigs:
            print(f"\n{name}: no signals")
            continue

        rets = [s['close_return'] for s in sigs]
        max_rets = [s['max_return'] for s in sigs]
        losses_list = [s for s in sigs if not s['is_win']]

        print(f"\n{name}")
        print(f"  Signals: {r['signals']}, Win: {r['win_rate']}%, "
              f"Avg ret: {r['avg_return']:+.2f}%, Avg max: {r['avg_max_return']:+.2f}%, "
              f"Sharpe: {r['sharpe']:.3f}")
        print(f"  Best: {r['best']:+.2f}%, Worst: {r['worst']:+.2f}%, "
              f"Avg score: {r['avg_score']:.1f}")

        # Return distribution
        bins = [-15, -5, 0, 1, 3, 5, 10, 20]
        labels = ['<-5%', '-5~0%', '0~1%', '1~3%', '3~5%', '5~10%', '>10%']
        dist = [0] * len(labels)
        for r_val in rets:
            for j in range(len(bins) - 1):
                if bins[j] <= r_val < bins[j + 1]:
                    dist[j] += 1
                    break
        print(f"  Return distribution: {dict(zip(labels, dist))}")

        # Yearly breakdown
        yearly = {}
        for s in sigs:
            yr = s['date'][:4]
            if yr not in yearly:
                yearly[yr] = {'sig': 0, 'win': 0, 'ret': []}
            yearly[yr]['sig'] += 1
            if s['is_win']:
                yearly[yr]['win'] += 1
            yearly[yr]['ret'].append(s['close_return'])
        print(f"  Yearly:")
        for yr in sorted(yearly.keys()):
            d = yearly[yr]
            avg_r = np.mean(d['ret'])
            wr = d['win'] / d['sig'] * 100
            print(f"    {yr}: {d['sig']} sig, {wr:.0f}% win, avg {avg_r:+.2f}%")

        # Loss analysis
        if losses_list:
            print(f"  Losses ({len(losses_list)}):")
            for s in losses_list[:5]:
                print(f"    {s['date']} score={s['score']:.0f} ret={s['close_return']:+.2f}% "
                      f"max={s['max_return']:+.2f}%")


def main():
    logger.info(f"Start {ETF_NAME} optimization...")

    engine = DataEngine()
    df = engine.get_history_kline(ETF_CODE)
    if df is None:
        logger.error(f"Cannot get data for {ETF_CODE}")
        return
    df_info = f"{len(df)} bars, {df['date'].iloc[0].strftime('%Y-%m-%d')}~{df['date'].iloc[-1].strftime('%Y-%m-%d')}"
    logger.info(f"{ETF_NAME} data: {df_info}")
    df = calc_all_indicators(df)

    algorithms = {
        'seasonal_value (current)': SeasonalValueAlgorithm(),
        'financial_value (baseline)': FinancialValueAlgorithm(),
        'extreme_reversal (baseline2)': ExtremeReversalAlgorithm(),
        'A. fin+MA200': FinValueMA200(),
        'B. fin+seasonal': FinValueSeasonal(),
        'C. fin+momentum': FinValueMomentum(),
    }

    results = {}
    for name, algo in algorithms.items():
        logger.info(f"  Backtesting {name}...")
        r = run_backtest(df, algo)
        results[name] = r
        if r['signals'] > 0:
            logger.info(f"  {name:35s} | sig:{r['signals']:3d} | win:{r['win_rate']:.1f}% | "
                        f"ret:{r['avg_return']:+.2f}% | max:{r['avg_max_return']:+.2f}% | "
                        f"Sharpe:{r['sharpe']:.3f}")
        else:
            logger.info(f"  {name:35s} | sig:  0")

    # Save JSON
    json_data = {k: {kk: vv for kk, vv in v.items() if kk != 'signal_list'} for k, v in results.items()}
    json_path = os.path.join(OUTPUT_DIR, '酒ETF_算法优化.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON saved: {json_path}")

    # Deep comparison
    deep_compare(results, [
        'financial_value (baseline)',
        'extreme_reversal (baseline2)',
        'A. fin+MA200',
        'B. fin+seasonal',
        'C. fin+momentum',
    ])

    # Summary
    print(f"\n{'='*95}")
    print(f"{ETF_NAME} Optimization Summary (sorted by win rate)")
    print(f"{'='*95}")
    print(f"{'Algorithm':<37s} {'Sig':>4s} {'Win':>6s} {'Ret':>7s} {'Max':>7s} {'Sharpe':>7s}")
    print("-" * 95)
    for name, r in sorted(results.items(), key=lambda x: (-x[1]['win_rate'], -x[1]['signals'])):
        marker = ''
        if 'current' in name:
            marker = ' <- current'
        elif 'baseline' in name:
            marker = ' <- baseline'
        print(f"{name:<37s} {r['signals']:4d} {r['win_rate']:5.1f}% {r['avg_return']:+6.2f}% "
              f"{r['avg_max_return']:+6.2f}% {r['sharpe']:6.3f}{marker}")


if __name__ == '__main__':
    main()
