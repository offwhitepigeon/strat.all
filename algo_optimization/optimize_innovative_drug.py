# -*- coding: utf-8 -*-
"""
Innovative Drug ETF (sz159992) - broad_reversal/extreme_reversal optimization
=============================================================================
Current algorithm: support_rebound (85 signals, 74.1% win, -0.33% return)
Best baselines:
  broad_reversal: 36 signals, 88.9% win, +0.75% return, +3.13% max
  extreme_reversal: 12 signals, 91.7% win, +0.59% return, +4.38% max
  seasonal_value: 58 signals, 84.5% win, +0.23% return, +2.59% max

Design 3 optimization variants based on broad_reversal:
  A. broad_reversal + MA200 trend filter (filter signals below MA200)
  B. broad_reversal + seasonal month weighting (boost favorable months)
  C. broad_reversal + Z-score extreme weighting (blend with extreme_reversal)
"""
import sys, os, json, logging
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import (ALGORITHM_MAP, BaseAlgorithm,
                        BroadReversalAlgorithm, ExtremeReversalAlgorithm,
                        SupportReboundAlgorithm)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'algo_optimization')
ETF_CODE = 'sz159992'
ETF_NAME = '创新药ETF'
START_DATE = '2024-01-01'
THRESHOLD = 60


# ======================================================================
# Variant A: broad_reversal + MA200 trend filter
# ======================================================================
class BroadReversalMA200(BaseAlgorithm):
    """
    broad_reversal + MA200 trend filter
    - Same 5 factors as broad_reversal (RSI 30%, BB 25%, consec_down 20%, KDJ 15%, Z+vol 10%)
    - NEW: MA200 trend filter - boost score when above MA200 (uptrend dip)
    - Hard filter: if below MA200 AND dev_ma200 < -15%, cap at 59 (bear market)
    """
    name = "broad_reversal_ma200"

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
        ma200 = float(last.get('ma200', 0))

        indicators.update({
            'rsi_14': rsi, 'rsi_7': rsi7, 'bb_percent_b': bb_pctb,
            'consec_down': consec_down, 'kdj_j': kdj_j,
            'vol_ratio': vol_ratio, 'dev_ma200': dev_ma200,
        })

        # F1: RSI (30%)
        if rsi < 25 and rsi7 < 20:
            s1 = 30
            reasons.append(f"RSI14={rsi:.0f}/RSI7={rsi7:.0f}")
        elif rsi < 30:
            s1 = 22
            reasons.append(f"RSI={rsi:.0f}")
        elif rsi < 35:
            s1 = 15
        elif rsi < 40:
            s1 = 8
        else:
            s1 = 0
        score += s1

        # F2: BB (25%)
        if bb_pctb < 0:
            s2 = 25
            reasons.append("BB下轨突破")
        elif bb_pctb < 0.05:
            s2 = 20
        elif bb_pctb < 0.15:
            s2 = 12
        else:
            s2 = 0
        score += s2

        # F3: consec_down (20%)
        if consec_down >= 5:
            s3 = 20
            reasons.append(f"连跌{consec_down}日")
        elif consec_down >= 4:
            s3 = 15
        elif consec_down >= 3:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # F4: KDJ (15%)
        if kdj_j < -5 and kdj_k < 20:
            s4 = 15
            reasons.append(f"KDJ J={kdj_j:.0f}")
        elif kdj_j < 10:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: Z+vol (10%)
        if zscore < -1.5 and vol_ratio > 1.2:
            s5 = 10
            reasons.append("Z+放量")
        elif zscore < -1:
            s5 = 5
        else:
            s5 = 0
        score += s5

        # F6: MA200 trend (bonus, not part of 100% base)
        # Above MA200 = uptrend dip (better), below MA200 = bear (risky)
        if dev_ma200 > 0:
            score = min(score + 5, 100)
            reasons.append(f"MA200上方({dev_ma200:+.1f}%)，趋势回踩")

        # Hard filter: deep bear market
        if dev_ma200 < -15:
            score = min(score, 59)
            reasons.append(f"MA200下方{dev_ma200:.1f}%，熊市封顶")

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# Variant B: broad_reversal + seasonal month weighting
# ======================================================================
class BroadReversalSeasonal(BaseAlgorithm):
    """
    broad_reversal + seasonal month boost
    - Same 5 factors as broad_reversal but reweighted (RSI 25%, BB 20%, consec 15%, KDJ 15%, Z+vol 10%)
    - F6: Seasonal month (15%) - boost in historically favorable months
      Based on seasonal_value logic: Q1 (Jan-Mar) and Q4 (Oct-Dec) tend to be favorable for pharma
    """
    name = "broad_reversal_seasonal"

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

        # Get current month from last date
        last_date = last.get('date')
        if last_date is not None:
            if hasattr(last_date, 'month'):
                month = last_date.month
            else:
                month = int(str(last_date)[5:7])
        else:
            month = 1

        indicators.update({
            'rsi_14': rsi, 'rsi_7': rsi7, 'bb_percent_b': bb_pctb,
            'consec_down': consec_down, 'kdj_j': kdj_j,
            'vol_ratio': vol_ratio, 'month': month,
        })

        # F1: RSI (25%)
        if rsi < 25 and rsi7 < 20:
            s1 = 25
            reasons.append(f"RSI14={rsi:.0f}/RSI7={rsi7:.0f}")
        elif rsi < 30:
            s1 = 18
            reasons.append(f"RSI={rsi:.0f}")
        elif rsi < 35:
            s1 = 12
        elif rsi < 40:
            s1 = 6
        else:
            s1 = 0
        score += s1

        # F2: BB (20%)
        if bb_pctb < 0:
            s2 = 20
            reasons.append("BB下轨突破")
        elif bb_pctb < 0.05:
            s2 = 16
        elif bb_pctb < 0.15:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # F3: consec_down (15%)
        if consec_down >= 5:
            s3 = 15
            reasons.append(f"连跌{consec_down}日")
        elif consec_down >= 4:
            s3 = 12
        elif consec_down >= 3:
            s3 = 8
        else:
            s3 = 0
        score += s3

        # F4: KDJ (15%)
        if kdj_j < -5 and kdj_k < 20:
            s4 = 15
            reasons.append(f"KDJ J={kdj_j:.0f}")
        elif kdj_j < 10:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: Z+vol (10%)
        if zscore < -1.5 and vol_ratio > 1.2:
            s5 = 10
            reasons.append("Z+放量")
        elif zscore < -1:
            s5 = 5
        else:
            s5 = 0
        score += s5

        # F6: Seasonal month (15%)
        # Favorable months for pharma: Jan-Mar (Q1 rally), Sep-Nov (Q4 healthcare conferences)
        favorable = {1, 2, 3, 9, 10, 11}
        neutral = {4, 5, 12}
        if month in favorable:
            s6 = 15
            reasons.append(f"{month}月季节性 favorable")
        elif month in neutral:
            s6 = 8
        else:
            s6 = 0
        score += s6

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# Variant C: broad_reversal + Z-score extreme weighting (blend with extreme)
# ======================================================================
class BroadExtremeBlend(BaseAlgorithm):
    """
    broad_reversal core + extreme_reversal Z-score weighting
    - F1: RSI (25%, broad core)
    - F2: BB (20%, broad)
    - F3: Z-Score extreme (20%, from extreme_reversal, higher weight)
      Z<-3->20, Z<-2.5->18, Z<-2->15, Z<-1.5->10, Z<-1->5
    - F4: KDJ (15%, broad)
    - F5: consec_down + vol (10%, mixed)
    - F6: MA60 deviation (10%, trend context)
      dev<-10%->10, <-7%->7, <-5%->5, <-3%->3
    Hard filter: dev_ma60 > 8% -> cap 59 (overbought, don't catch falling knife in uptrend)
    """
    name = "broad_extreme_blend"

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

        # F1: RSI (25%)
        if rsi < 25 and rsi7 < 20:
            s1 = 25
            reasons.append(f"RSI14={rsi:.0f}/RSI7={rsi7:.0f}")
        elif rsi < 30:
            s1 = 20
            reasons.append(f"RSI={rsi:.0f}")
        elif rsi < 35:
            s1 = 12
        elif rsi < 40:
            s1 = 6
        else:
            s1 = 0
        score += s1

        # F2: BB (20%)
        if bb_pctb < 0:
            s2 = 20
            reasons.append("BB下轨突破")
        elif bb_pctb < 0.05:
            s2 = 16
        elif bb_pctb < 0.15:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # F3: Z-Score extreme (20%, from extreme_reversal)
        if zscore < -3:
            s3 = 20
            reasons.append(f"Z={zscore:.1f} extreme")
        elif zscore < -2.5:
            s3 = 18
            reasons.append(f"Z={zscore:.1f}")
        elif zscore < -2:
            s3 = 15
            reasons.append(f"Z={zscore:.1f}")
        elif zscore < -1.5:
            s3 = 10
        elif zscore < -1:
            s3 = 5
        else:
            s3 = 0
        score += s3

        # F4: KDJ (15%)
        if kdj_j < -5 and kdj_k < 20:
            s4 = 15
            reasons.append(f"KDJ J={kdj_j:.0f}")
        elif kdj_j < 10:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: consec_down + vol (10%, mixed)
        if vol_ratio > 1.5 and consec_down >= 3:
            s5 = 10
            reasons.append(f"连跌{consec_down}+放量{vol_ratio:.1f}")
        elif vol_ratio > 1.2 and consec_down >= 2:
            s5 = 6
        elif consec_down >= 4:
            s5 = 5
        else:
            s5 = 0
        score += s5

        # F6: MA60 deviation (10%)
        if dev_ma60 < -10:
            s6 = 10
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%")
        elif dev_ma60 < -7:
            s6 = 7
        elif dev_ma60 < -5:
            s6 = 5
        elif dev_ma60 < -3:
            s6 = 3
        else:
            s6 = 0
        score += s6

        # Hard filter: overbought
        if dev_ma60 > 8:
            score = min(score, 59)
            reasons.append(f"MA60上方+{dev_ma60:.1f}%，超买封顶")

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


def deep_compare(df, results, names):
    """Print detailed comparison of selected algorithms"""
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
        wins_list = [s for s in sigs if s['is_win']]
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
        'support_rebound (current)': SupportReboundAlgorithm(),
        'broad_reversal (baseline1)': BroadReversalAlgorithm(),
        'extreme_reversal (baseline2)': ExtremeReversalAlgorithm(),
        'A. broad+MA200': BroadReversalMA200(),
        'B. broad+seasonal': BroadReversalSeasonal(),
        'C. broad+extreme blend': BroadExtremeBlend(),
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
    json_path = os.path.join(OUTPUT_DIR, '创新药ETF_算法优化.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON saved: {json_path}")

    # Deep comparison of top candidates
    deep_compare(df, results, [
        'broad_reversal (baseline1)',
        'extreme_reversal (baseline2)',
        'A. broad+MA200',
        'B. broad+seasonal',
        'C. broad+extreme blend',
    ])

    # Summary table
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
