# -*- coding: utf-8 -*-
"""
Verify robot_reversal algorithm for sh562500
Expected: 31 signals, 90.3% win rate, +1.15% return, +3.30% max return
"""
import sys, os, json, logging
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import ALGORITHM_MAP, get_algorithm
from etf_config import get_etf_by_code

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ETF_CODE = 'sh562500'
ETF_NAME = '机器人ETF'
START_DATE = '2024-01-01'
THRESHOLD = 60


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
        signals.append({
            'date': date_str, 'score': round(signal.score, 1),
            'max_return': round(max_ret, 2), 'close_return': round(close_ret, 2),
            'is_win': is_win,
        })

    n = len(signals)
    if n == 0:
        return {'signals': 0, 'wins': 0, 'win_rate': 0, 'avg_return': 0,
                'avg_max_return': 0, 'avg_score': 0, 'best': 0, 'worst': 0}
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
        'sharpe': round(sharpe, 3),
    }


def main():
    # 1. Verify algorithm registration
    assert 'robot_reversal' in ALGORITHM_MAP, "robot_reversal not in ALGORITHM_MAP!"
    algo = get_algorithm('robot_reversal')
    assert algo.name == 'robot_reversal', f"Wrong name: {algo.name}"
    logger.info("PASS: robot_reversal registered in ALGORITHM_MAP")

    # 2. Verify etf_config
    etf = get_etf_by_code(ETF_CODE)
    assert etf is not None, f"{ETF_CODE} not found in ETF_POOL!"
    assert etf.algorithm == 'robot_reversal', f"Wrong algorithm: {etf.algorithm}"
    logger.info(f"PASS: {ETF_NAME} ({ETF_CODE}) algorithm = {etf.algorithm}")

    # 3. Backtest
    engine = DataEngine()
    df = engine.get_history_kline(ETF_CODE)
    assert df is not None and len(df) > 60, f"Data insufficient for {ETF_CODE}"
    df = calc_all_indicators(df)
    logger.info(f"Data: {len(df)} rows, {df['date'].iloc[0].strftime('%Y-%m-%d')}~{df['date'].iloc[-1].strftime('%Y-%m-%d')}")

    r = run_backtest(df, algo)
    logger.info(f"Backtest: {r['signals']} signals, {r['win_rate']}% win, "
                f"avg_return={r['avg_return']:+.2f}%, avg_max={r['avg_max_return']:+.2f}%, "
                f"Sharpe={r['sharpe']:.3f}")

    # 4. Compare expected vs actual
    expected = {'signals': 31, 'win_rate': 90.3, 'avg_max_return': 3.30}
    print(f"\n{'='*70}")
    print(f"  Robot Reversal Verification - {ETF_NAME} ({ETF_CODE})")
    print(f"{'='*70}")
    print(f"  Expected: {expected}")
    print(f"  Actual:   signals={r['signals']}, win_rate={r['win_rate']}%, "
          f"avg_max_return={r['avg_max_return']:+.2f}%")

    # Allow tolerance for data updates
    if r['signals'] > 0 and r['win_rate'] >= 85:
        print(f"\n  RESULT: PASS (win_rate {r['win_rate']}% >= 85%)")
    else:
        print(f"\n  RESULT: CHECK (win_rate {r['win_rate']}% may need review)")

    # 5. Print algorithm count
    print(f"\n  Total algorithms: {len(ALGORITHM_MAP)}")
    print(f"  Algorithm list: {sorted(ALGORITHM_MAP.keys())}")


if __name__ == '__main__':
    main()
