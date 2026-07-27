# -*- coding: utf-8 -*-
"""
新能源ETF 算法变更验证
========================
验证: volatility_breakout -> new_energy_reversal
预期: 78信号, 78.2%胜率, +4.87%均最大收益
"""
import sys, os, logging
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import get_algorithm, ALGORITHM_MAP
from etf_config import get_etf_by_code

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ETF_CODE = 'sh516160'
ETF_NAME = '新能源ETF'
START_DATE = '2024-01-01'
THRESHOLD = 60

EXPECTED = {
    'signals': 78,
    'win_rate': 78.2,
    'avg_max_return': 4.87,
}


def main():
    # 1. 验证配置
    try:
        target = get_etf_by_code(ETF_CODE)
    except ValueError:
        print(f"FAIL: {ETF_CODE} 不在ETF列表中")
        return
    if target.algorithm != 'new_energy_reversal':
        print(f"FAIL: 算法仍为 {target.algorithm}, 期望 new_energy_reversal")
        return
    print(f"[OK] etf_config: {ETF_CODE} algorithm = {target.algorithm}")

    # 2. 验证算法注册
    if 'new_energy_reversal' not in ALGORITHM_MAP:
        print("FAIL: new_energy_reversal 不在 ALGORITHM_MAP")
        return
    print(f"[OK] ALGORITHM_MAP: new_energy_reversal 已注册 (共{len(ALGORITHM_MAP)}种算法)")

    # 3. 验证回测
    engine = DataEngine()
    df = engine.get_history_kline(ETF_CODE)
    if df is None:
        print(f"FAIL: 无法获取 {ETF_CODE} 数据")
        return
    df = calc_all_indicators(df)
    print(f"[OK] 数据: {len(df)}条, {df['date'].iloc[0].strftime('%Y-%m-%d')}~{df['date'].iloc[-1].strftime('%Y-%m-%d')}")

    algo = get_algorithm('new_energy_reversal')
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
        if signal.score < THRESHOLD:
            continue

        buy_price = float(df.iloc[i]['close'])
        future = df.iloc[i + 1: i + 1 + 3]
        if len(future) < 1:
            continue
        future_high = float(future['high'].max())
        max_ret = (future_high / buy_price - 1) * 100
        is_win = max_ret > 0.5
        signals.append({'date': date_str, 'score': round(signal.score, 1),
                        'max_return': round(max_ret, 2), 'is_win': is_win,
                        'level': signal.level, 'reasons': signal.reasons[:2]})

    n = len(signals)
    wins = sum(1 for s in signals if s['is_win'])
    wr = round(wins / n * 100, 1) if n > 0 else 0
    avg_max = round(np.mean([s['max_return'] for s in signals]), 2) if n > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"回测结果: {n}信号, {wr}%胜率, +{avg_max}%均最大收益")
    print(f"{'=' * 60}")

    # 对比预期
    ok = True
    if n == EXPECTED['signals']:
        print(f"[OK] 信号数: {n} == {EXPECTED['signals']}")
    else:
        print(f"[WARN] 信号数: {n} != {EXPECTED['signals']}")
        ok = False

    if abs(wr - EXPECTED['win_rate']) < 1:
        print(f"[OK] 胜率: {wr}% ~= {EXPECTED['win_rate']}%")
    else:
        print(f"[WARN] 胜率: {wr}% != {EXPECTED['win_rate']}%")
        ok = False

    if abs(avg_max - EXPECTED['avg_max_return']) < 0.5:
        print(f"[OK] 均最大收益: +{avg_max}% ~= +{EXPECTED['avg_max_return']}%")
    else:
        print(f"[WARN] 均最大收益: +{avg_max}% != +{EXPECTED['avg_max_return']}%")
        ok = False

    # 信号明细
    print(f"\n{'日期':<12s} {'分数':>5s} {'级别':<6s} {'最大收益':>8s} {'胜':>3s} 理由")
    print("-" * 70)
    for s in signals:
        win = 'Y' if s['is_win'] else 'N'
        reasons = '; '.join(s.get('reasons', [])) if s.get('reasons') else '-'
        print(f"{s['date']:<12s} {s['score']:5.1f} {s['level']:<6s} {s['max_return']:+7.2f}% {win:>3s} {reasons}")

    # 最新信号
    if signals:
        latest = signals[-1]
        print(f"\n最新信号: {latest['date']} score={latest['score']:.1f} level={latest['level']}")

    print(f"\n{'=' * 60}")
    print(f"验证结果: {'PASS' if ok else 'CHECK'}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
