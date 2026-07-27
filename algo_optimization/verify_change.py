# -*- coding: utf-8 -*-
"""
验证有色金属ETF算法更换后的信号生成和回测
"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import ALGORITHM_MAP, get_algorithm
from etf_config import get_etf_by_code
import numpy as np

ETF_CODE = 'sh512400'
START_DATE = '2024-01-01'
THRESHOLD = 60

# 1. 验证配置
etf = get_etf_by_code(ETF_CODE)
print(f"ETF配置验证:")
print(f"  {etf.code} {etf.name} -> algorithm={etf.algorithm}")
assert etf.algorithm == 'volatility_breakout', f"算法应为volatility_breakout, 实际={etf.algorithm}"
print(f"  配置正确")

# 2. 验证算法可用
algo = get_algorithm(etf.algorithm)
print(f"  算法类: {algo.__class__.__name__}, name={algo.name}")

# 3. 回测验证
engine = DataEngine()
df = engine.get_history_kline(ETF_CODE)
df = calc_all_indicators(df)
print(f"\n回测验证 ({START_DATE}~最新):")
print(f"  数据: {len(df)}条")

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
    future_high = float(future['high'].max())
    max_ret = (future_high / buy_price - 1) * 100
    is_win = max_ret > 0.5

    signals.append({
        'date': date_str,
        'score': round(signal.score, 1),
        'max_return': round(max_ret, 2),
        'is_win': is_win,
        'reasons': signal.reasons[:2],
    })

n = len(signals)
wins = sum(1 for s in signals if s['is_win'])
wr = wins / n * 100 if n else 0
avg_ret = np.mean([s['max_return'] for s in signals]) if signals else 0

print(f"  信号数: {n}")
print(f"  胜率: {wr:.1f}%")
print(f"  平均最大收益: {avg_ret:+.2f}%")
print(f"  预期: 24信号, 91.7%胜率, +3.78%平均最大收益")
print(f"  {'PASS' if n == 24 and abs(wr - 91.7) < 0.1 else 'CHECK'}")

# 4. 显示最近3个信号
print(f"\n最近3个信号:")
for s in signals[-3:]:
    tag = 'WIN' if s['is_win'] else 'LOSS'
    print(f"  {s['date']} score={s['score']:.1f} [{tag}] max={s['max_return']:+.2f}% - {'; '.join(s['reasons'])}")

# 5. 同时显示当前(最新一天)的信号
print(f"\n最新交易日信号:")
latest_slice = df
latest_signal = algo.calculate(latest_slice)
print(f"  score={latest_signal.score:.1f} level={latest_signal.level}")
print(f"  reasons: {latest_signal.reasons}")
