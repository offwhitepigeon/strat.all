# -*- coding: utf-8 -*-
"""
黄金ETF - support_rebound 信号深度分析 + 优化变体对比
"""
import sys, os, json, logging
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import ALGORITHM_MAP, BaseAlgorithm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'algo_optimization')
ETF_CODE = 'sh518880'
ETF_NAME = '黄金ETF'
START_DATE = '2024-01-01'
THRESHOLD = 60


# ======================================================================
# 变体1: 增强型 - 加入MA200趋势过滤
# ======================================================================
class SupportRebound增强(BaseAlgorithm):
    """
    support_rebound 黄金增强版
    - 新增F6: MA200趋势确认(15%) - 黄金是长期趋势品种,趋势下方不抄底
    - F1/F2/F4降权腾空间
    - F3 MACD保持(黄金MACD背离信号有效)
    """
    name = "support_rebound_gold_enhanced"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        close = df['close']
        high = df['high']
        low = df['low']
        rsi = float(last.get('rsi_14', 50))
        macd_hist = float(last.get('macd_hist', 0))
        ma60 = float(last.get('ma60', 0))
        ma200 = float(last.get('ma200', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        zscore = float(last.get('zscore_20', 0))
        kdj_j = float(last.get('kdj_j', 50))
        momentum_20 = float(last.get('momentum_20', 0))

        if len(low) >= 60:
            support = float(low.iloc[-60:].min())
        else:
            support = float(low.min())

        if support > 0 and price > 0:
            dev_support = (price - support) / support * 100
        else:
            dev_support = 100

        indicators.update({
            'rsi_14': rsi, 'macd_hist': macd_hist, 'dev_support': dev_support,
            'bb_percent_b': bb_pctb, 'zscore_20': zscore, 'kdj_j': kdj_j,
            'ma200': ma200, 'momentum_20': momentum_20,
        })

        # F1: 支撑位 (25%, 降权)
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

        # F2: RSI超卖 (20%, 降权)
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

        # F3: MACD背离 (20%, 保持)
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

        # F4: 布林带 (10%, 降权)
        if bb_pctb < 0.05:
            s4 = 10
            reasons.append("布林带下轨")
        elif bb_pctb < 0.15:
            s4 = 6
        else:
            s4 = 0
        score += s4

        # F5: Z+KDJ (10%, 保持)
        if zscore < -1.5 and kdj_j < 10:
            s5 = 10
            reasons.append("Z偏低+KDJ超卖")
        elif zscore < -1:
            s5 = 5
        else:
            s5 = 0
        score += s5

        # F6: MA200趋势确认 (15%, 新增)
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


# ======================================================================
# 变体2: 量价型 - 加入放量+动量
# ======================================================================
class SupportRebound量价(BaseAlgorithm):
    """
    support_rebound 黄金量价版
    - 新增F6: 放量+动量组合(20%)
    - F1/F2/F3/F4/F5全部降权
    - 底部放量是黄金强信号
    """
    name = "support_rebound_gold_volprice"

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
        vol_ratio = float(last.get('vol_ratio_20', 1))
        momentum_20 = float(last.get('momentum_20', 0))

        if len(low) >= 60:
            support = float(low.iloc[-60:].min())
        else:
            support = float(low.min())

        dev_support = (price - support) / support * 100 if support > 0 and price > 0 else 100

        indicators.update({
            'rsi_14': rsi, 'dev_support': dev_support,
            'bb_percent_b': bb_pctb, 'zscore_20': zscore,
            'kdj_j': kdj_j, 'vol_ratio': vol_ratio, 'momentum_20': momentum_20,
        })

        # F1: 支撑位 (20%)
        if dev_support <= 2:
            s1 = 20
            reasons.append(f"触及支撑(偏离{dev_support:+.1f}%)")
        elif dev_support <= 5:
            s1 = 14
        elif dev_support <= 8:
            s1 = 8
        else:
            s1 = 0
        score += s1

        # F2: RSI (20%)
        if rsi < 25:
            s2 = 20
            reasons.append(f"RSI={rsi:.0f},极度超卖")
        elif rsi < 30:
            s2 = 15
            reasons.append(f"RSI={rsi:.0f},超卖")
        elif rsi < 35:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # F3: MACD (15%)
        if len(df) >= 10:
            recent_hist = df['macd_hist'].iloc[-5:].values
            if macd_hist < 0 and recent_hist[-1] > recent_hist[0]:
                s3 = 15
                reasons.append("MACD底背离")
            elif macd_hist > 0 and float(df['macd_hist'].iloc[-2]) < 0:
                s3 = 12
                reasons.append("MACD翻红")
            elif macd_hist < 0 and macd_hist > float(df['macd_hist'].iloc[-3]):
                s3 = 8
            else:
                s3 = 0
        else:
            s3 = 0
        score += s3

        # F4: 布林带 (10%)
        if bb_pctb < 0.05:
            s4 = 10
        elif bb_pctb < 0.15:
            s4 = 6
        else:
            s4 = 0
        score += s4

        # F5: KDJ (10%) - 独立,不要求Z
        if kdj_j < 0:
            s5 = 10
            reasons.append(f"KDJ超卖(J={kdj_j:.0f})")
        elif kdj_j < 10:
            s5 = 6
        elif kdj_j < 20:
            s5 = 3
        else:
            s5 = 0
        score += s5

        # F6: 放量+动量 (25%, 新增, 权重大)
        if vol_ratio > 2.0 and momentum_20 > 0:
            s6 = 25
            reasons.append(f"放量+上升动量(量比{vol_ratio:.1f})")
        elif vol_ratio > 1.5 and momentum_20 > 0:
            s6 = 18
            reasons.append(f"放量+动量正(量比{vol_ratio:.1f})")
        elif vol_ratio > 1.5:
            s6 = 12
            reasons.append(f"放量(量比{vol_ratio:.1f})")
        elif vol_ratio > 1.2:
            s6 = 6
        else:
            s6 = 0
        score += s6

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 变体3: 混合型 - 趋势+放量+KDJ独立
# ======================================================================
class SupportRebound混合(BaseAlgorithm):
    """
    support_rebound 黄金混合版
    - MA200趋势过滤(15%) + 放量(15%) + KDJ独立(10%)
    - F1支撑25%, F2 RSI20%, F3 MACD15%
    - 趋势+放量+支撑三重确认
    """
    name = "support_rebound_gold_mixed"

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
        kdj_k = float(last.get('kdj_k', 50))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        momentum_20 = float(last.get('momentum_20', 0))

        if len(low) >= 60:
            support = float(low.iloc[-60:].min())
        else:
            support = float(low.min())

        dev_support = (price - support) / support * 100 if support > 0 and price > 0 else 100

        indicators.update({
            'rsi_14': rsi, 'dev_support': dev_support,
            'bb_percent_b': bb_pctb, 'kdj_j': kdj_j,
            'vol_ratio': vol_ratio, 'ma200': ma200, 'momentum_20': momentum_20,
        })

        # F1: 支撑位 (25%)
        if dev_support <= 2:
            s1 = 25
            reasons.append(f"触及支撑(偏离{dev_support:+.1f}%)")
        elif dev_support <= 5:
            s1 = 18
            reasons.append(f"接近支撑(偏离{dev_support:+.1f}%)")
        elif dev_support <= 8:
            s1 = 10
        else:
            s1 = 0
        score += s1

        # F2: RSI (20%)
        if rsi < 25:
            s2 = 20
            reasons.append(f"RSI={rsi:.0f},极度超卖")
        elif rsi < 30:
            s2 = 15
            reasons.append(f"RSI={rsi:.0f},超卖")
        elif rsi < 35:
            s2 = 10
        else:
            s2 = 0
        score += s2

        # F3: MACD (15%)
        if len(df) >= 10:
            recent_hist = df['macd_hist'].iloc[-5:].values
            if macd_hist < 0 and recent_hist[-1] > recent_hist[0]:
                s3 = 15
                reasons.append("MACD底背离")
            elif macd_hist > 0 and float(df['macd_hist'].iloc[-2]) < 0:
                s3 = 12
                reasons.append("MACD翻红")
            elif macd_hist < 0 and macd_hist > float(df['macd_hist'].iloc[-3]):
                s3 = 8
            else:
                s3 = 0
        else:
            s3 = 0
        score += s3

        # F4: KDJ独立 (10%) - 不要求Z
        if kdj_j < 0 and kdj_k < 20:
            s4 = 10
            reasons.append(f"KDJ超卖(J={kdj_j:.0f})")
        elif kdj_j < 10:
            s4 = 6
        elif kdj_j < 20:
            s4 = 3
        else:
            s4 = 0
        score += s4

        # F5: MA200趋势 (15%, 新增)
        if ma200 > 0 and price > ma200 and momentum_20 > 0:
            s5 = 15
            reasons.append("上升趋势中回调")
        elif ma200 > 0 and price > ma200:
            s5 = 10
            reasons.append("价格在MA200上方")
        elif ma200 > 0 and price > ma200 * 0.97:
            s5 = 5
        else:
            s5 = 0
        score += s5

        # F6: 放量 (15%, 新增)
        if vol_ratio > 1.8 and dev_support < 5:
            s6 = 15
            reasons.append(f"放量触支撑(量比{vol_ratio:.1f})")
        elif vol_ratio > 1.3:
            s6 = 8
        else:
            s6 = 0
        score += s6

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 回测引擎 (复用)
# ======================================================================
def run_backtest(df, algo, threshold=THRESHOLD, hold_days=3):
    signals = []
    for i in range(60, len(df) - hold_days):
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
        future = df.iloc[i + 1: i + 1 + hold_days]
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
        last = df.iloc[i]
        ind_data = {
            'rsi_14': float(last.get('rsi_14', 50)),
            'macd_hist': float(last.get('macd_hist', 0)),
            'bb_percent_b': float(last.get('bb_percent_b', 0.5)),
            'zscore_20': float(last.get('zscore_20', 0)),
            'kdj_j': float(last.get('kdj_j', 50)),
            'vol_ratio': float(last.get('vol_ratio_20', 1)),
            'momentum_20': float(last.get('momentum_20', 0)),
            'ma200': float(last.get('ma200', 0)),
            'close_price': float(last.get('close', 0)),
        }
        # support
        if len(df) >= 60:
            ind_data['dev_support'] = (float(last.get('close',0)) - float(df['low'].iloc[max(0,i-59):i+1].min())) / max(float(df['low'].iloc[max(0,i-59):i+1].min()), 0.001) * 100
        signals.append({
            'date': date_str, 'score': round(signal.score, 1),
            'buy_price': round(buy_price, 4),
            'max_return': round(max_ret, 2), 'close_return': round(close_ret, 2),
            'is_win': is_win, 'days_to_win': days_to_win,
            'reasons': signal.reasons[:3], 'indicators': ind_data,
        })
    return signals


def compute_stats(signals):
    n = len(signals)
    if n == 0:
        return {'n': 0, 'win_rate': 0, 'avg_return': 0, 'avg_max_return': 0,
                'sharpe': 0, 'worst': 0, 'avg_loss': 0, 'signal_list': []}
    rets = [s['close_return'] for s in signals]
    max_rets = [s['max_return'] for s in signals]
    wins = [s for s in signals if s['is_win']]
    losses = [s for s in signals if not s['is_win']]
    loss_rets = [s['close_return'] for s in losses] if losses else []
    return {
        'n': n, 'wins': len(wins), 'losses': len(losses),
        'win_rate': round(len(wins) / n * 100, 1),
        'avg_return': round(np.mean(rets), 2),
        'avg_max_return': round(np.mean(max_rets), 2),
        'std_return': round(np.std(rets), 2),
        'sharpe': round(np.mean(rets) / np.std(rets), 2) if np.std(rets) > 0 else 0,
        'best': round(max(max_rets), 2),
        'worst': round(min(rets), 2),
        'worst_max': round(min(max_rets), 2),
        'avg_win_max': round(np.mean([s['max_return'] for s in wins]), 2) if wins else 0,
        'avg_loss': round(np.mean(loss_rets), 2) if loss_rets else 0,
        'signal_list': signals,
    }


def main():
    logger.info("加载数据...")
    engine = DataEngine()
    df = engine.get_history_kline(ETF_CODE)
    df = calc_all_indicators(df)
    logger.info(f"数据: {len(df)}条")

    # 原版
    logger.info("运行原版 support_rebound...")
    original_algo = ALGORITHM_MAP['support_rebound']()
    original_signals = run_backtest(df, original_algo)
    original_stats = compute_stats(original_signals)
    logger.info(f"  原版: {original_stats['n']}信号, {original_stats['win_rate']}%, Sharpe={original_stats['sharpe']}")

    # 分析亏损和边缘信号
    print("\n" + "="*100)
    print("原版信号分析 (亏损 + 边缘盈利)")
    print("="*100)
    for s in original_signals:
        if not s['is_win'] or s['max_return'] < 1.5:
            ind = s.get('indicators', {})
            tag = "LOSS" if not s['is_win'] else "MARGINAL"
            print(f"  [{tag}] {s['date']} score={s['score']:.1f} max={s['max_return']:+.2f}% close={s['close_return']:+.2f}%")
            print(f"        RSI={ind.get('rsi_14',0):.1f} Sup_dev={ind.get('dev_support',0):.1f}% "
                  f"MACD={ind.get('macd_hist',0):.4f} BB%B={ind.get('bb_percent_b',0):.2f} "
                  f"Z={ind.get('zscore_20',0):.1f} KDJ_J={ind.get('kdj_j',0):.0f} "
                  f"Vol={ind.get('vol_ratio',0):.1f} M20={ind.get('momentum_20',0):+.1f}% "
                  f"vsMA200={((ind.get('close_price',0)/max(ind.get('ma200',1),0.001)-1)*100):+.1f}%")

    # 变体
    variants = []
    for algo_cls, label in [
        (SupportRebound增强, '增强版(MA200趋势)'),
        (SupportRebound量价, '量价版(放量+动量)'),
        (SupportRebound混合, '混合版(趋势+放量+KDJ)'),
    ]:
        algo = algo_cls()
        logger.info(f"运行 {label}...")
        sigs = run_backtest(df, algo)
        stats = compute_stats(sigs)
        logger.info(f"  {label}: {stats['n']}信号, {stats['win_rate']}%, Sharpe={stats['sharpe']}")
        variants.append((label, stats))

    # 输出
    print(f"\n{'='*90}")
    print(f"{'算法':<30s} {'信号':>4s} {'胜率':>6s} {'均收益':>7s} {'均最大':>7s} {'Sharpe':>7s} {'最差':>7s} {'亏损均':>7s}")
    print("-"*90)
    print(f"{'原版 support_rebound':<30s} {original_stats['n']:4d} {original_stats['win_rate']:5.1f}% "
          f"{original_stats['avg_return']:+6.2f}% {original_stats['avg_max_return']:+6.2f}% "
          f"{original_stats['sharpe']:7.2f} {original_stats['worst']:+6.2f}% {original_stats['avg_loss']:+6.2f}%")
    for name, stats in variants:
        better = (stats['win_rate'] >= original_stats['win_rate'] and
                  stats['sharpe'] > original_stats['sharpe'] and
                  stats['n'] >= original_stats['n'] * 0.8)
        marker = ' <-- 最优' if better else ''
        print(f"{name:<30s} {stats['n']:4d} {stats['win_rate']:5.1f}% "
              f"{stats['avg_return']:+6.2f}% {stats['avg_max_return']:+6.2f}% "
              f"{stats['sharpe']:7.2f} {stats['worst']:+6.2f}% {stats['avg_loss']:+6.2f}%{marker}")

    # HTML
    all_results = [('原版 support_rebound', original_stats)] + variants
    rows = ""
    for name, stats in all_results:
        wr_color = '#27ae60' if stats['win_rate'] >= 80 else '#e67e22'
        ret_color = '#27ae60' if stats['avg_return'] >= 0 else '#e74c3c'
        sharpe_color = '#27ae60' if stats['sharpe'] >= 0.3 else '#e67e22'
        rows += f"<tr><td><strong>{name}</strong></td><td>{stats['n']}</td><td style='color:{wr_color};font-weight:bold;'>{stats['win_rate']:.1f}%</td><td style='color:{ret_color};'>{stats['avg_return']:+.2f}%</td><td>{stats['avg_max_return']:+.2f}%</td><td style='color:{sharpe_color};font-weight:bold;'>{stats['sharpe']:.2f}</td><td>{stats['worst']:+.2f}%</td><td>{stats['avg_loss']:+.2f}%</td><td>{stats['best']:+.2f}%</td></tr>"

    best_variant = None
    for name, stats in variants:
        if (stats['win_rate'] >= original_stats['win_rate'] and
            stats['sharpe'] > original_stats['sharpe'] and
            stats['n'] >= original_stats['n'] * 0.8):
            best_variant = (name, stats)
            break

    if best_variant:
        conclusion = f"推荐 {best_variant[0]}: 胜率{best_variant[1]['win_rate']:.1f}% Sharpe{best_variant[1]['sharpe']:.2f} 信号{best_variant[1]['n']}个, 优于原版"
        conclusion_color = '#27ae60'
    else:
        conclusion = "优化变体未全面优于原版, 建议直接使用原版 support_rebound"
        conclusion_color = '#e67e22'

    # 信号明细
    analysis_rows = ""
    for s in original_signals:
        is_win = s['is_win']
        win_class = 'win' if is_win else 'loss'
        ind = s.get('indicators', {})
        ind_str = (f"RSI:{ind.get('rsi_14',0):.0f} Sup:{ind.get('dev_support',0):.1f}% "
                   f"MACD:{ind.get('macd_hist',0):.4f} BB%B:{ind.get('bb_percent_b',0):.2f} "
                   f"KDJ_J:{ind.get('kdj_j',0):.0f} Vol:{ind.get('vol_ratio',0):.1f} "
                   f"M20:{ind.get('momentum_20',0):+.1f}%")
        ret_color = '#27ae60' if s['close_return'] >= 0 else '#e74c3c'
        maxret_color = '#27ae60' if s['max_return'] >= 0 else '#e74c3c'
        marker = ' *** LOSS' if not is_win else (' * marginal' if s['max_return'] < 1.5 else '')
        analysis_rows += f"<tr class='{win_class}'><td>{s['date']}</td><td style='font-weight:bold;'>{s['score']:.1f}</td><td class='win-cell'>{'WIN' if is_win else 'LOSS'}</td><td style='color:{maxret_color};font-weight:bold;'>{s['max_return']:+.2f}%{marker}</td><td style='color:{ret_color};'>{s['close_return']:+.2f}%</td><td style='font-size:11px;'>{ind_str}</td><td style='font-size:11px;'>{'; '.join(s.get('reasons',[]))}</td></tr>"

    # 变体信号明细
    variant_detail = ""
    for name, stats in variants:
        for s in stats.get('signal_list', []):
            is_win = s['is_win']
            win_class = 'win' if is_win else 'loss'
            ret_color = '#27ae60' if s['close_return'] >= 0 else '#e74c3c'
            maxret_color = '#27ae60' if s['max_return'] >= 0 else '#e74c3c'
            short_name = name.split('(')[0].strip()
            variant_detail += f"<tr class='{win_class}'><td>{short_name}</td><td>{s['date']}</td><td style='font-weight:bold;'>{s['score']:.1f}</td><td class='win-cell'>{'WIN' if is_win else 'LOSS'}</td><td style='color:{maxret_color};font-weight:bold;'>{s['max_return']:+.2f}%</td><td style='color:{ret_color};'>{s['close_return']:+.2f}%</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>黄金ETF - support_rebound 优化分析</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif; background:#f5f6fa; color:#2c3e50; padding:20px; line-height:1.6; }}
.header {{ background:linear-gradient(135deg,#1a1a2e,#16213e); color:white; padding:25px; border-radius:12px; margin-bottom:20px; }}
.header h1 {{ font-size:22px; margin-bottom:8px; }} .header .meta {{ opacity:0.8; font-size:13px; }}
.conclusion {{ background:{conclusion_color}; color:white; padding:15px 20px; border-radius:10px; margin-bottom:20px; font-size:15px; }}
.section {{ background:white; border-radius:12px; padding:20px; margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,0.08); overflow-x:auto; }}
.section h2 {{ font-size:17px; margin-bottom:15px; color:#2c3e50; border-bottom:2px solid #ecf0f1; padding-bottom:10px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ background:#f8f9fa; padding:10px; text-align:left; border-bottom:2px solid #dee2e6; white-space:nowrap; }}
td {{ padding:8px 10px; border-bottom:1px solid #ecf0f1; }}
tr.win {{ background:#f0fff4; }} tr.loss {{ background:#fff5f5; }}
.win-cell {{ font-weight:bold; text-align:center; }}
tr.win .win-cell {{ color:#27ae60; }} tr.loss .win-cell {{ color:#e74c3c; }}
.footer {{ text-align:center; padding:20px; color:#95a5a6; font-size:12px; }}
</style></head><body>
<div class="header"><h1>黄金ETF - support_rebound 优化分析</h1>
<div class="meta">ETF: {ETF_CODE} | 回测: {START_DATE}~最新 | T+3 | 阈值&ge;{THRESHOLD}</div></div>
<div class="conclusion">{conclusion}</div>
<div class="section"><h2>原版信号深度分析</h2>
<table><tr><th>日期</th><th>信号分</th><th>结果</th><th>最大收益</th><th>收盘收益</th><th>关键指标</th><th>触发理由</th></tr>
{analysis_rows}</table></div>
<div class="section"><h2>原版 vs 优化变体对比</h2>
<table><tr><th>算法</th><th>信号数</th><th>胜率</th><th>均收益</th><th>均最大</th><th>Sharpe</th><th>最差</th><th>亏损均</th><th>最佳</th></tr>
{rows}</table></div>
<div class="section"><h2>优化变体信号明细</h2>
<table><tr><th>算法</th><th>日期</th><th>信号分</th><th>结果</th><th>最大收益</th><th>收盘收益</th></tr>
{variant_detail}</table></div>
<div class="footer"><p>黄金ETF support_rebound 优化分析 | T+3胜率回测</p></div>
</body></html>"""
    html_path = os.path.join(OUTPUT_DIR, '黄金ETF_support_rebound优化.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML: {html_path}")

    json_data = {
        'original': {k: v for k, v in original_stats.items() if k != 'signal_list'},
        'variants': {name: {k: v for k, v in stats.items() if k != 'signal_list'} for name, stats in variants},
    }
    json_path = os.path.join(OUTPUT_DIR, '黄金ETF_support_rebound优化.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"JSON: {json_path}")


if __name__ == '__main__':
    main()
