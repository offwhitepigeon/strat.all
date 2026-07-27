# -*- coding: utf-8 -*-
"""
有色金属ETF - volatility_breakout 信号深度分析 + 优化变体对比
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
ETF_CODE = 'sh512400'
ETF_NAME = '有色金属ETF'
START_DATE = '2024-01-01'
THRESHOLD = 60


# ======================================================================
# 优化变体1: 有色增强型 - 加入趋势过滤+KDJ
# ======================================================================
class VolBreakout有色增强(BaseAlgorithm):
    """
    volatility_breakout 有色金属增强版

    优化点:
    1. F1/F2降权(各25→20), 腾出15%给趋势因子
    2. F5从Z-score改为KDJ(更适合A股商品)
    3. 新增F6: 趋势上下文 - 在MA60附近企稳加分, 远离MA60>15%减分
    4. F3 RSI阈值微调: <30给20分(原15), <35给12分(原8) - 商品超卖更常见
    """
    name = "vol_breakout_ys_enhanced"

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
        kdj_j = float(last.get('kdj_j', 50))
        kdj_k = float(last.get('kdj_k', 50))
        dev_ma60 = float(last.get('dev_ma60', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        momentum_20 = float(last.get('momentum_20', 0))
        consec_down = int(last.get('consec_down', 0))

        # 计算ATR波动率的历史分位
        if 'atr_pct' in df.columns and len(df) >= 60:
            atr_series = df['atr_pct'].iloc[-60:]
            atr_rank = float((atr_series < atr_pct).sum() / len(atr_series) * 100)
        else:
            atr_rank = 50

        if 'bb_bandwidth' in df.columns and len(df) >= 60:
            bw_series = df['bb_bandwidth'].iloc[-60:]
            bw_rank = float((bw_series < bb_bw).sum() / len(bw_series) * 100)
        else:
            bw_rank = 50

        indicators.update({
            'rsi_14': rsi, 'rsi_7': rsi7, 'bb_percent_b': bb_pctb,
            'atr_pct': atr_pct, 'atr_rank': atr_rank, 'bw_rank': bw_rank,
            'vol_ratio': vol_ratio, 'kdj_j': kdj_j, 'dev_ma60': dev_ma60,
        })

        # F1: 波动率收缩 (20%)
        if atr_rank < 20 and bw_rank < 30:
            s1 = 20
            reasons.append("波动率+布林带宽度收缩,临近变盘")
        elif atr_rank < 30:
            s1 = 12
            reasons.append("波动率处于低位")
        else:
            s1 = 0
        score += s1

        # F2: 布林带下轨 (20%)
        if bb_pctb < 0.05:
            s2 = 20
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s2 = 12
        else:
            s2 = 0
        score += s2

        # F3: RSI超卖 (20%) - 微调: <30给满分
        if rsi < 25:
            s3 = 20
            reasons.append(f"RSI={rsi:.0f},极度超卖")
        elif rsi < 30:
            s3 = 16
            reasons.append(f"RSI={rsi:.0f},超卖")
        elif rsi < 35:
            s3 = 8
        else:
            s3 = 0
        score += s3

        # F4: 成交量放量 (15%)
        if vol_ratio > 1.8:
            s4 = 15
            reasons.append(f"量比={vol_ratio:.1f},放量信号")
        elif vol_ratio > 1.3:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: KDJ超卖 (10%) - 替换Z-score
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

        # F6: 趋势上下文 (15%) - 新增
        if dev_ma60 > -3 and momentum_20 > 0:
            s6 = 15
            reasons.append("趋势健康,回调买入")
        elif dev_ma60 > -8:
            s6 = 10
            reasons.append("中度回调")
        elif dev_ma60 > -15:
            s6 = 5
        else:
            s6 = 0  # 远离MA60,不加不减(让其他因子决定)
        score += s6

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 优化变体2: 有色宽松型 - 放宽阈值增加信号
# ======================================================================
class VolBreakout有色宽松(BaseAlgorithm):
    """
    volatility_breakout 有色金属宽松版

    优化点:
    1. F1: ATR rank < 25(原20), BB bw rank < 35(原30) - 更容易触发
    2. F2: bb_pctb < 0.10(原0.05)给满分, < 0.20(原0.15)给部分分
    3. F3: RSI < 30给20分(原15), < 35给12分(原8)
    4. F4: vol_ratio > 1.5(原1.8)给满分
    5. F5: Z-score < -1.2(原-1.5)给满分
    """
    name = "vol_breakout_ys_loose"

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

        if 'atr_pct' in df.columns and len(df) >= 60:
            atr_series = df['atr_pct'].iloc[-60:]
            atr_rank = float((atr_series < atr_pct).sum() / len(atr_series) * 100)
        else:
            atr_rank = 50

        if 'bb_bandwidth' in df.columns and len(df) >= 60:
            bw_series = df['bb_bandwidth'].iloc[-60:]
            bw_rank = float((bw_series < bb_bw).sum() / len(bw_series) * 100)
        else:
            bw_rank = 50

        indicators.update({
            'rsi_14': rsi, 'rsi_7': rsi7, 'bb_percent_b': bb_pctb,
            'atr_pct': atr_pct, 'atr_rank': atr_rank, 'bw_rank': bw_rank,
            'vol_ratio': vol_ratio, 'zscore_20': zscore,
        })

        # F1: 波动率收缩 (25%) - 放宽
        if atr_rank < 25 and bw_rank < 35:
            s1 = 25
            reasons.append("波动率+布林带宽度收缩,临近变盘")
        elif atr_rank < 35:
            s1 = 15
            reasons.append("波动率处于低位")
        else:
            s1 = 0
        score += s1

        # F2: 布林带下轨 (25%) - 放宽
        if bb_pctb < 0.10:
            s2 = 25
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.20:
            s2 = 15
        else:
            s2 = 0
        score += s2

        # F3: RSI超卖 (20%) - 放宽
        if rsi < 25:
            s3 = 20
            reasons.append(f"RSI={rsi:.0f},极度超卖")
        elif rsi < 30:
            s3 = 15
            reasons.append(f"RSI={rsi:.0f},超卖")
        elif rsi < 35:
            s3 = 12
            reasons.append(f"RSI={rsi:.0f},偏超卖")
        else:
            s3 = 0
        score += s3

        # F4: 成交量放量 (15%) - 放宽
        if vol_ratio > 1.5:
            s4 = 15
            reasons.append(f"量比={vol_ratio:.1f},放量信号")
        elif vol_ratio > 1.2:
            s4 = 8
        else:
            s4 = 0
        score += s4

        # F5: Z-score偏低 (15%) - 放宽
        if zscore < -1.2:
            s5 = 15
            reasons.append(f"Z-score={zscore:.1f}")
        elif zscore < -0.8:
            s5 = 8
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 优化变体3: 有色混合型 - 增强趋势+宽松阈值
# ======================================================================
class VolBreakout有色混合(BaseAlgorithm):
    """
    volatility_breakout 有色金属混合版

    结合增强版和宽松版的优点:
    1. 适中放宽F1/F2阈值
    2. RSI < 30给满分(增强)
    3. KDJ替换Z-score(增强)
    4. 新增趋势因子(增强)
    5. 重新分配权重: F1=20, F2=20, F3=20, F4=10, F5=10, F6=20
    """
    name = "vol_breakout_ys_mixed"

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
        kdj_j = float(last.get('kdj_j', 50))
        kdj_k = float(last.get('kdj_k', 50))
        dev_ma60 = float(last.get('dev_ma60', 0))
        momentum_20 = float(last.get('momentum_20', 0))
        momentum_10 = float(last.get('momentum_10', 0))
        consec_down = int(last.get('consec_down', 0))

        if 'atr_pct' in df.columns and len(df) >= 60:
            atr_series = df['atr_pct'].iloc[-60:]
            atr_rank = float((atr_series < atr_pct).sum() / len(atr_series) * 100)
        else:
            atr_rank = 50

        if 'bb_bandwidth' in df.columns and len(df) >= 60:
            bw_series = df['bb_bandwidth'].iloc[-60:]
            bw_rank = float((bw_series < bb_bw).sum() / len(bw_series) * 100)
        else:
            bw_rank = 50

        indicators.update({
            'rsi_14': rsi, 'rsi_7': rsi7, 'bb_percent_b': bb_pctb,
            'atr_pct': atr_pct, 'atr_rank': atr_rank, 'bw_rank': bw_rank,
            'vol_ratio': vol_ratio, 'kdj_j': kdj_j, 'dev_ma60': dev_ma60,
        })

        # F1: 波动率收缩 (20%) - 适中放宽
        if atr_rank < 25 and bw_rank < 35:
            s1 = 20
            reasons.append("波动率收缩,临近变盘")
        elif atr_rank < 35:
            s1 = 12
            reasons.append("波动率低位")
        else:
            s1 = 0
        score += s1

        # F2: 布林带下轨 (20%) - 适中放宽
        if bb_pctb < 0.08:
            s2 = 20
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.18:
            s2 = 12
        else:
            s2 = 0
        score += s2

        # F3: RSI超卖 (20%) - 增强
        if rsi < 25:
            s3 = 20
            reasons.append(f"RSI={rsi:.0f},极度超卖")
        elif rsi < 30:
            s3 = 16
            reasons.append(f"RSI={rsi:.0f},超卖")
        elif rsi < 35:
            s3 = 8
        else:
            s3 = 0
        score += s3

        # F4: 成交量放量 (10%) - 降权
        if vol_ratio > 1.5:
            s4 = 10
            reasons.append(f"量比={vol_ratio:.1f}")
        elif vol_ratio > 1.2:
            s4 = 5
        else:
            s4 = 0
        score += s4

        # F5: KDJ超卖 (10%) - 替换Z-score
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

        # F6: 趋势上下文 (20%) - 新增,权重更大
        if dev_ma60 > -3 and momentum_20 > 0:
            s6 = 20
            reasons.append("趋势健康回调")
        elif dev_ma60 > -8:
            s6 = 12
            reasons.append("中度回调")
        elif dev_ma60 > -15:
            s6 = 6
        else:
            s6 = 0
        score += s6

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 回测引擎
# ======================================================================
def run_backtest(df, algo, threshold=THRESHOLD, hold_days=3):
    """运行单个算法回测"""
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

        # 提取关键指标用于分析
        last = df.iloc[i]
        indicators_data = {
            'rsi_14': float(last.get('rsi_14', 50)),
            'bb_percent_b': float(last.get('bb_percent_b', 0.5)),
            'atr_rank': 0,
            'vol_ratio': float(last.get('vol_ratio_20', 1)),
            'zscore_20': float(last.get('zscore_20', 0)),
            'kdj_j': float(last.get('kdj_j', 50)),
            'dev_ma60': float(last.get('dev_ma60', 0)),
            'momentum_20': float(last.get('momentum_20', 0)),
            'consec_down': int(last.get('consec_down', 0)),
        }

        # 计算atr_rank
        if 'atr_pct' in df.columns and len(df) >= 60:
            atr_series = df['atr_pct'].iloc[max(0,i-59):i+1]
            indicators_data['atr_rank'] = float((atr_series < float(last.get('atr_pct', 0))).sum() / len(atr_series) * 100)

        signals.append({
            'date': date_str,
            'score': round(signal.score, 1),
            'buy_price': round(buy_price, 4),
            'max_return': round(max_ret, 2),
            'close_return': round(close_ret, 2),
            'is_win': is_win,
            'days_to_win': days_to_win,
            'reasons': signal.reasons[:3],
            'indicators': indicators_data,
        })

    return signals


def compute_stats(signals):
    n = len(signals)
    if n == 0:
        return {'n': 0, 'win_rate': 0, 'avg_return': 0, 'avg_max_return': 0,
                'sharpe': 0, 'worst': 0, 'avg_loss': 0}

    rets = [s['close_return'] for s in signals]
    max_rets = [s['max_return'] for s in signals]
    wins = [s for s in signals if s['is_win']]
    losses = [s for s in signals if not s['is_win']]
    loss_rets = [s['close_return'] for s in losses] if losses else []

    return {
        'n': n,
        'wins': len(wins),
        'losses': len(losses),
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


def generate_html(original_stats, variants, original_signals):
    """生成对比HTML"""

    # 原始信号分析表
    analysis_rows = ""
    for s in original_signals:
        is_win = s['is_win']
        win_class = 'win' if is_win else 'loss'
        ind = s.get('indicators', {})
        ind_str = (f"RSI:{ind.get('rsi_14',0):.0f} "
                   f"BB%B:{ind.get('bb_percent_b',0):.2f} "
                   f"ATR_rank:{ind.get('atr_rank',0):.0f} "
                   f"Vol:{ind.get('vol_ratio',0):.1f} "
                   f"Z:{ind.get('zscore_20',0):.1f} "
                   f"KDJ_J:{ind.get('kdj_j',0):.0f} "
                   f"MA60:{ind.get('dev_ma60',0):+.1f}% "
                   f"M20:{ind.get('momentum_20',0):+.1f}%")

        ret_color = '#27ae60' if s['close_return'] >= 0 else '#e74c3c'
        maxret_color = '#27ae60' if s['max_return'] >= 0 else '#e74c3c'
        marker = ''
        if not is_win:
            marker = ' *** LOSS'
        elif s['max_return'] < 1:
            marker = ' * marginal'

        analysis_rows += f"""
        <tr class="{win_class}">
            <td>{s['date']}</td>
            <td style="font-weight:bold;">{s['score']:.1f}</td>
            <td>{s['buy_price']:.3f}</td>
            <td class="win-cell">{'WIN' if is_win else 'LOSS'}</td>
            <td style="color:{maxret_color};font-weight:bold;">{s['max_return']:+.2f}%{marker}</td>
            <td style="color:{ret_color};">{s['close_return']:+.2f}%</td>
            <td style="font-size:11px;">{ind_str}</td>
            <td style="font-size:11px;">{'; '.join(s.get('reasons',[]))}</td>
        </tr>"""

    # 变体对比
    all_results = [('原版 volatility_breakout', original_stats)] + variants
    variant_rows = ""
    for name, stats in all_results:
        wr_color = '#27ae60' if stats['win_rate'] >= 90 else '#e67e22'
        ret_color = '#27ae60' if stats['avg_return'] >= 0 else '#e74c3c'
        sharpe_color = '#27ae60' if stats['sharpe'] >= 0.7 else '#e67e22'
        variant_rows += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{stats['n']}</td>
            <td style="color:{wr_color};font-weight:bold;">{stats['win_rate']:.1f}%</td>
            <td style="color:{ret_color};">{stats['avg_return']:+.2f}%</td>
            <td>{stats['avg_max_return']:+.2f}%</td>
            <td style="color:{sharpe_color};font-weight:bold;">{stats['sharpe']:.2f}</td>
            <td>{stats['worst']:+.2f}%</td>
            <td>{stats['avg_loss']:+.2f}%</td>
            <td>{stats['best']:+.2f}%</td>
        </tr>"""

    # 变体信号明细
    variant_detail_rows = ""
    for name, stats in variants:
        for s in stats.get('signal_list', []):
            is_win = s['is_win']
            win_class = 'win' if is_win else 'loss'
            ret_color = '#27ae60' if s['close_return'] >= 0 else '#e74c3c'
            maxret_color = '#27ae60' if s['max_return'] >= 0 else '#e74c3c'
            variant_detail_rows += f"""
            <tr class="{win_class}">
                <td>{name.split(' ')[0] if ' ' in name else name}</td>
                <td>{s['date']}</td>
                <td style="font-weight:bold;">{s['score']:.1f}</td>
                <td class="win-cell">{'WIN' if is_win else 'LOSS'}</td>
                <td style="color:{maxret_color};font-weight:bold;">{s['max_return']:+.2f}%</td>
                <td style="color:{ret_color};">{s['close_return']:+.2f}%</td>
            </tr>"""

    # 结论
    best_variant = None
    for name, stats in variants:
        if (stats['win_rate'] >= original_stats['win_rate'] and
            stats['sharpe'] > original_stats['sharpe'] and
            stats['n'] >= original_stats['n'] * 0.8):
            best_variant = (name, stats)
            break

    if best_variant:
        conclusion = f"推荐使用 {best_variant[0]}: 胜率{best_variant[1]['win_rate']:.1f}% Sharpe{best_variant[1]['sharpe']:.2f} 信号{best_variant[1]['n']}个, 全面优于原版"
        conclusion_color = '#27ae60'
    else:
        conclusion = "优化变体未全面优于原版, 建议直接使用原版 volatility_breakout"
        conclusion_color = '#e67e22'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>有色金属ETF - volatility_breakout 优化分析</title>
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
        th {{ background:#f8f9fa; padding:10px; text-align:left; border-bottom:2px solid #dee2e6; white-space:nowrap; }}
        td {{ padding:8px 10px; border-bottom:1px solid #ecf0f1; }}
        tr.win {{ background:#f0fff4; }}
        tr.loss {{ background:#fff5f5; }}
        .win-cell {{ font-weight:bold; text-align:center; }}
        tr.win .win-cell {{ color:#27ae60; }}
        tr.loss .win-cell {{ color:#e74c3c; }}
        .footer {{ text-align:center; padding:20px; color:#95a5a6; font-size:12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>有色金属ETF - volatility_breakout 优化分析</h1>
        <div class="meta">ETF: {ETF_CODE} | 回测: {START_DATE}~最新 | T+3 | 阈值&ge;{THRESHOLD}</div>
    </div>

    <div class="conclusion">{conclusion}</div>

    <div class="section">
        <h2>原版信号深度分析 (*** LOSS = 亏损信号, * marginal = 边缘盈利)</h2>
        <table>
            <tr><th>日期</th><th>信号分</th><th>买入价</th><th>结果</th><th>最大收益</th><th>收盘收益</th><th>关键指标</th><th>触发理由</th></tr>
            {analysis_rows}
        </table>
    </div>

    <div class="section">
        <h2>原版 vs 优化变体对比</h2>
        <table>
            <tr><th>算法</th><th>信号数</th><th>胜率</th><th>均收益</th><th>均最大</th><th>Sharpe</th><th>最差</th><th>亏损均</th><th>最佳</th></tr>
            {variant_rows}
        </table>
    </div>

    <div class="section">
        <h2>优化变体信号明细</h2>
        <table>
            <tr><th>算法</th><th>日期</th><th>信号分</th><th>结果</th><th>最大收益</th><th>收盘收益</th></tr>
            {variant_detail_rows}
        </table>
    </div>

    <div class="footer"><p>有色金属ETF volatility_breakout 优化分析 | T+3胜率回测</p></div>
</body>
</html>"""
    return html


def main():
    logger.info("加载数据...")
    engine = DataEngine()
    df = engine.get_history_kline(ETF_CODE)
    df = calc_all_indicators(df)
    logger.info(f"数据: {len(df)}条")

    # 1. 原版
    logger.info("运行原版 volatility_breakout...")
    original_algo = ALGORITHM_MAP['volatility_breakout']()
    original_signals = run_backtest(df, original_algo)
    original_stats = compute_stats(original_signals)
    logger.info(f"  原版: {original_stats['n']}信号, {original_stats['win_rate']}%, Sharpe={original_stats['sharpe']}")

    # 分析亏损和边缘信号
    print("\n" + "="*100)
    print("原版信号分析 (亏损 + 边缘盈利)")
    print("="*100)
    for s in original_signals:
        if not s['is_win'] or s['max_return'] < 1:
            ind = s.get('indicators', {})
            tag = "LOSS" if not s['is_win'] else "MARGINAL"
            print(f"  [{tag}] {s['date']} score={s['score']:.1f} max={s['max_return']:+.2f}% close={s['close_return']:+.2f}%")
            print(f"        RSI={ind.get('rsi_14',0):.1f} BB%B={ind.get('bb_percent_b',0):.2f} ATR_rank={ind.get('atr_rank',0):.0f} "
                  f"Vol={ind.get('vol_ratio',0):.1f} Z={ind.get('zscore_20',0):.1f} "
                  f"KDJ_J={ind.get('kdj_j',0):.0f} MA60={ind.get('dev_ma60',0):+.1f}% M20={ind.get('momentum_20',0):+.1f}%")

    # 2. 优化变体
    variants = []
    for algo_cls, label in [
        (VolBreakout有色增强, '增强版(趋势+KDJ)'),
        (VolBreakout有色宽松, '宽松版(放宽阈值)'),
        (VolBreakout有色混合, '混合版(趋势+宽松+KDJ)'),
    ]:
        algo = algo_cls()
        logger.info(f"运行 {label}...")
        sigs = run_backtest(df, algo)
        stats = compute_stats(sigs)
        logger.info(f"  {label}: {stats['n']}信号, {stats['win_rate']}%, Sharpe={stats['sharpe']}")
        variants.append((label, stats))

    # 对比输出
    print(f"\n{'='*90}")
    print(f"{'算法':<30s} {'信号':>4s} {'胜率':>6s} {'均收益':>7s} {'均最大':>7s} {'Sharpe':>7s} {'最差':>7s} {'亏损均':>7s}")
    print("-"*90)
    print(f"{'原版 vol_breakout':<30s} {original_stats['n']:4d} {original_stats['win_rate']:5.1f}% "
          f"{original_stats['avg_return']:+6.2f}% {original_stats['avg_max_return']:+6.2f}% "
          f"{original_stats['sharpe']:7.2f} {original_stats['worst']:+6.2f}% {original_stats['avg_loss']:+6.2f}%")
    for name, stats in variants:
        marker = ' <-- 最优' if (stats['win_rate'] >= original_stats['win_rate'] and
                                stats['sharpe'] > original_stats['sharpe'] and
                                stats['n'] >= original_stats['n'] * 0.8) else ''
        print(f"{name:<30s} {stats['n']:4d} {stats['win_rate']:5.1f}% "
              f"{stats['avg_return']:+6.2f}% {stats['avg_max_return']:+6.2f}% "
              f"{stats['sharpe']:7.2f} {stats['worst']:+6.2f}% {stats['avg_loss']:+6.2f}%{marker}")

    # 保存
    html = generate_html(original_stats, variants, original_signals)
    html_path = os.path.join(OUTPUT_DIR, '有色ETF_vol_breakout优化.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML: {html_path}")

    # JSON
    json_data = {
        'original': {k: v for k, v in original_stats.items() if k != 'signal_list'},
        'variants': {name: {k: v for k, v in stats.items() if k != 'signal_list'} for name, stats in variants},
    }
    json_path = os.path.join(OUTPUT_DIR, '有色ETF_vol_breakout优化.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"JSON: {json_path}")


if __name__ == '__main__':
    main()
