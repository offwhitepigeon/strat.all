# -*- coding: utf-8 -*-
"""
标普生物科技ETF - trend_pullback 信号深度分析 + 优化变体对比
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
ETF_CODE = 'sz159502'
ETF_NAME = '标普生物科技ETF'
START_DATE = '2024-01-01'
THRESHOLD = 60


# ======================================================================
# 变体1: 增强型 - 加宽回踩+KDJ
# ======================================================================
class TrendPullback增强(BaseAlgorithm):
    """
    trend_pullback 生物科技增强版
    - F2回踩范围放宽: -4%~+2%(原-3%~+1%), 适配生物科技高波动
    - F5从布林带改为KDJ(A股/跨境ETF更有效)
    - F3 RSI微调: <25给满分(原<30), 适配高波动品种
    """
    name = "trend_pullback_enhanced"

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
        kdj_j = float(last.get('kdj_j', 50))
        kdj_k = float(last.get('kdj_k', 50))
        vol_ratio = float(last.get('vol_ratio_20', 1))

        indicators.update({
            'rsi_14': rsi, 'macd_hist': macd_hist,
            'dev_ma20': dev_ma20, 'ma200': ma200,
            'momentum_20': momentum_20, 'kdj_j': kdj_j,
        })

        # 前提: MA200趋势保护
        if ma200 > 0 and price < ma200 * 0.95:
            return self._build_result(0, ["跌破MA200 5%,趋势破坏"], indicators)

        # F1: 上升趋势确认 (20%)
        if ma200 > 0 and price > ma200 and momentum_20 > 0:
            s1 = 20
            reasons.append("上升趋势中(MA200之上)")
        elif ma200 > 0 and price > ma200:
            s1 = 12
            reasons.append("价格在MA200上方")
        else:
            s1 = 5
        score += s1

        # F2: 回踩MA20 (30%) - 放宽范围
        if ma20 > 0:
            dev_pct = (price - ma20) / ma20 * 100
            if -4 <= dev_pct <= 2:
                s2 = 30
                reasons.append(f"精确回踩MA20(偏离{dev_pct:+.1f}%)")
            elif -6 <= dev_pct < -4:
                s2 = 25
                reasons.append(f"回踩MA20附近(偏离{dev_pct:+.1f}%)")
            elif -10 <= dev_pct < -6:
                s2 = 15
                reasons.append("略低于MA20,超卖回踩")
            elif 2 < dev_pct <= 4:
                s2 = 10
            else:
                s2 = 0
        else:
            s2 = 0
        score += s2

        # F3: RSI低位 (20%) - 适配高波动
        if rsi < 25:
            s3 = 20
            reasons.append(f"RSI={rsi:.0f},极度超卖")
        elif rsi < 30:
            s3 = 16
            reasons.append(f"RSI={rsi:.0f},超卖")
        elif rsi < 35:
            s3 = 12
        elif rsi < 40:
            s3 = 6
        else:
            s3 = 0
        score += s3

        # F4: MACD (15%)
        if len(df) >= 3:
            prev_hist = float(df['macd_hist'].iloc[-2])
            if macd_hist > 0 and prev_hist < 0:
                s4 = 15
                reasons.append("MACD翻红,反弹信号")
            elif macd_hist > prev_hist and macd_hist < 0:
                s4 = 10
                reasons.append("MACD柱线收窄")
            elif macd_hist > 0:
                s4 = 5
            else:
                s4 = 0
        else:
            s4 = 0
        score += s4

        # F5: KDJ超卖 (15%) - 替换布林带
        if kdj_j < 0 and kdj_k < 20:
            s5 = 15
            reasons.append(f"KDJ超卖(J={kdj_j:.0f})")
        elif kdj_j < 10:
            s5 = 10
        elif kdj_j < 20:
            s5 = 5
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 变体2: 量价型 - 加入放量因子
# ======================================================================
class TrendPullback量价(BaseAlgorithm):
    """
    trend_pullback 生物科技量价版
    - 新增F6: 放量回踩(生物科技底部放量信号强)
    - F1/F2降权腾出空间
    - F5改为KDJ+量比组合
    """
    name = "trend_pullback_volprice"

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
        kdj_j = float(last.get('kdj_j', 50))
        kdj_k = float(last.get('kdj_k', 50))
        vol_ratio = float(last.get('vol_ratio_20', 1))

        indicators.update({
            'rsi_14': rsi, 'macd_hist': macd_hist,
            'dev_ma20': dev_ma20, 'ma200': ma200,
            'momentum_20': momentum_20, 'kdj_j': kdj_j,
            'vol_ratio': vol_ratio,
        })

        if ma200 > 0 and price < ma200 * 0.95:
            return self._build_result(0, ["跌破MA200,趋势破坏"], indicators)

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

        # F2: 回踩MA20 (25%) - 适中放宽
        if ma20 > 0:
            dev_pct = (price - ma20) / ma20 * 100
            if -4 <= dev_pct <= 2:
                s2 = 25
                reasons.append(f"回踩MA20(偏离{dev_pct:+.1f}%)")
            elif -6 <= dev_pct < -4:
                s2 = 18
            elif -10 <= dev_pct < -6:
                s2 = 10
            elif 2 < dev_pct <= 4:
                s2 = 8
            else:
                s2 = 0
        else:
            s2 = 0
        score += s2

        # F3: RSI (20%)
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

        # F4: MACD (15%)
        if len(df) >= 3:
            prev_hist = float(df['macd_hist'].iloc[-2])
            if macd_hist > 0 and prev_hist < 0:
                s4 = 15
                reasons.append("MACD翻红")
            elif macd_hist > prev_hist and macd_hist < 0:
                s4 = 10
                reasons.append("MACD收窄")
            elif macd_hist > 0:
                s4 = 5
            else:
                s4 = 0
        else:
            s4 = 0
        score += s4

        # F5: KDJ (10%)
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

        # F6: 放量回踩 (15%) - 新增
        if vol_ratio > 1.5 and dev_ma20 < 0:
            s6 = 15
            reasons.append(f"放量回踩(量比{vol_ratio:.1f})")
        elif vol_ratio > 1.2:
            s6 = 8
        else:
            s6 = 0
        score += s6

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 变体3: 混合型 - 宽松+KDJ+量价
# ======================================================================
class TrendPullback混合(BaseAlgorithm):
    """
    trend_pullback 生物科技混合版
    - 放宽F2回踩范围 + KDJ + 放量 + RSI微调
    - 权重: F1=15, F2=25, F3=20, F4=10, F5=10, F6=20
    """
    name = "trend_pullback_mixed"

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
        bb_pctb = float(last.get('bb_percent_b', 0.5))

        indicators.update({
            'rsi_14': rsi, 'dev_ma20': dev_ma20, 'ma200': ma200,
            'momentum_20': momentum_20, 'kdj_j': kdj_j, 'vol_ratio': vol_ratio,
        })

        if ma200 > 0 and price < ma200 * 0.93:
            return self._build_result(0, ["跌破MA200 7%,趋势破坏"], indicators)

        # F1: 趋势 (15%)
        if ma200 > 0 and price > ma200 and momentum_20 > 0:
            s1 = 15
            reasons.append("上升趋势中")
        elif ma200 > 0 and price > ma200:
            s1 = 8
        else:
            s1 = 3
        score += s1

        # F2: 回踩MA20 (25%) - 放宽
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

        # F3: RSI (20%)
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
            elif macd_hist > 0:
                s4 = 3
            else:
                s4 = 0
        else:
            s4 = 0
        score += s4

        # F5: KDJ (10%)
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

        # F6: 放量 (20%) - 新增,权重大
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


# ======================================================================
# 回测引擎
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
            'dev_ma20': float(last.get('dev_ma20', 0)),
            'ma200': float(last.get('ma200', 0)),
            'momentum_20': float(last.get('momentum_20', 0)),
            'bb_percent_b': float(last.get('bb_percent_b', 0.5)),
            'kdj_j': float(last.get('kdj_j', 50)),
            'vol_ratio': float(last.get('vol_ratio_20', 1)),
            'close_price': float(last.get('close', 0)),
        }

        signals.append({
            'date': date_str,
            'score': round(signal.score, 1),
            'buy_price': round(buy_price, 4),
            'max_return': round(max_ret, 2),
            'close_return': round(close_ret, 2),
            'is_win': is_win,
            'days_to_win': days_to_win,
            'reasons': signal.reasons[:3],
            'indicators': ind_data,
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


def generate_html(original_stats, variants, original_signals):
    analysis_rows = ""
    for s in original_signals:
        is_win = s['is_win']
        win_class = 'win' if is_win else 'loss'
        ind = s.get('indicators', {})
        ind_str = (f"RSI:{ind.get('rsi_14',0):.0f} "
                   f"MA20:{ind.get('dev_ma20',0):+.1f}% "
                   f"MACD:{ind.get('macd_hist',0):.3f} "
                   f"M20:{ind.get('momentum_20',0):+.1f}% "
                   f"BB%B:{ind.get('bb_percent_b',0):.2f} "
                   f"KDJ_J:{ind.get('kdj_j',0):.0f} "
                   f"Vol:{ind.get('vol_ratio',0):.1f} "
                   f"vsMA200:{((ind.get('close_price',0)/max(ind.get('ma200',1),0.001)-1)*100):+.1f}%")
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
            <td class="win-cell">{'WIN' if is_win else 'LOSS'}</td>
            <td style="color:{maxret_color};font-weight:bold;">{s['max_return']:+.2f}%{marker}</td>
            <td style="color:{ret_color};">{s['close_return']:+.2f}%</td>
            <td style="font-size:11px;">{ind_str}</td>
            <td style="font-size:11px;">{'; '.join(s.get('reasons',[]))}</td>
        </tr>"""

    all_results = [('原版 trend_pullback', original_stats)] + variants
    variant_rows = ""
    for name, stats in all_results:
        wr_color = '#27ae60' if stats['win_rate'] >= 80 else ('#e67e22' if stats['win_rate'] >= 70 else '#e74c3c')
        ret_color = '#27ae60' if stats['avg_return'] >= 0 else '#e74c3c'
        sharpe_color = '#27ae60' if stats['sharpe'] >= 0.5 else '#e67e22'
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
        conclusion = "优化变体未全面优于原版, 建议直接使用原版 trend_pullback"
        conclusion_color = '#e67e22'

    variant_detail_rows = ""
    for name, stats in variants:
        for s in stats.get('signal_list', []):
            is_win = s['is_win']
            win_class = 'win' if is_win else 'loss'
            ret_color = '#27ae60' if s['close_return'] >= 0 else '#e74c3c'
            maxret_color = '#27ae60' if s['max_return'] >= 0 else '#e74c3c'
            short_name = name.split(' ')[0] if ' ' in name else name
            variant_detail_rows += f"""
            <tr class="{win_class}">
                <td>{short_name}</td>
                <td>{s['date']}</td>
                <td style="font-weight:bold;">{s['score']:.1f}</td>
                <td class="win-cell">{'WIN' if is_win else 'LOSS'}</td>
                <td style="color:{maxret_color};font-weight:bold;">{s['max_return']:+.2f}%</td>
                <td style="color:{ret_color};">{s['close_return']:+.2f}%</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>标普生物科技ETF - trend_pullback 优化分析</title>
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
        <h1>标普生物科技ETF - trend_pullback 优化分析</h1>
        <div class="meta">ETF: {ETF_CODE} | 回测: {START_DATE}~最新 | T+3 | 阈值&ge;{THRESHOLD}</div>
    </div>
    <div class="conclusion">{conclusion}</div>
    <div class="section">
        <h2>原版信号深度分析</h2>
        <table>
            <tr><th>日期</th><th>信号分</th><th>结果</th><th>最大收益</th><th>收盘收益</th><th>关键指标</th><th>触发理由</th></tr>
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
    <div class="footer"><p>标普生物科技ETF trend_pullback 优化分析 | T+3胜率回测</p></div>
</body>
</html>"""
    return html


def main():
    logger.info("加载数据...")
    engine = DataEngine()
    df = engine.get_history_kline(ETF_CODE)
    df = calc_all_indicators(df)
    logger.info(f"数据: {len(df)}条")

    # 原版
    logger.info("运行原版 trend_pullback...")
    original_algo = ALGORITHM_MAP['trend_pullback']()
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
            print(f"        RSI={ind.get('rsi_14',0):.1f} MA20_dev={ind.get('dev_ma20',0):+.1f}% "
                  f"MACD={ind.get('macd_hist',0):.4f} M20={ind.get('momentum_20',0):+.1f}% "
                  f"BB%B={ind.get('bb_percent_b',0):.2f} KDJ_J={ind.get('kdj_j',0):.0f} "
                  f"Vol={ind.get('vol_ratio',0):.1f}")

    # 变体
    variants = []
    for algo_cls, label in [
        (TrendPullback增强, '增强版(宽回踩+KDJ)'),
        (TrendPullback量价, '量价版(放量+KDJ)'),
        (TrendPullback混合, '混合版(宽松+KDJ+放量)'),
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
    print(f"{'原版 trend_pullback':<30s} {original_stats['n']:4d} {original_stats['win_rate']:5.1f}% "
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

    # HTML + JSON
    html = generate_html(original_stats, variants, original_signals)
    html_path = os.path.join(OUTPUT_DIR, '标普生物科技ETF_trend_pullback优化.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML: {html_path}")

    json_data = {
        'original': {k: v for k, v in original_stats.items() if k != 'signal_list'},
        'variants': {name: {k: v for k, v in stats.items() if k != 'signal_list'} for name, stats in variants},
    }
    json_path = os.path.join(OUTPUT_DIR, '标普生物科技ETF_trend_pullback优化.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"JSON: {json_path}")


if __name__ == '__main__':
    main()
