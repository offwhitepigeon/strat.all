# -*- coding: utf-8 -*-
"""
有色金属ETF - financial_value vs volatility_breakout 深度对比
"""
import sys, os, json, logging
from datetime import datetime
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import ALGORITHM_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'algo_optimization')
ETF_CODE = 'sh512400'
ETF_NAME = '有色金属ETF'
START_DATE = '2024-01-01'
THRESHOLD = 60


def run_algorithm(algo_name):
    """运行单个算法, 返回信号列表"""
    engine = DataEngine()
    df = engine.get_history_kline(ETF_CODE)
    df = calc_all_indicators(df)

    algo = ALGORITHM_MAP[algo_name]()
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
        future_close = float(future.iloc[-1]['close'])
        max_ret = (future_high / buy_price - 1) * 100
        close_ret = (future_close / buy_price - 1) * 100
        is_win = max_ret > 0.5

        # T+1, T+2, T+3 逐日收益
        daily_returns = []
        for j in range(min(3, len(future))):
            d_high = float(future.iloc[j]['high'])
            d_close = float(future.iloc[j]['close'])
            daily_returns.append({
                't%d_high' % (j+1): round((d_high / buy_price - 1) * 100, 2),
                't%d_close' % (j+1): round((d_close / buy_price - 1) * 100, 2),
            })

        days_to_win = 0
        if is_win:
            for j in range(len(future)):
                if (float(future.iloc[j]['high']) / buy_price - 1) * 100 > 0.5:
                    days_to_win = j + 1
                    break

        signals.append({
            'date': date_str,
            'score': round(signal.score, 1),
            'buy_price': round(buy_price, 4),
            'max_return': round(max_ret, 2),
            'close_return': round(close_ret, 2),
            'is_win': is_win,
            'days_to_win': days_to_win,
            'reasons': signal.reasons[:2],
            'daily': daily_returns,
        })

    return signals


def compute_stats(signals):
    """计算详细统计"""
    n = len(signals)
    if n == 0:
        return {}

    rets = [s['close_return'] for s in signals]
    max_rets = [s['max_return'] for s in signals]
    wins = [s for s in signals if s['is_win']]
    losses = [s for s in signals if not s['is_win']]

    # 收益分布
    dist = {
        '>+5%': sum(1 for r in max_rets if r > 5),
        '+3~5%': sum(1 for r in max_rets if 3 <= r <= 5),
        '+1~3%': sum(1 for r in max_rets if 1 <= r < 3),
        '0~1%': sum(1 for r in max_rets if 0 <= r < 1),
        '-1~0%': sum(1 for r in max_rets if -1 <= r < 0),
        '<-1%': sum(1 for r in max_rets if r < -1),
    }

    # 亏损信号分析
    loss_rets = [s['close_return'] for s in losses] if losses else []
    win_rets = [s['max_return'] for s in wins] if wins else []

    # 按年统计
    by_year = {}
    for s in signals:
        yr = s['date'][:4]
        if yr not in by_year:
            by_year[yr] = {'signals': 0, 'wins': 0, 'rets': []}
        by_year[yr]['signals'] += 1
        if s['is_win']:
            by_year[yr]['wins'] += 1
        by_year[yr]['rets'].append(s['close_return'])

    for yr in by_year:
        d = by_year[yr]
        d['win_rate'] = round(d['wins'] / d['signals'] * 100, 1) if d['signals'] > 0 else 0
        d['avg_ret'] = round(np.mean(d['rets']), 2) if d['rets'] else 0
        del d['rets']

    # 信号分分布
    scores = [s['score'] for s in signals]
    score_dist = {
        '60-70': sum(1 for s in scores if 60 <= s < 70),
        '70-80': sum(1 for s in scores if 70 <= s < 80),
        '80-90': sum(1 for s in scores if 80 <= s < 90),
        '90+': sum(1 for s in scores if s >= 90),
    }

    # 月份分布
    by_month = {}
    for s in signals:
        m = s['date'][:7]
        by_month[m] = by_month.get(m, 0) + 1

    # 达标天数分布
    dtw = [s['days_to_win'] for s in wins] if wins else []
    dtw_dist = {
        'T+1': sum(1 for d in dtw if d == 1),
        'T+2': sum(1 for d in dtw if d == 2),
        'T+3': sum(1 for d in dtw if d == 3),
    }

    return {
        'n': n,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(len(wins) / n * 100, 1),
        'avg_return': round(np.mean(rets), 2),
        'avg_max_return': round(np.mean(max_rets), 2),
        'std_return': round(np.std(rets), 2),
        'std_max_return': round(np.std(max_rets), 2),
        'sharpe_like': round(np.mean(rets) / np.std(rets), 2) if np.std(rets) > 0 else 0,
        'max_sharpe': round(np.mean(max_rets) / np.std(max_rets), 2) if np.std(max_rets) > 0 else 0,
        'best': round(max(max_rets), 2),
        'worst': round(min(rets), 2),
        'worst_max': round(min(max_rets), 2),
        'avg_win_max': round(np.mean(win_rets), 2) if win_rets else 0,
        'avg_loss_ret': round(np.mean(loss_rets), 2) if loss_rets else 0,
        'dist': dist,
        'by_year': by_year,
        'score_dist': score_dist,
        'by_month': by_month,
        'dtw_dist': dtw_dist,
        'avg_score': round(np.mean(scores), 1),
    }


def generate_html(fv_stats, vb_stats, fv_signals, vb_signals):
    """生成对比HTML"""

    def metric_row(label, fv_val, vb_val, higher_better=True):
        fv_better = fv_val > vb_val if higher_better else fv_val < vb_val
        fv_color = '#27ae60' if (fv_better and fv_val != vb_val) else '#2c3e50'
        vb_color = '#27ae60' if (not fv_better and fv_val != vb_val) else '#2c3e50'
        return f"""
            <tr>
                <td class="metric-label">{label}</td>
                <td style="color:{fv_color};font-weight:bold;">{fv_val}</td>
                <td style="color:{vb_color};font-weight:bold;">{vb_val}</td>
            </tr>"""

    def dist_row(label, fv_val, vb_val):
        total_fv = fv_stats['n']
        total_vb = vb_stats['n']
        fv_pct = f"{fv_val} ({fv_val/total_fv*100:.0f}%)" if total_fv else "0"
        vb_pct = f"{vb_val} ({vb_val/total_vb*100:.0f}%)" if total_vb else "0"
        return f"""
            <tr>
                <td class="metric-label">{label}</td>
                <td>{fv_pct}</td>
                <td>{vb_pct}</td>
            </tr>"""

    # 逐年对比
    years = sorted(set(list(fv_stats.get('by_year', {}).keys()) + list(vb_stats.get('by_year', {}).keys())))
    year_rows = ""
    for yr in years:
        fv = fv_stats.get('by_year', {}).get(yr, {})
        vb = vb_stats.get('by_year', {}).get(yr, {})
        fv_str = f"{fv.get('signals',0)}信号/{fv.get('win_rate',0)}%/{fv.get('avg_ret',0):+.2f}%" if fv else "-"
        vb_str = f"{vb.get('signals',0)}信号/{vb.get('win_rate',0)}%/{vb.get('avg_ret',0):+.2f}%" if vb else "-"
        year_rows += f"<tr><td>{yr}</td><td>{fv_str}</td><td>{vb_str}</td></tr>"

    # 信号明细
    detail_rows = ""
    for algo_name, sigs in [('financial_value', fv_signals), ('volatility_breakout', vb_signals)]:
        for s in sigs:
            is_win = s['is_win']
            win_class = 'win' if is_win else 'loss'
            win_text = 'WIN' if is_win else 'LOSS'
            ret_color = '#27ae60' if s['close_return'] >= 0 else '#e74c3c'
            maxret_color = '#27ae60' if s['max_return'] >= 0 else '#e74c3c'
            reasons = '; '.join(s.get('reasons', [])) if s.get('reasons') else '-'
            detail_rows += f"""
            <tr class="{win_class}">
                <td>{algo_name}</td>
                <td>{s['date']}</td>
                <td style="font-weight:bold;">{s['score']:.1f}</td>
                <td>{s['buy_price']:.3f}</td>
                <td class="win-cell">{win_text}</td>
                <td style="color:{maxret_color};font-weight:bold;">{s['max_return']:+.2f}%</td>
                <td style="color:{ret_color};">{s['close_return']:+.2f}%</td>
                <td>{s['days_to_win']}d</td>
                <td class="reasons-cell">{reasons}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>有色金属ETF - financial_value vs volatility_breakout</title>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif; background:#f5f6fa; color:#2c3e50; padding:20px; line-height:1.6; }}
        .header {{ background:linear-gradient(135deg,#1a1a2e,#16213e); color:white; padding:25px; border-radius:12px; margin-bottom:20px; }}
        .header h1 {{ font-size:22px; margin-bottom:8px; }}
        .header .meta {{ opacity:0.8; font-size:13px; }}
        .section {{ background:white; border-radius:12px; padding:20px; margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,0.08); overflow-x:auto; }}
        .section h2 {{ font-size:17px; margin-bottom:15px; color:#2c3e50; border-bottom:2px solid #ecf0f1; padding-bottom:10px; }}
        table {{ width:100%; border-collapse:collapse; font-size:13px; }}
        th {{ background:#f8f9fa; padding:10px; text-align:left; border-bottom:2px solid #dee2e6; white-space:nowrap; }}
        td {{ padding:8px 10px; border-bottom:1px solid #ecf0f1; }}
        .metric-label {{ font-weight:bold; color:#495057; background:#f8f9fa; }}
        tr.win {{ background:#f0fff4; }}
        tr.loss {{ background:#fff5f5; }}
        .win-cell {{ font-weight:bold; text-align:center; }}
        tr.win .win-cell {{ color:#27ae60; }}
        tr.loss .win-cell {{ color:#e74c3c; }}
        .reasons-cell {{ font-size:11px; color:#555; max-width:280px; }}
        .algo-header {{ background:#e74c3c; color:white; }} 
        .footer {{ text-align:center; padding:20px; color:#95a5a6; font-size:12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>有色金属ETF - financial_value vs volatility_breakout 深度对比</h1>
        <div class="meta">
            ETF: {ETF_CODE} {ETF_NAME} |
            回测: {START_DATE} ~ 最新 | 阈值&ge;{THRESHOLD} | T+3胜率(max_ret > 0.5%)
        </div>
    </div>

    <div class="section">
        <h2>核心指标对比</h2>
        <table>
            <tr class="algo-header">
                <th>指标</th>
                <th>financial_value</th>
                <th>volatility_breakout</th>
            </tr>
            {metric_row('信号数', fv_stats['n'], vb_stats['n'])}
            {metric_row('胜率', f"{fv_stats['win_rate']}%", f"{vb_stats['win_rate']}%")}
            {metric_row('平均收盘收益', f"{fv_stats['avg_return']:+.2f}%", f"{vb_stats['avg_return']:+.2f}%")}
            {metric_row('平均最大收益', f"{fv_stats['avg_max_return']:+.2f}%", f"{vb_stats['avg_max_return']:+.2f}%")}
            {metric_row('收益标准差', f"{fv_stats['std_return']:.2f}%", f"{vb_stats['std_return']:.2f}%")}
            {metric_row('最大收益标准差', f"{fv_stats['std_max_return']:.2f}%", f"{vb_stats['std_max_return']:.2f}%")}
            {metric_row('风险调整(收益/标准差)', f"{fv_stats['sharpe_like']:.2f}", f"{vb_stats['sharpe_like']:.2f}")}
            {metric_row('最大收益风险调整', f"{fv_stats['max_sharpe']:.2f}", f"{vb_stats['max_sharpe']:.2f}")}
            {metric_row('最佳单笔', f"{fv_stats['best']:+.2f}%", f"{vb_stats['best']:+.2f}%")}
            {metric_row('最差收盘', f"{fv_stats['worst']:+.2f}%", f"{vb_stats['worst']:+.2f}%")}
            {metric_row('最差最大收益', f"{fv_stats['worst_max']:+.2f}%", f"{vb_stats['worst_max']:+.2f}%")}
            {metric_row('盈利信号均最大收益', f"{fv_stats['avg_win_max']:+.2f}%", f"{vb_stats['avg_win_max']:+.2f}%")}
            {metric_row('亏损信号均收盘收益', f"{fv_stats['avg_loss_ret']:+.2f}%", f"{vb_stats['avg_loss_ret']:+.2f}%")}
            {metric_row('平均信号分', f"{fv_stats['avg_score']:.1f}", f"{vb_stats['avg_score']:.1f}")}
        </table>
    </div>

    <div class="section">
        <h2>最大收益分布</h2>
        <table>
            <tr class="algo-header"><th>收益区间(max_return)</th><th>financial_value</th><th>volatility_breakout</th></tr>
            {dist_row('>+5%', fv_stats['dist']['>+5%'], vb_stats['dist']['>+5%'])}
            {dist_row('+3~5%', fv_stats['dist']['+3~5%'], vb_stats['dist']['+3~5%'])}
            {dist_row('+1~3%', fv_stats['dist']['+1~3%'], vb_stats['dist']['+1~3%'])}
            {dist_row('0~1%', fv_stats['dist']['0~1%'], vb_stats['dist']['0~1%'])}
            {dist_row('-1~0%', fv_stats['dist']['-1~0%'], vb_stats['dist']['-1~0%'])}
            {dist_row('<-1%', fv_stats['dist']['<-1%'], vb_stats['dist']['<-1%'])}
        </table>
    </div>

    <div class="section">
        <h2>逐年表现</h2>
        <table>
            <tr class="algo-header"><th>年份</th><th>financial_value</th><th>volatility_breakout</th></tr>
            {year_rows}
        </table>
    </div>

    <div class="section">
        <h2>信号分数分布</h2>
        <table>
            <tr class="algo-header"><th>分数区间</th><th>financial_value</th><th>volatility_breakout</th></tr>
            {dist_row('60-70', fv_stats['score_dist']['60-70'], vb_stats['score_dist']['60-70'])}
            {dist_row('70-80', fv_stats['score_dist']['70-80'], vb_stats['score_dist']['70-80'])}
            {dist_row('80-90', fv_stats['score_dist']['80-90'], vb_stats['score_dist']['80-90'])}
            {dist_row('90+', fv_stats['score_dist']['90+'], vb_stats['score_dist']['90+'])}
        </table>
    </div>

    <div class="section">
        <h2>胜利信号达标天数分布</h2>
        <table>
            <tr class="algo-header"><th>达标天数</th><th>financial_value</th><th>volatility_breakout</th></tr>
            {dist_row('T+1', fv_stats['dtw_dist']['T+1'], vb_stats['dtw_dist']['T+1'])}
            {dist_row('T+2', fv_stats['dtw_dist']['T+2'], vb_stats['dtw_dist']['T+2'])}
            {dist_row('T+3', fv_stats['dtw_dist']['T+3'], vb_stats['dtw_dist']['T+3'])}
        </table>
    </div>

    <div class="section">
        <h2>全部信号明细</h2>
        <table>
            <tr class="algo-header">
                <th>算法</th><th>日期</th><th>信号分</th><th>买入价</th>
                <th>结果</th><th>最大收益</th><th>收盘收益</th><th>达标天数</th><th>理由</th>
            </tr>
            {detail_rows}
        </table>
    </div>

    <div class="footer">
        <p>有色金属ETF算法优化 - financial_value vs volatility_breakout 深度对比</p>
    </div>
</body>
</html>"""
    return html


def main():
    logger.info("运行 financial_value...")
    fv_signals = run_algorithm('financial_value')
    logger.info(f"  {len(fv_signals)} 信号")

    logger.info("运行 volatility_breakout...")
    vb_signals = run_algorithm('volatility_breakout')
    logger.info(f"  {len(vb_signals)} 信号")

    fv_stats = compute_stats(fv_signals)
    vb_stats = compute_stats(vb_signals)

    # 打印对比
    print(f"\n{'='*80}")
    print(f"{'指标':<25s} {'financial_value':>20s} {'volatility_breakout':>20s}")
    print(f"{'='*80}")
    print(f"{'信号数':<25s} {fv_stats['n']:>20d} {vb_stats['n']:>20d}")
    print(f"{'胜率':<25s} {fv_stats['win_rate']:>19.1f}% {vb_stats['win_rate']:>19.1f}%")
    print(f"{'平均收盘收益':<23s} {fv_stats['avg_return']:>+19.2f}% {vb_stats['avg_return']:>+19.2f}%")
    print(f"{'平均最大收益':<23s} {fv_stats['avg_max_return']:>+19.2f}% {vb_stats['avg_max_return']:>+19.2f}%")
    print(f"{'收益标准差':<24s} {fv_stats['std_return']:>19.2f}% {vb_stats['std_return']:>19.2f}%")
    print(f"{'风险调整(收益/σ)':<22s} {fv_stats['sharpe_like']:>20.2f} {vb_stats['sharpe_like']:>20.2f}")
    print(f"{'最大收益风险调整':<21s} {fv_stats['max_sharpe']:>20.2f} {vb_stats['max_sharpe']:>20.2f}")
    print(f"{'最差收盘':<25s} {fv_stats['worst']:>+19.2f}% {vb_stats['worst']:>+19.2f}%")
    print(f"{'最差最大收益':<23s} {fv_stats['worst_max']:>+19.2f}% {vb_stats['worst_max']:>+19.2f}%")
    print(f"{'盈利均最大收益':<22s} {fv_stats['avg_win_max']:>+19.2f}% {vb_stats['avg_win_max']:>+19.2f}%")
    print(f"{'亏损均收盘收益':<22s} {fv_stats['avg_loss_ret']:>+19.2f}% {vb_stats['avg_loss_ret']:>+19.2f}%")
    print(f"{'平均信号分':<24s} {fv_stats['avg_score']:>20.1f} {vb_stats['avg_score']:>20.1f}")

    print(f"\n{'收益分布(max_return):':<25s}")
    for k in ['>+5%', '+3~5%', '+1~3%', '0~1%', '-1~0%', '<-1%']:
        print(f"  {k:<23s} {fv_stats['dist'][k]:>10d} {vb_stats['dist'][k]:>20d}")

    print(f"\n{'逐年表现:'}")
    years = sorted(set(list(fv_stats.get('by_year', {}).keys()) + list(vb_stats.get('by_year', {}).keys())))
    for yr in years:
        fv = fv_stats.get('by_year', {}).get(yr, {})
        vb = vb_stats.get('by_year', {}).get(yr, {})
        fv_str = f"{fv.get('signals',0)}信号/{fv.get('win_rate',0)}%/{fv.get('avg_ret',0):+.2f}%" if fv else "-"
        vb_str = f"{vb.get('signals',0)}信号/{vb.get('win_rate',0)}%/{vb.get('avg_ret',0):+.2f}%" if vb else "-"
        print(f"  {yr}: FV={fv_str:<30s} VB={vb_str}")

    # 保存JSON
    json_data = {
        'financial_value': fv_stats,
        'volatility_breakout': vb_stats,
    }
    json_path = os.path.join(OUTPUT_DIR, '有色ETF_FVvsVB.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"JSON: {json_path}")

    # 生成HTML
    html = generate_html(fv_stats, vb_stats, fv_signals, vb_signals)
    html_path = os.path.join(OUTPUT_DIR, '有色ETF_FVvsVB.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML: {html_path}")


if __name__ == '__main__':
    main()
