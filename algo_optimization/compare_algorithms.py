# -*- coding: utf-8 -*-
"""
ETF全算法对比回测 (通用版)
==========================
对指定ETF用全部13种算法分别回测, 找出最佳算法, 与当前算法对比。

用法:
    python compare_algorithms.py --code sz159502 --name 标普生物科技ETF --current support_rebound
    python compare_algorithms.py --code sh512400 --name 有色金属ETF --current cycle_momentum
"""
import sys, os, json, logging, argparse
from datetime import datetime
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

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'algo_optimization')
START_DATE = '2024-01-01'
THRESHOLD = 60


def run_all_algorithms(etf_code, etf_name, current_algo):
    engine = DataEngine()
    df = engine.get_history_kline(etf_code)
    if df is None:
        logger.error(f"无法获取 {etf_code} 数据")
        return

    logger.info(f"{etf_name} 数据: {len(df)}条, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    df = calc_all_indicators(df)

    results = {}

    for algo_name, algo_cls in ALGORITHM_MAP.items():
        algo = algo_cls()
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

            days_to_win = 0
            if is_win:
                for j in range(len(future)):
                    if (float(future.iloc[j]['high']) / buy_price - 1) * 100 > 0.5:
                        days_to_win = j + 1
                        break

            signals.append({
                'date': date_str,
                'score': round(signal.score, 1),
                'level': signal.level,
                'buy_price': round(buy_price, 4),
                'max_return': round(max_ret, 2),
                'close_return': round(close_ret, 2),
                'is_win': is_win,
                'days_to_win': days_to_win,
                'reasons': signal.reasons[:2],
            })

        n = len(signals)
        if n == 0:
            results[algo_name] = {
                'algorithm': algo_name, 'signals': 0, 'wins': 0, 'win_rate': 0,
                'avg_return': 0, 'avg_max_return': 0, 'avg_score': 0,
                'best': 0, 'worst': 0, 'signal_list': [],
            }
            logger.info(f"  {algo_name:25s} | 信号:  0")
            continue

        wins = sum(1 for s in signals if s['is_win'])
        rets = [s['close_return'] for s in signals]
        max_rets = [s['max_return'] for s in signals]
        scores = [s['score'] for s in signals]

        results[algo_name] = {
            'algorithm': algo_name,
            'signals': n,
            'wins': wins,
            'win_rate': round(wins / n * 100, 1),
            'avg_return': round(np.mean(rets), 2),
            'avg_max_return': round(np.mean(max_rets), 2),
            'avg_score': round(np.mean(scores), 1),
            'best': round(max(max_rets), 2),
            'worst': round(min(rets), 2),
            'signal_list': signals,
        }
        logger.info(f"  {algo_name:25s} | 信号:{n:3d} | 胜率:{wins/n*100:.1f}% | 均收益:{np.mean(rets):+.2f}% | 均最大:{np.mean(max_rets):+.2f}%")

    return results


def generate_html(results, etf_code, etf_name, current_algo):
    sorted_results = sorted(results.values(), key=lambda x: (-x['win_rate'], -x['signals']))

    current = results.get(current_algo, {})
    candidates = [r for r in sorted_results if r['signals'] >= 5]
    best = max(candidates, key=lambda x: (x['win_rate'], x['avg_max_return'])) if candidates else sorted_results[0]

    rows = ""
    for r in sorted_results:
        is_current = r['algorithm'] == current_algo
        is_best_algo = r['algorithm'] == best['algorithm']
        wr_color = '#27ae60' if r['win_rate'] >= 75 else ('#e67e22' if r['win_rate'] >= 60 else '#e74c3c')
        ret_color = '#27ae60' if r['avg_return'] >= 0 else '#e74c3c'
        maxret_color = '#27ae60' if r['avg_max_return'] >= 0 else '#e74c3c'
        marker = ''
        if is_current:
            marker = ' [当前]'
        if is_best_algo and not is_current:
            marker = ' [最佳]'
        rows += f"""
            <tr {'class="best-row"' if is_best_algo else ''} {'class="current-row"' if is_current and not is_best_algo else ''}>
                <td><strong>{r['algorithm']}</strong>{marker}</td>
                <td>{r['signals']}</td>
                <td>{r['wins']}</td>
                <td style="color:{wr_color};font-weight:bold;">{r['win_rate']:.1f}%</td>
                <td style="color:{ret_color};">{r['avg_return']:+.2f}%</td>
                <td style="color:{maxret_color};">{r['avg_max_return']:+.2f}%</td>
                <td>{r['avg_score']:.1f}</td>
                <td style="color:#27ae60;">{r['best']:+.2f}%</td>
                <td style="color:#e74c3c;">{r['worst']:+.2f}%</td>
            </tr>"""

    detail_rows = ""
    for algo_data in [current, best]:
        algo_name = algo_data.get('algorithm', '')
        signals = algo_data.get('signal_list', [])
        for s in signals:
            is_win = s['is_win']
            win_class = 'win' if is_win else 'loss'
            win_text = 'Y' if is_win else 'N'
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
                <td>{s['days_to_win']}日</td>
                <td class="reasons-cell">{reasons}</td>
            </tr>"""

    if best['algorithm'] == current_algo:
        conclusion = f"当前算法 {current_algo} 已是最佳, 无需更换算法, 建议直接优化算法参数"
        conclusion_color = '#27ae60'
    else:
        conclusion = f"最佳算法为 {best['algorithm']} (胜率{best['win_rate']:.1f}%, {best['signals']}信号), 当前 {current_algo} (胜率{current.get('win_rate',0):.1f}%, {current.get('signals',0)}信号), 建议考虑更换或优化"
        conclusion_color = '#e67e22'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{etf_name}算法对比</title>
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
        th {{ background:#f8f9fa; padding:10px; text-align:left; border-bottom:2px solid #dee2e6; color:#495057; white-space:nowrap; }}
        td {{ padding:8px 10px; border-bottom:1px solid #ecf0f1; }}
        tr.win {{ background:#f0fff4; }}
        tr.loss {{ background:#fff5f5; }}
        .win-cell {{ font-weight:bold; text-align:center; }}
        tr.win .win-cell {{ color:#27ae60; }}
        tr.loss .win-cell {{ color:#e74c3c; }}
        tr.best-row {{ background:#fff3cd; }}
        tr.current-row {{ background:#e8f4fd; }}
        .reasons-cell {{ font-size:11px; color:#555; max-width:300px; }}
        .footer {{ text-align:center; padding:20px; color:#95a5a6; font-size:12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{etf_name} 全算法对比回测</h1>
        <div class="meta">
            ETF: {etf_code} {etf_name} |
            回测区间: {START_DATE} ~ 最新 |
            信号阈值: &ge;{THRESHOLD}分 |
            胜率定义: T+3最高价收益 &gt; 0.5% |
            算法数: {len(results)}种
        </div>
    </div>

    <div class="conclusion">{conclusion}</div>

    <div class="section">
        <h2>算法对比 (按胜率降序, [当前]={current_algo}, [最佳]=推荐)</h2>
        <table>
            <tr>
                <th>算法</th><th>信号数</th><th>胜利</th><th>胜率</th>
                <th>平均收盘收益</th><th>平均最大收益</th><th>平均信号分</th>
                <th>最佳最大收益</th><th>最差收盘收益</th>
            </tr>
            {rows}
        </table>
    </div>

    <div class="section">
        <h2>当前算法 vs 最佳算法 - 信号明细</h2>
        <table>
            <tr>
                <th>算法</th><th>日期</th><th>信号分</th><th>买入价</th>
                <th>胜</th><th>最大收益</th><th>收盘收益</th><th>达标天数</th><th>理由</th>
            </tr>
            {detail_rows}
        </table>
    </div>

    <div class="footer">
        <p>{etf_name}算法优化 | 13种算法对比 | T+3胜率回测</p>
        <p>本报告仅供参考，不构成投资建议。</p>
    </div>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description='ETF全算法对比回测')
    parser.add_argument('--code', required=True, help='ETF代码 (如 sh512400)')
    parser.add_argument('--name', required=True, help='ETF名称 (如 有色金属ETF)')
    parser.add_argument('--current', required=True, help='当前算法名 (如 cycle_momentum)')
    args = parser.parse_args()

    etf_code = args.code
    etf_name = args.name
    current_algo = args.current

    logger.info(f"开始 {etf_name}({etf_code}) 全算法对比回测, 当前算法={current_algo}")

    results = run_all_algorithms(etf_code, etf_name, current_algo)

    # 保存JSON
    json_path = os.path.join(OUTPUT_DIR, f'{etf_name}_算法对比.json')
    json_data = {k: {kk: vv for kk, vv in v.items() if kk != 'signal_list'} for k, v in results.items()}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON: {json_path}")

    # 生成HTML
    html = generate_html(results, etf_code, etf_name, current_algo)
    html_path = os.path.join(OUTPUT_DIR, f'{etf_name}_算法对比.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML: {html_path}")

    # 打印摘要
    print(f"\n{'=' * 80}")
    print(f"{etf_name} 全算法对比 (按胜率降序)")
    print(f"{'=' * 80}")
    print(f"{'算法':<26s} {'信号':>4s} {'胜率':>6s} {'均收益':>7s} {'均最大':>7s}")
    print("-" * 80)
    sorted_results = sorted(results.values(), key=lambda x: (-x['win_rate'], -x['signals']))
    candidates = [r for r in sorted_results if r['signals'] >= 5]
    best_algo = max(candidates, key=lambda x: (x['win_rate'], x['avg_max_return'])) if candidates else sorted_results[0]
    for r in sorted_results:
        marker = ''
        if r['algorithm'] == current_algo:
            marker = ' <- 当前'
        elif r['algorithm'] == best_algo['algorithm']:
            marker = ' <- 最佳'
        print(f"{r['algorithm']:<26s} {r['signals']:4d} {r['win_rate']:5.1f}% {r['avg_return']:+6.2f}% {r['avg_max_return']:+6.2f}%{marker}")


if __name__ == '__main__':
    main()
