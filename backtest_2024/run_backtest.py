# -*- coding: utf-8 -*-
"""
ETF多策略回测脚本 - 2024-01-01起
=================================
对24只ETF × 13种算法进行全面回测，生成JSON数据 + HTML报告。

用法:
    python backtest_2024/run_backtest.py

输出:
    backtest_2024/backtest_result.json   - 完整回测数据
    backtest_2024/backtest_report.html    - HTML可视化报告
"""

import sys
import os
import json
import logging
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backtest_engine import BacktestEngine
from etf_config import ETF_POOL

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 输出目录
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_backtest():
    """执行回测并保存结果"""
    engine = BacktestEngine()

    logger.info("=" * 60)
    logger.info("ETF多策略回测 - 2024-01-01起")
    logger.info(f"ETF数量: {len(ETF_POOL)}")
    logger.info("=" * 60)

    result = engine.run_backtest(
        signal_threshold=60,
        start_date='2024-01-01',
    )

    # 保存JSON
    json_path = os.path.join(OUTPUT_DIR, 'backtest_result.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"JSON结果已保存: {json_path}")

    return result


def generate_html_report(result):
    """生成HTML回测报告"""
    meta = result['metadata']
    summary = result['summary']
    stats_by_etf = result['stats_by_etf']
    signals = result['signals']

    # 按算法统计
    algo_stats = summary.get('by_algorithm', {})
    level_stats = summary.get('by_level', {})
    score_bins = summary.get('by_score_bin', {})

    # 按ETF统计表格行
    etf_rows = ""
    for s in sorted(stats_by_etf, key=lambda x: -x['win_rate']):
        wr = s['win_rate'] * 100
        wr_color = '#27ae60' if wr >= 75 else ('#e67e22' if wr >= 60 else '#e74c3c')
        etf_rows += f"""
            <tr>
                <td><strong>{s['etf_name']}</strong></td>
                <td><span class="algo-tag">{s['algorithm']}</span></td>
                <td>{s['total_signals']}</td>
                <td style="color:{wr_color};font-weight:bold;">{wr:.1f}%</td>
                <td>{s['wins']}</td>
                <td>{s['losses']}</td>
                <td>{s['avg_score']:.1f}</td>
                <td style="color:{'#27ae60' if s['avg_return_3d'] >= 0 else '#e74c3c'};">{s['avg_return_3d']:+.2f}%</td>
                <td style="color:{'#27ae60' if s['avg_max_return'] >= 0 else '#e74c3c'};">{s['avg_max_return']:+.2f}%</td>
                <td>{s['median_return_3d']:+.2f}%</td>
                <td style="color:#27ae60;">{s['best_return']:+.2f}%</td>
                <td style="color:#e74c3c;">{s['worst_return']:+.2f}%</td>
            </tr>"""

    # 按算法统计表格行
    algo_rows = ""
    for algo, stats in sorted(algo_stats.items(), key=lambda x: -x[1]['win_rate']):
        wr = stats['win_rate']
        wr_color = '#27ae60' if wr >= 75 else ('#e67e22' if wr >= 60 else '#e74c3c')
        etfs = ', '.join(stats.get('etfs', []))
        algo_rows += f"""
            <tr>
                <td><span class="algo-tag">{algo}</span></td>
                <td>{stats['total_signals']}</td>
                <td style="color:{wr_color};font-weight:bold;">{wr:.1f}%</td>
                <td>{stats['wins']}</td>
                <td style="color:{'#27ae60' if stats['avg_return_3d'] >= 0 else '#e74c3c'};">{stats['avg_return_3d']:+.2f}%</td>
                <td style="color:{'#27ae60' if stats['avg_max_return'] >= 0 else '#e74c3c'};">{stats['avg_max_return']:+.2f}%</td>
                <td>{etfs}</td>
            </tr>"""

    # 按信号等级统计
    level_rows = ""
    for level, stats in sorted(level_stats.items(), key=lambda x: -x[1]['win_rate']):
        wr = stats['win_rate']
        wr_color = '#27ae60' if wr >= 75 else ('#e67e22' if wr >= 60 else '#e74c3c')
        level_rows += f"""
            <tr>
                <td><span class="level-badge level-{level.lower()}">{level}</span></td>
                <td>{stats['total_signals']}</td>
                <td style="color:{wr_color};font-weight:bold;">{wr:.1f}%</td>
                <td style="color:{'#27ae60' if stats['avg_return_3d'] >= 0 else '#e74c3c'};">{stats['avg_return_3d']:+.2f}%</td>
            </tr>"""

    # 按得分区间统计
    score_rows = ""
    for bin_name, stats in sorted(score_bins.items()):
        wr = stats['win_rate']
        wr_color = '#27ae60' if wr >= 75 else ('#e67e22' if wr >= 60 else '#e74c3c')
        score_rows += f"""
            <tr>
                <td><strong>{bin_name}分</strong></td>
                <td>{stats['total_signals']}</td>
                <td style="color:{wr_color};font-weight:bold;">{wr:.1f}%</td>
                <td style="color:{'#27ae60' if stats['avg_return_3d'] >= 0 else '#e74c3c'};">{stats['avg_return_3d']:+.2f}%</td>
            </tr>"""

    # 信号明细表（最近50条，按日期倒序）
    detail_rows = ""
    recent_signals = sorted(signals, key=lambda x: x['date'], reverse=True)[:50]
    for sig in recent_signals:
        is_win = sig['is_win']
        win_class = 'win' if is_win else 'loss'
        win_text = '✓' if is_win else '✗'
        ret_color = '#27ae60' if sig['return_3d'] >= 0 else '#e74c3c'
        maxret_color = '#27ae60' if sig['max_return_3d'] >= 0 else '#e74c3c'
        detail_rows += f"""
            <tr class="{win_class}">
                <td>{sig['date']}</td>
                <td><strong>{sig['etf_name']}</strong></td>
                <td><span class="algo-tag">{sig['algorithm']}</span></td>
                <td>{sig['buy_price']:.3f}</td>
                <td style="font-weight:bold;">{sig['score']:.1f}</td>
                <td><span class="level-badge level-{sig['level'].lower()}">{sig['level']}</span></td>
                <td class="win-cell">{win_text}</td>
                <td>{sig['hold_days_to_win']}日</td>
                <td style="color:{maxret_color};font-weight:bold;">{sig['max_return_3d']:+.2f}%</td>
                <td style="color:{ret_color};">{sig['return_3d']:+.2f}%</td>
                <td>{sig['high_3d']:.3f}</td>
                <td>{sig['close_3d']:.3f}</td>
            </tr>"""

    # 按月统计
    monthly_stats = {}
    for sig in signals:
        month = sig['date'][:7]  # YYYY-MM
        if month not in monthly_stats:
            monthly_stats[month] = {'signals': 0, 'wins': 0, 'returns': []}
        monthly_stats[month]['signals'] += 1
        if sig['is_win']:
            monthly_stats[month]['wins'] += 1
        monthly_stats[month]['returns'].append(sig['return_3d'])

    month_rows = ""
    for month in sorted(monthly_stats.keys()):
        ms = monthly_stats[month]
        wr = ms['wins'] / ms['signals'] * 100 if ms['signals'] > 0 else 0
        avg_ret = sum(ms['returns']) / len(ms['returns']) if ms['returns'] else 0
        wr_color = '#27ae60' if wr >= 75 else ('#e67e22' if wr >= 60 else '#e74c3c')
        month_rows += f"""
            <tr>
                <td><strong>{month}</strong></td>
                <td>{ms['signals']}</td>
                <td style="color:{wr_color};font-weight:bold;">{wr:.1f}%</td>
                <td>{ms['wins']}</td>
                <td style="color:{'#27ae60' if avg_ret >= 0 else '#e74c3c'};">{avg_ret:+.2f}%</td>
            </tr>"""

    total_wr = summary['total_win_rate']
    total_wr_color = '#27ae60' if total_wr >= 75 else ('#e67e22' if total_wr >= 60 else '#e74c3c')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF多策略回测报告 | 2024-01-01起</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif;
            background: #f5f6fa; color: #2c3e50; padding: 20px; line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px;
        }}
        .header h1 {{ font-size: 24px; margin-bottom: 8px; }}
        .header .meta {{ opacity: 0.8; font-size: 14px; }}
        .summary-cards {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
        .card {{
            background: white; padding: 20px; border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1; min-width: 140px;
            text-align: center;
        }}
        .card .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .card .label {{ font-size: 13px; color: #7f8c8d; margin-top: 5px; }}
        .card.success .value {{ color: #27ae60; }}
        .card.alert .value {{ color: #e74c3c; }}
        .card.warn .value {{ color: #e67e22; }}
        .section {{
            background: white; border-radius: 12px; padding: 20px;
            margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow-x: auto;
        }}
        .section h2 {{
            font-size: 18px; margin-bottom: 15px; color: #2c3e50;
            border-bottom: 2px solid #ecf0f1; padding-bottom: 10px;
        }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{
            background: #f8f9fa; padding: 10px; text-align: left;
            border-bottom: 2px solid #dee2e6; color: #495057; white-space: nowrap;
        }}
        td {{
            padding: 8px 10px; border-bottom: 1px solid #ecf0f1; vertical-align: top;
        }}
        tr.win {{ background: #f0fff4; }}
        tr.loss {{ background: #fff5f5; }}
        .win-cell {{ font-weight: bold; text-align: center; }}
        tr.win .win-cell {{ color: #27ae60; }}
        tr.loss .win-cell {{ color: #e74c3c; }}
        .algo-tag {{
            background: #ecf0f1; padding: 2px 8px; border-radius: 4px;
            font-size: 11px; color: #2c3e50; white-space: nowrap;
        }}
        .level-badge {{
            padding: 2px 8px; border-radius: 10px; font-size: 11px;
            font-weight: bold; color: white; white-space: nowrap;
        }}
        .level-strong_buy {{ background: #e74c3c; }}
        .level-buy {{ background: #27ae60; }}
        .level-light_buy {{ background: #f39c12; }}
        .level-watch {{ background: #7f8c8d; }}
        .level-wait {{ background: #bdc3c7; }}
        .footer {{
            text-align: center; padding: 20px; color: #95a5a6;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ETF多策略回测报告</h1>
        <div class="meta">
            回测区间: {meta.get('start_date', '2024-01-01')} ~ {meta.get('end_date', '最新')} |
            ETF数量: {meta.get('etf_count', 0)}只 |
            算法数量: 13种 |
            信号阈值: &ge;{meta.get('signal_threshold', 60)}分 |
            胜率定义: T+3/T+5最高价收益 &gt; 0.5% |
            回测耗时: {meta.get('duration_seconds', 0)}秒
        </div>
    </div>

    <div class="summary-cards">
        <div class="card">
            <div class="value">{summary['total_signals']}</div>
            <div class="label">总信号数</div>
        </div>
        <div class="card success">
            <div class="value" style="color:{total_wr_color};">{total_wr:.1f}%</div>
            <div class="label">总胜率</div>
        </div>
        <div class="card success">
            <div class="value">{summary['total_wins']}</div>
            <div class="label">胜利次数</div>
        </div>
        <div class="card warn">
            <div class="value">{summary.get('avg_return_3d', 0):+.2f}%</div>
            <div class="label">平均T+N收盘收益</div>
        </div>
        <div class="card success">
            <div class="value">{summary.get('avg_max_return', 0):+.2f}%</div>
            <div class="label">平均T+N最大收益</div>
        </div>
    </div>

    <div class="section">
        <h2>算法表现排名（按胜率降序）</h2>
        <table>
            <tr>
                <th>算法</th>
                <th>信号数</th>
                <th>胜率</th>
                <th>胜利</th>
                <th>平均收盘收益</th>
                <th>平均最大收益</th>
                <th>适用ETF</th>
            </tr>
            {algo_rows}
        </table>
    </div>

    <div class="section">
        <h2>各ETF回测统计（按胜率降序）</h2>
        <table>
            <tr>
                <th>ETF名称</th>
                <th>算法</th>
                <th>信号数</th>
                <th>胜率</th>
                <th>胜利</th>
                <th>失败</th>
                <th>平均分</th>
                <th>平均收盘收益</th>
                <th>平均最大收益</th>
                <th>中位收益</th>
                <th>最佳最大收益</th>
                <th>最差收盘收益</th>
            </tr>
            {etf_rows}
        </table>
    </div>

    <div class="section">
        <h2>信号等级分布</h2>
        <table>
            <tr><th>信号等级</th><th>信号数</th><th>胜率</th><th>平均收益</th></tr>
            {level_rows}
        </table>
    </div>

    <div class="section">
        <h2>信号分段胜率</h2>
        <table>
            <tr><th>得分区间</th><th>信号数</th><th>胜率</th><th>平均收益</th></tr>
            {score_rows}
        </table>
    </div>

    <div class="section">
        <h2>月度统计</h2>
        <table>
            <tr><th>月份</th><th>信号数</th><th>胜率</th><th>胜利</th><th>平均收益</th></tr>
            {month_rows}
        </table>
    </div>

    <div class="section">
        <h2>信号明细（最近50条，按日期倒序）</h2>
        <table>
            <tr>
                <th>日期</th>
                <th>ETF</th>
                <th>算法</th>
                <th>买入价</th>
                <th>信号分</th>
                <th>等级</th>
                <th>胜</th>
                <th>达标天数</th>
                <th>最大收益</th>
                <th>收盘收益</th>
                <th>T+N最高</th>
                <th>T+N收盘</th>
            </tr>
            {detail_rows}
        </table>
    </div>

    <div class="footer">
        <p>ETF多策略信号系统 v2.0 | 24只ETF × 13种算法 | 回测起点: 2024-01-01</p>
        <p>胜率定义: T+3/T+5（石油类5日）内最高价收益 &gt; 0.5% | 信号阈值: &ge;60分</p>
        <p>本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
    </div>
</body>
</html>"""

    html_path = os.path.join(OUTPUT_DIR, 'backtest_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML报告已生成: {html_path}")
    return html_path


def main():
    # 1. 运行回测
    result = run_backtest()

    # 2. 生成HTML报告
    generate_html_report(result)

    # 3. 打印摘要
    summary = result['summary']
    print(f"\n{'=' * 60}")
    print(f"回测摘要 (2024-01-01起)")
    print(f"{'=' * 60}")
    print(f"总信号数: {summary['total_signals']}")
    print(f"总胜率: {summary['total_win_rate']:.1f}%")
    print(f"平均收益: {summary.get('avg_return_3d', 0):+.2f}%")
    print(f"平均最大收益: {summary.get('avg_max_return', 0):+.2f}%")

    print(f"\n按算法统计:")
    for algo, stats in sorted(summary.get('by_algorithm', {}).items(),
                              key=lambda x: -x[1]['win_rate']):
        print(f"  {algo:25s} | 信号:{stats['total_signals']:3d} | "
              f"胜率:{stats['win_rate']:5.1f}% | "
              f"平均收益:{stats['avg_return_3d']:+.2f}%")


if __name__ == '__main__':
    main()
