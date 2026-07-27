# -*- coding: utf-8 -*-
"""
7ETF组合回测 - T+1盈利卖出策略
================================
标的: 红利/沪深300/双创50/黄金股/黄金/石油LOF/标普油气 (7只)
卖出规则: 买入后T+1收盘盈利则卖出, 否则推迟一天, 最晚T+N卖出
  - N=3: 红利/沪深300/双创50/黄金股/黄金
  - N=5: 石油LOF/标普油气
初始净值=1, 每只ETF等权(1/7资金), 各ETF独立交易

输出:
  backtest_2024/focus7_result.json   - 回测数据
  backtest_2024/focus7_report.html    - HTML报告(含净值曲线)
"""

import sys
import os
import json
import logging
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from etf_config import ETF_POOL, ETFTarget
from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import get_algorithm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'backtest_2024')

# 目标7只ETF代码
TARGET_CODES = {
    'sh510880',  # 红利ETF
    'sh510300',  # 沪深300ETF
    'sz159782',  # 双创50ETF
    'sh517520',  # 黄金股ETF
    'sh518880',  # 黄金ETF
    'sz162411',  # 石油LOF
    'sz161129',  # 标普油气ETF
}

START_DATE = '2024-01-01'
SIGNAL_THRESHOLD = 60


def run_backtest():
    """执行回测"""
    engine = DataEngine()
    target_etfs = [e for e in ETF_POOL if e.code in TARGET_CODES]
    logger.info(f"目标ETF: {len(target_etfs)}只")

    # 每只ETF的 trades 列表
    all_trades = []

    for etf in target_etfs:
        df = engine.get_history_kline(etf.code)
        if df is None or len(df) < 250:
            logger.warning(f"{etf.name} 数据不足，跳过")
            continue
        df = calc_all_indicators(df)
        # 截取回测区间
        mask = df['date'] >= START_DATE
        df_bt = df[mask].copy()
        if len(df_bt) < 30:
            continue

        algorithm = get_algorithm(etf.algorithm)
        hold_days = getattr(etf, 'hold_days', 3)

        # 逐日生成信号
        trades = []
        for i in range(len(df_bt)):
            idx = df_bt.index[i]
            # 找到在原df中的位置
            pos = df.index.get_loc(idx)
            if pos < 60:
                continue
            df_slice = df.iloc[:pos + 1]
            try:
                signal = algorithm.calculate(df_slice)
            except Exception:
                continue
            if signal.score < SIGNAL_THRESHOLD:
                continue

            buy_price = float(df.iloc[pos]['close'])
            buy_date = df.iloc[pos]['date']

            # 获取未来N日数据
            future = df.iloc[pos + 1: pos + 1 + hold_days]
            if len(future) < 1:
                continue

            # === 卖出策略 ===
            # T+1收盘盈利则卖出, 否则推迟, 最晚T+N卖出
            sell_price = buy_price
            sell_day = 0
            sell_date = buy_date
            for j in range(len(future)):
                day_close = float(future.iloc[j]['close'])
                ret = (day_close / buy_price - 1) * 100
                sell_day = j + 1
                sell_date = future.iloc[j]['date']
                if ret > 0:
                    # 盈利, 卖出
                    sell_price = day_close
                    break
                sell_price = day_close  # 更新为最新收盘价(亏损也记录)
            # 如果到了最后一天还没卖出, 按最后一天收盘卖出
            # (sell_price 已经是最后一个检查日的收盘价)

            trade_return = (sell_price / buy_price - 1) * 100

            trade = {
                'date': buy_date.strftime('%Y-%m-%d') if hasattr(buy_date, 'strftime') else str(buy_date)[:10],
                'sell_date': sell_date.strftime('%Y-%m-%d') if hasattr(sell_date, 'strftime') else str(sell_date)[:10],
                'etf_code': etf.code,
                'etf_name': etf.name,
                'algorithm': etf.algorithm,
                'hold_days': hold_days,
                'buy_price': round(buy_price, 4),
                'sell_price': round(sell_price, 4),
                'sell_day': sell_day,
                'score': round(signal.score, 1),
                'level': signal.level,
                'return_pct': round(trade_return, 2),
                'is_win': trade_return > 0.5,
            }
            trades.append(trade)

        all_trades.extend(trades)
        if trades:
            wr = sum(1 for t in trades if t['is_win']) / len(trades) * 100
            avg_ret = sum(t['return_pct'] for t in trades) / len(trades)
            logger.info(f"{etf.name:12s} | 信号:{len(trades):3d} | 胜率:{wr:.1f}% | 平均收益:{avg_ret:+.2f}%")

    return all_trades, target_etfs


def calc_nav(all_trades, target_etfs, start_date='2024-01-01'):
    """
    计算净值:
    每只ETF等权(1/7初始资金), 各ETF独立交易
    组合净值 = 7只ETF净值的等权平均
    """
    n_etfs = len(target_etfs)

    # 按ETF分组, 计算每只ETF的净值曲线
    etf_trades = defaultdict(list)
    for t in all_trades:
        etf_trades[t['etf_code']].append(t)

    # 每只ETF的净值
    etf_navs = {}  # {code: [(date, nav), ...]}
    for etf in target_etfs:
        trades = sorted(etf_trades.get(etf.code, []), key=lambda x: x['date'])
        nav = 1.0
        nav_curve = [(start_date, 1.0)]
        for t in trades:
            ret = t['return_pct'] / 100
            nav *= (1 + ret)
            nav_curve.append((t['sell_date'], nav))
        etf_navs[etf.code] = {
            'name': etf.name,
            'nav_curve': nav_curve,
            'final_nav': nav,
            'trade_count': len(trades),
            'wins': sum(1 for t in trades if t['is_win']),
        }

    # 组合净值 = 等权平均
    # 收集所有日期点
    all_dates = set()
    for data in etf_navs.values():
        for date_str, _ in data['nav_curve']:
            all_dates.add(date_str)
    all_dates = sorted(all_dates)

    # 在每个日期点, 计算组合净值
    portfolio_nav = []
    for date_str in all_dates:
        navs = []
        for etf in target_etfs:
            data = etf_navs[etf.code]
            # 找到该ETF在该日期或之前最近的净值
            etf_nav = 1.0
            for d, n in data['nav_curve']:
                if d <= date_str:
                    etf_nav = n
                else:
                    break
            navs.append(etf_nav)
        portfolio_nav.append({
            'date': date_str,
            'nav': round(sum(navs) / n_etfs, 4),
        })

    return etf_navs, portfolio_nav


def generate_html(all_trades, etf_navs, portfolio_nav, target_etfs, duration):
    """生成HTML报告"""
    n_etfs = len(target_etfs)

    # 总体统计
    total_signals = len(all_trades)
    total_wins = sum(1 for t in all_trades if t['is_win'])
    total_win_rate = total_wins / total_signals * 100 if total_signals > 0 else 0
    avg_ret = sum(t['return_pct'] for t in all_trades) / total_signals if total_signals else 0
    avg_hold = sum(t['sell_day'] for t in all_trades) / total_signals if total_signals else 0

    # 组合最终净值
    final_nav = portfolio_nav[-1]['nav'] if portfolio_nav else 1.0
    max_nav = max(p['nav'] for p in portfolio_nav) if portfolio_nav else 1.0
    min_nav = min(p['nav'] for p in portfolio_nav) if portfolio_nav else 1.0

    # 最大回撤
    peak = 1.0
    max_dd = 0
    for p in portfolio_nav:
        if p['nav'] > peak:
            peak = p['nav']
        dd = (p['nav'] / peak - 1) * 100
        if dd < max_dd:
            max_dd = dd

    # 年化收益 (假设2024-01-01到2026-07-22约2.55年)
    start_dt = datetime.strptime('2024-01-01', '%Y-%m-%d')
    end_dt = datetime.now()
    years = (end_dt - start_dt).days / 365.25
    if years > 0 and final_nav > 0:
        cagr = (final_nav ** (1 / years) - 1) * 100
    else:
        cagr = 0

    # 各ETF统计表
    etf_rows = ""
    for etf in target_etfs:
        data = etf_navs[etf.code]
        tc = data['trade_count']
        w = data['wins']
        wr = w / tc * 100 if tc > 0 else 0
        fn = data['final_nav']
        trades = sorted([t for t in all_trades if t['etf_code'] == etf.code], key=lambda x: x['date'])
        avg_r = sum(t['return_pct'] for t in trades) / tc if tc else 0
        avg_h = sum(t['sell_day'] for t in trades) / tc if tc else 0
        wr_color = '#27ae60' if wr >= 75 else ('#e67e22' if wr >= 60 else '#e74c3c')
        nav_color = '#27ae60' if fn >= 1 else '#e74c3c'
        etf_rows += f"""
            <tr>
                <td><strong>{etf.name}</strong></td>
                <td><span class="algo-tag">{etf.algorithm}</span></td>
                <td>{tc}</td>
                <td style="color:{wr_color};font-weight:bold;">{wr:.1f}%</td>
                <td>{w}</td>
                <td style="color:{nav_color};font-weight:bold;">{fn:.4f}</td>
                <td style="color:{'#27ae60' if avg_r >= 0 else '#e74c3c'};">{avg_r:+.2f}%</td>
                <td>{avg_h:.1f}日</td>
            </tr>"""

    # 交易明细 (按日期倒序, 最近50条)
    detail_rows = ""
    for t in sorted(all_trades, key=lambda x: x['date'], reverse=True)[:50]:
        is_win = t['is_win']
        win_class = 'win' if is_win else 'loss'
        win_text = '✓' if is_win else '✗'
        ret_color = '#27ae60' if t['return_pct'] >= 0 else '#e74c3c'
        detail_rows += f"""
            <tr class="{win_class}">
                <td>{t['date']}</td>
                <td>{t['sell_date']}</td>
                <td><strong>{t['etf_name']}</strong></td>
                <td><span class="algo-tag">{t['algorithm']}</span></td>
                <td>{t['buy_price']:.3f}</td>
                <td>{t['sell_price']:.3f}</td>
                <td>T+{t['sell_day']}</td>
                <td>{t['score']:.1f}</td>
                <td class="win-cell">{win_text}</td>
                <td style="color:{ret_color};font-weight:bold;">{t['return_pct']:+.2f}%</td>
            </tr>"""

    # 净值曲线SVG
    # 计算SVG坐标
    nav_values = [p['nav'] for p in portfolio_nav]
    dates = [p['date'] for p in portfolio_nav]
    n_points = len(nav_values)

    svg_w = 900
    svg_h = 360
    pad_l = 60
    pad_r = 20
    pad_t = 20
    pad_b = 40
    plot_w = svg_w - pad_l - pad_r
    plot_h = svg_h - pad_t - pad_b

    nav_min = min(min(nav_values), 0.95)
    nav_max = max(max(nav_values), 1.05)
    y_range = nav_max - nav_min

    # 采样: 如果点太多, 降采样
    if n_points > 200:
        step = n_points // 200
        sampled_idx = list(range(0, n_points, step)) + [n_points - 1]
    else:
        sampled_idx = list(range(n_points))

    # 生成path
    points = []
    for idx in sampled_idx:
        x = pad_l + (idx / max(n_points - 1, 1)) * plot_w
        y = pad_t + (1 - (nav_values[idx] - nav_min) / y_range) * plot_h
        points.append(f"{x:.1f},{y:.1f}")

    path_d = "M " + " L ".join(points)

    # 基准线 (nav=1)
    y_base = pad_t + (1 - (1.0 - nav_min) / y_range) * plot_h

    # Y轴标签
    y_labels = ""
    for v in [nav_min, nav_min + y_range * 0.25, nav_min + y_range * 0.5, nav_min + y_range * 0.75, nav_max]:
        y = pad_t + (1 - (v - nav_min) / y_range) * plot_h
        y_labels += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{svg_w - pad_r}" y2="{y:.1f}" stroke="#ecf0f1" stroke-width="1"/><text x="{pad_l - 5}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#7f8c8d">{v:.3f}</text>'

    # X轴标签 (每3个月)
    x_labels = ""
    seen_months = set()
    for idx in sampled_idx:
        d = dates[idx]
        month = d[:7]
        if month not in seen_months:
            seen_months.add(month)
            x = pad_l + (idx / max(n_points - 1, 1)) * plot_w
            x_labels += f'<text x="{x:.1f}" y="{svg_h - 15}" text-anchor="middle" font-size="10" fill="#7f8c8d">{d[2:7]}</text>'

    nav_color = '#27ae60' if final_nav >= 1 else '#e74c3c'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>7ETF组合回测报告 | T+1盈利卖出策略</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif; background: #f5f6fa; color: #2c3e50; padding: 20px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 24px; margin-bottom: 8px; }}
        .header .meta {{ opacity: 0.8; font-size: 14px; }}
        .summary-cards {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
        .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1; min-width: 120px; text-align: center; }}
        .card .value {{ font-size: 26px; font-weight: bold; color: #2c3e50; }}
        .card .label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
        .card.success .value {{ color: #27ae60; }}
        .card.alert .value {{ color: #e74c3c; }}
        .card.warn .value {{ color: #e67e22; }}
        .section {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow-x: auto; }}
        .section h2 {{ font-size: 18px; margin-bottom: 15px; color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #f8f9fa; padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6; color: #495057; white-space: nowrap; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #ecf0f1; }}
        tr.win {{ background: #f0fff4; }}
        tr.loss {{ background: #fff5f5; }}
        .win-cell {{ font-weight: bold; text-align: center; }}
        tr.win .win-cell {{ color: #27ae60; }}
        tr.loss .win-cell {{ color: #e74c3c; }}
        .algo-tag {{ background: #ecf0f1; padding: 2px 8px; border-radius: 4px; font-size: 11px; color: #2c3e50; white-space: nowrap; }}
        .footer {{ text-align: center; padding: 20px; color: #95a5a6; font-size: 12px; }}
        .chart-container {{ width: 100%; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>7ETF组合回测报告</h1>
        <div class="meta">
            标的: 红利/沪深300/双创50/黄金股/黄金/石油LOF/标普油气 |
            卖出策略: T+1收盘盈利则卖出, 否则推迟, 最晚T+N卖出 |
            回测区间: {START_DATE} ~ 最新 |
            初始净值: 1.0000 |
            耗时: {duration:.1f}秒
        </div>
    </div>

    <div class="summary-cards">
        <div class="card">
            <div class="value">{total_signals}</div>
            <div class="label">总交易数</div>
        </div>
        <div class="card success">
            <div class="value">{total_win_rate:.1f}%</div>
            <div class="label">胜率(收益>0.5%)</div>
        </div>
        <div class="card warn">
            <div class="value">{avg_ret:+.2f}%</div>
            <div class="label">平均单笔收益</div>
        </div>
        <div class="card">
            <div class="value">{avg_hold:.1f}日</div>
            <div class="label">平均持有天数</div>
        </div>
        <div class="card {('success' if final_nav >= 1 else 'alert')}">
            <div class="value" style="color:{nav_color};">{final_nav:.4f}</div>
            <div class="label">组合最终净值</div>
        </div>
        <div class="card success">
            <div class="value">{cagr:.2f}%</div>
            <div class="label">年化收益(CAGR)</div>
        </div>
        <div class="card alert">
            <div class="value">{max_dd:.1f}%</div>
            <div class="label">最大回撤</div>
        </div>
    </div>

    <div class="section">
        <h2>组合净值曲线</h2>
        <div class="chart-container">
            <svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">
                {y_labels}
                <line x1="{pad_l}" y1="{y_base:.1f}" x2="{svg_w - pad_r}" y2="{y_base:.1f}" stroke="#bdc3c7" stroke-width="1" stroke-dasharray="5,3"/>
                <text x="{pad_l - 5}" y="{y_base + 4:.1f}" text-anchor="end" font-size="11" fill="#95a5a6">1.000</text>
                <path d="{path_d}" fill="none" stroke="{nav_color}" stroke-width="2"/>
                {x_labels}
                <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{svg_h - pad_b}" stroke="#2c3e50" stroke-width="1"/>
                <line x1="{pad_l}" y1="{svg_h - pad_b}" x2="{svg_w - pad_r}" y2="{svg_h - pad_b}" stroke="#2c3e50" stroke-width="1"/>
            </svg>
        </div>
    </div>

    <div class="section">
        <h2>各ETF回测统计</h2>
        <table>
            <tr>
                <th>ETF名称</th>
                <th>算法</th>
                <th>交易数</th>
                <th>胜率</th>
                <th>胜利</th>
                <th>最终净值</th>
                <th>平均收益</th>
                <th>平均持有</th>
            </tr>
            {etf_rows}
        </table>
    </div>

    <div class="section">
        <h2>交易明细（最近50条，按买入日期倒序）</h2>
        <table>
            <tr>
                <th>买入日期</th>
                <th>卖出日期</th>
                <th>ETF</th>
                <th>算法</th>
                <th>买入价</th>
                <th>卖出价</th>
                <th>持有</th>
                <th>信号分</th>
                <th>胜</th>
                <th>收益</th>
            </tr>
            {detail_rows}
        </table>
    </div>

    <div class="footer">
        <p>7ETF组合回测 | T+1盈利卖出策略 | 初始净值1.0000 | 回测起点{START_DATE}</p>
        <p>卖出规则: T+1收盘盈利(收益>0)则卖出, 否则推迟一天, 最晚T+N(3或5)强制卖出</p>
        <p>每只ETF等权分配(1/7资金), 各ETF独立交易, 组合净值=7只ETF净值等权平均</p>
        <p>本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
    </div>
</body>
</html>"""
    return html


def main():
    start_time = datetime.now()

    # 1. 运行回测
    all_trades, target_etfs = run_backtest()

    # 2. 计算净值
    etf_navs, portfolio_nav = calc_nav(all_trades, target_etfs, START_DATE)

    # 3. 生成HTML
    duration = (datetime.now() - start_time).total_seconds()
    html = generate_html(all_trades, etf_navs, portfolio_nav, target_etfs, duration)

    html_path = os.path.join(OUTPUT_DIR, 'focus7_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML报告已生成: {html_path}")

    # 4. 保存JSON
    result = {
        'metadata': {
            'start_date': START_DATE,
            'etf_count': len(target_etfs),
            'initial_nav': 1.0,
            'duration_seconds': round(duration, 1),
        },
        'summary': {
            'total_trades': len(all_trades),
            'total_wins': sum(1 for t in all_trades if t['is_win']),
            'win_rate': round(sum(1 for t in all_trades if t['is_win']) / len(all_trades) * 100, 1) if all_trades else 0,
            'avg_return': round(sum(t['return_pct'] for t in all_trades) / len(all_trades), 2) if all_trades else 0,
            'avg_hold_days': round(sum(t['sell_day'] for t in all_trades) / len(all_trades), 1) if all_trades else 0,
            'final_nav': portfolio_nav[-1]['nav'] if portfolio_nav else 1.0,
            'max_nav': max(p['nav'] for p in portfolio_nav) if portfolio_nav else 1.0,
            'min_nav': min(p['nav'] for p in portfolio_nav) if portfolio_nav else 1.0,
        },
        'etf_navs': {code: {
            'name': data['name'],
            'final_nav': data['final_nav'],
            'trade_count': data['trade_count'],
            'wins': data['wins'],
            'nav_curve': data['nav_curve'],
        } for code, data in etf_navs.items()},
        'portfolio_nav': portfolio_nav,
        'trades': all_trades,
    }
    json_path = os.path.join(OUTPUT_DIR, 'focus7_result.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"JSON结果已保存: {json_path}")

    # 5. 打印摘要
    final_nav = portfolio_nav[-1]['nav'] if portfolio_nav else 1.0
    peak = 1.0
    max_dd = 0
    for p in portfolio_nav:
        if p['nav'] > peak:
            peak = p['nav']
        dd = (p['nav'] / peak - 1) * 100
        if dd < max_dd:
            max_dd = dd

    years = (datetime.now() - datetime.strptime(START_DATE, '%Y-%m-%d')).days / 365.25
    cagr = (final_nav ** (1 / years) - 1) * 100 if years > 0 and final_nav > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"7ETF组合回测摘要 (T+1盈利卖出策略)")
    print(f"{'=' * 60}")
    print(f"总交易数: {len(all_trades)}")
    print(f"总胜率: {sum(1 for t in all_trades if t['is_win']) / len(all_trades) * 100:.1f}%")
    print(f"平均收益: {sum(t['return_pct'] for t in all_trades) / len(all_trades):+.2f}%")
    print(f"平均持有: {sum(t['sell_day'] for t in all_trades) / len(all_trades):.1f}日")
    print(f"组合最终净值: {final_nav:.4f}")
    print(f"年化收益(CAGR): {cagr:.2f}%")
    print(f"最大回撤: {max_dd:.1f}%")

    print(f"\n各ETF净值:")
    for etf in target_etfs:
        data = etf_navs[etf.code]
        tc = data['trade_count']
        wr = data['wins'] / tc * 100 if tc else 0
        print(f"  {data['name']:12s} | 交易:{tc:3d} | 胜率:{wr:.1f}% | 净值:{data['final_nav']:.4f}")


if __name__ == '__main__':
    main()
