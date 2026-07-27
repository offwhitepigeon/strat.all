# -*- coding: utf-8 -*-
"""
B1策略完整回测 - 持有到T+N卖出
================================
7ETF组合, 初始净值1, 等权1/7, 各ETF独立交易
卖出策略: 买入后持有到T+N(3或5)按收盘价卖出

输出: backtest_2024/b1_backtest_report.html
"""
import sys, os, json, logging
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from etf_config import ETF_POOL
from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import get_algorithm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'backtest_2024')
TARGET_CODES = {'sh510880','sh510300','sz159782','sh517520','sh518880','sz162411','sz161129'}
START_DATE = '2024-01-01'
SIGNAL_THRESHOLD = 60


def run_backtest():
    engine = DataEngine()
    target_etfs = [e for e in ETF_POOL if e.code in TARGET_CODES]

    all_trades = []
    etf_navs = {}

    for etf in target_etfs:
        df = engine.get_history_kline(etf.code)
        if df is None or len(df) < 250:
            continue
        df = calc_all_indicators(df)
        algorithm = get_algorithm(etf.algorithm)
        hold_days = getattr(etf, 'hold_days', 3)

        trades = []
        nav = 1.0
        nav_curve = [(START_DATE, 1.0)]

        for i in range(60, len(df) - hold_days):
            date_str = df.iloc[i]['date'].strftime('%Y-%m-%d')
            if date_str < START_DATE:
                continue
            df_slice = df.iloc[:i + 1]
            try:
                signal = algorithm.calculate(df_slice)
            except Exception:
                continue
            if signal.score < SIGNAL_THRESHOLD:
                continue

            buy_price = float(df.iloc[i]['close'])
            future = df.iloc[i + 1: i + 1 + hold_days]
            if len(future) < hold_days:
                continue

            # B1: 持有到T+N收盘卖出
            sell_price = float(future.iloc[-1]['close'])
            sell_date = future.iloc[-1]['date']
            sell_date_str = sell_date.strftime('%Y-%m-%d') if hasattr(sell_date, 'strftime') else str(sell_date)[:10]
            ret = (sell_price / buy_price - 1) * 100
            nav *= (1 + ret / 100)

            # T+N内最高价和最大收益(参考用)
            future_high = float(future['high'].max())
            max_ret = (future_high / buy_price - 1) * 100

            trade = {
                'date': date_str,
                'sell_date': sell_date_str,
                'etf_code': etf.code,
                'etf_name': etf.name,
                'algorithm': etf.algorithm,
                'hold_days': hold_days,
                'buy_price': round(buy_price, 4),
                'sell_price': round(sell_price, 4),
                'score': round(signal.score, 1),
                'level': signal.level,
                'action': signal.action,
                'return_pct': round(ret, 2),
                'is_win': ret > 0.5,
                'max_return': round(max_ret, 2),
                'future_high': round(future_high, 4),
                'reasons': signal.reasons[:3] if signal.reasons else [],
            }
            trades.append(trade)
            nav_curve.append((sell_date_str, round(nav, 4)))

        etf_navs[etf.code] = {
            'name': etf.name,
            'algorithm': etf.algorithm,
            'hold_days': hold_days,
            'final_nav': round(nav, 4),
            'trade_count': len(trades),
            'wins': sum(1 for t in trades if t['is_win']),
            'nav_curve': nav_curve,
        }
        all_trades.extend(trades)
        if trades:
            wr = sum(1 for t in trades if t['is_win']) / len(trades) * 100
            avg_r = sum(t['return_pct'] for t in trades) / len(trades)
            logger.info(f"{etf.name:12s} | 交易:{len(trades):3d} | 胜率:{wr:.1f}% | 均收益:{avg_r:+.2f}% | 净值:{nav:.4f}")

    # 组合净值 = 等权平均
    n_etfs = len(target_etfs)
    all_dates = set()
    for data in etf_navs.values():
        for d, _ in data['nav_curve']:
            all_dates.add(d)
    all_dates = sorted(all_dates)

    portfolio_nav = []
    for date_str in all_dates:
        navs = []
        for etf in target_etfs:
            data = etf_navs.get(etf.code, {})
            etf_nav = 1.0
            for d, n in data.get('nav_curve', []):
                if d <= date_str:
                    etf_nav = n
                else:
                    break
            navs.append(etf_nav)
        portfolio_nav.append({'date': date_str, 'nav': round(sum(navs) / n_etfs, 4)})

    return all_trades, etf_navs, portfolio_nav, target_etfs


def calc_metrics(trades, portfolio_nav):
    n = len(trades)
    wins = sum(1 for t in trades if t['is_win'])
    rets = [t['return_pct'] for t in trades]
    holds = [t['hold_days'] for t in trades]
    final_nav = portfolio_nav[-1]['nav'] if portfolio_nav else 1.0

    # 最大回撤
    peak = 1.0
    max_dd = 0
    for p in portfolio_nav:
        if p['nav'] > peak:
            peak = p['nav']
        dd = (p['nav'] / peak - 1) * 100
        if dd < max_dd:
            max_dd = dd

    # 年化
    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_dt = datetime.now()
    years = max((end_dt - start_dt).days / 365.25, 0.01)
    cagr = (final_nav ** (1 / years) - 1) * 100 if final_nav > 0 else -100

    # Sharpe
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(n)
    else:
        sharpe = 0

    return {
        'total_trades': n, 'wins': wins, 'win_rate': round(wins / n * 100, 1) if n else 0,
        'avg_return': round(np.mean(rets), 2) if rets else 0,
        'avg_hold': round(np.mean(holds), 1) if holds else 0,
        'final_nav': round(final_nav, 4),
        'cagr': round(cagr, 2), 'max_dd': round(max_dd, 1),
        'sharpe': round(sharpe, 2),
        'total_return': round((final_nav - 1) * 100, 2),
    }


def generate_html(trades, etf_navs, portfolio_nav, target_etfs, metrics, duration):
    n_etfs = len(target_etfs)

    # --- 净值曲线SVG ---
    nav_values = [p['nav'] for p in portfolio_nav]
    dates = [p['date'] for p in portfolio_nav]
    n_points = len(nav_values)

    svg_w, svg_h = 920, 400
    pad_l, pad_r, pad_t, pad_b = 65, 25, 25, 45
    plot_w = svg_w - pad_l - pad_r
    plot_h = svg_h - pad_t - pad_b

    nav_min = min(min(nav_values), 0.95)
    nav_max = max(max(nav_values), 1.10)
    y_range = nav_max - nav_min

    if n_points > 250:
        step = n_points // 250
        sampled = list(range(0, n_points, step)) + [n_points - 1]
    else:
        sampled = list(range(n_points))

    # 净值曲线path
    pts = []
    for idx in sampled:
        x = pad_l + (idx / max(n_points - 1, 1)) * plot_w
        y = pad_t + (1 - (nav_values[idx] - nav_min) / y_range) * plot_h
        pts.append(f"{x:.1f},{y:.1f}")
    path_d = "M " + " L ".join(pts)

    # 填充区域
    fill_pts = pts[:]
    x_last = pad_l + (sampled[-1] / max(n_points - 1, 1)) * plot_w
    x_first = pad_l + (sampled[0] / max(n_points - 1, 1)) * plot_w
    fill_pts.append(f"{x_last:.1f},{pad_t + plot_h:.1f}")
    fill_pts.append(f"{x_first:.1f},{pad_t + plot_h:.1f}")
    fill_d = "M " + " L ".join(fill_pts) + " Z"

    # 基准线
    y_base = pad_t + (1 - (1.0 - nav_min) / y_range) * plot_h

    # Y轴标签
    y_labels = ""
    for v in [nav_min, nav_min + y_range * 0.25, nav_min + y_range * 0.5, nav_min + y_range * 0.75, nav_max]:
        y = pad_t + (1 - (v - nav_min) / y_range) * plot_h
        y_labels += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{svg_w - pad_r}" y2="{y:.1f}" stroke="#ecf0f1" stroke-width="1"/><text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#7f8c8d">{v:.3f}</text>'

    # X轴标签
    x_labels = ""
    seen = set()
    for idx in sampled:
        d = dates[idx]
        m = d[:7]
        if m not in seen:
            seen.add(m)
            x = pad_l + (idx / max(n_points - 1, 1)) * plot_w
            x_labels += f'<text x="{x:.1f}" y="{svg_h - 15}" text-anchor="middle" font-size="10" fill="#7f8c8d">{d[2:7]}</text>'

    nav_color = '#27ae60' if metrics['final_nav'] >= 1 else '#e74c3c'

    # --- 各ETF统计表 ---
    etf_rows = ""
    for etf in target_etfs:
        data = etf_navs.get(etf.code, {})
        tc = data.get('trade_count', 0)
        w = data.get('wins', 0)
        wr = w / tc * 100 if tc else 0
        fn = data.get('final_nav', 1.0)
        etf_trades = [t for t in trades if t['etf_code'] == etf.code]
        avg_r = sum(t['return_pct'] for t in etf_trades) / tc if tc else 0
        avg_s = sum(t['score'] for t in etf_trades) / tc if tc else 0
        best = max(t['return_pct'] for t in etf_trades) if etf_trades else 0
        worst = min(t['return_pct'] for t in etf_trades) if etf_trades else 0
        wr_color = '#27ae60' if wr >= 60 else ('#e67e22' if wr >= 45 else '#e74c3c')
        nav_c = '#27ae60' if fn >= 1 else '#e74c3c'
        etf_rows += f"""
            <tr>
                <td><strong>{data.get('name', etf.name)}</strong></td>
                <td><span class="algo-tag">{data.get('algorithm', '')}</span></td>
                <td>T+{data.get('hold_days', 3)}</td>
                <td>{tc}</td>
                <td style="color:{wr_color};font-weight:bold;">{wr:.1f}%</td>
                <td>{w}</td>
                <td style="color:{nav_c};font-weight:bold;">{fn:.4f}</td>
                <td style="color:{'#27ae60' if avg_r >= 0 else '#e74c3c'};">{avg_r:+.2f}%</td>
                <td>{avg_s:.1f}</td>
                <td style="color:#27ae60;">{best:+.2f}%</td>
                <td style="color:#e74c3c;">{worst:+.2f}%</td>
            </tr>"""

    # --- 信号明细(最近60条) ---
    detail_rows = ""
    for t in sorted(trades, key=lambda x: x['date'], reverse=True)[:60]:
        is_win = t['is_win']
        win_class = 'win' if is_win else 'loss'
        win_text = '✓' if is_win else '✗'
        ret_color = '#27ae60' if t['return_pct'] >= 0 else '#e74c3c'
        max_color = '#27ae60' if t['max_return'] >= 0 else '#e74c3c'
        reasons = '<br>'.join(f'• {r}' for r in t.get('reasons', [])) if t.get('reasons') else '—'
        detail_rows += f"""
            <tr class="{win_class}">
                <td>{t['date']}</td>
                <td>{t['sell_date']}</td>
                <td><strong>{t['etf_name']}</strong></td>
                <td><span class="algo-tag">{t['algorithm']}</span></td>
                <td>T+{t['hold_days']}</td>
                <td>{t['buy_price']:.3f}</td>
                <td>{t['sell_price']:.3f}</td>
                <td style="font-weight:bold;">{t['score']:.1f}</td>
                <td class="win-cell">{win_text}</td>
                <td style="color:{max_color};">{t['max_return']:+.2f}%</td>
                <td style="color:{ret_color};font-weight:bold;">{t['return_pct']:+.2f}%</td>
                <td class="reasons-cell">{reasons}</td>
            </tr>"""

    # --- 月度统计 ---
    monthly = defaultdict(lambda: {'trades': 0, 'wins': 0, 'returns': [], 'nav_start': 0, 'nav_end': 0})
    for t in trades:
        m = t['date'][:7]
        monthly[m]['trades'] += 1
        if t['is_win']:
            monthly[m]['wins'] += 1
        monthly[m]['returns'].append(t['return_pct'])

    # 计算每月净值
    for p in portfolio_nav:
        m = p['date'][:7]
        if monthly[m]['nav_start'] == 0:
            monthly[m]['nav_start'] = p['nav']
        monthly[m]['nav_end'] = p['nav']

    month_rows = ""
    for m in sorted(monthly.keys()):
        ms = monthly[m]
        wr = ms['wins'] / ms['trades'] * 100 if ms['trades'] else 0
        avg_r = sum(ms['returns']) / len(ms['returns']) if ms['returns'] else 0
        mret = (ms['nav_end'] / ms['nav_start'] - 1) * 100 if ms['nav_start'] else 0
        wr_color = '#27ae60' if wr >= 60 else ('#e67e22' if wr >= 45 else '#e74c3c')
        month_rows += f"""
            <tr>
                <td><strong>{m}</strong></td>
                <td>{ms['trades']}</td>
                <td style="color:{wr_color};font-weight:bold;">{wr:.1f}%</td>
                <td style="color:{'#27ae60' if avg_r >= 0 else '#e74c3c'};">{avg_r:+.2f}%</td>
                <td style="color:{'#27ae60' if mret >= 0 else '#e74c3c'};">{mret:+.2f}%</td>
                <td>{ms['nav_end']:.4f}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>B1策略回测报告 | 持有到T+N卖出</title>
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
        .reasons-cell {{ font-size: 11px; color: #555; max-width: 250px; }}
        .chart-container {{ width: 100%; overflow-x: auto; }}
        .footer {{ text-align: center; padding: 20px; color: #95a5a6; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>B1策略回测报告 — 持有到T+N卖出</h1>
        <div class="meta">
            标的: 红利/沪深300/双创50/黄金股/黄金/石油LOF/标普油气(7只) |
            卖出策略: 买入后持有到T+N(T+3或T+5)按收盘价卖出 |
            回测区间: {START_DATE} ~ 最新 |
            初始净值: 1.0000 | 每ETF等权1/7 | 耗时: {duration:.1f}秒
        </div>
    </div>

    <div class="summary-cards">
        <div class="card">
            <div class="value">{metrics['total_trades']}</div>
            <div class="label">总交易数</div>
        </div>
        <div class="card success">
            <div class="value">{metrics['win_rate']:.1f}%</div>
            <div class="label">胜率(收益>0.5%)</div>
        </div>
        <div class="card warn">
            <div class="value">{metrics['avg_return']:+.2f}%</div>
            <div class="label">平均单笔收益</div>
        </div>
        <div class="card">
            <div class="value">T+{metrics['avg_hold']:.0f}</div>
            <div class="label">平均持有天数</div>
        </div>
        <div class="card success">
            <div class="value" style="color:{nav_color};">{metrics['final_nav']:.4f}</div>
            <div class="label">组合最终净值</div>
        </div>
        <div class="card success">
            <div class="value">{metrics['cagr']:.2f}%</div>
            <div class="label">年化收益(CAGR)</div>
        </div>
        <div class="card alert">
            <div class="value">{metrics['max_dd']:.1f}%</div>
            <div class="label">最大回撤</div>
        </div>
        <div class="card">
            <div class="value">{metrics['sharpe']:.2f}</div>
            <div class="label">Sharpe</div>
        </div>
    </div>

    <div class="section">
        <h2>组合净值曲线 (初始净值=1.0000)</h2>
        <div class="chart-container">
            <svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">
                <defs><linearGradient id="navGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="{nav_color}" stop-opacity="0.15"/>
                    <stop offset="100%" stop-color="{nav_color}" stop-opacity="0.02"/>
                </linearGradient></defs>
                {y_labels}
                <line x1="{pad_l}" y1="{y_base:.1f}" x2="{svg_w - pad_r}" y2="{y_base:.1f}" stroke="#bdc3c7" stroke-width="1" stroke-dasharray="5,3"/>
                <text x="{pad_l - 8}" y="{y_base + 4:.1f}" text-anchor="end" font-size="11" fill="#95a5a6">1.000</text>
                <path d="{fill_d}" fill="url(#navGrad)"/>
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
                <th>ETF名称</th><th>算法</th><th>持有</th><th>交易数</th><th>胜率</th>
                <th>胜利</th><th>最终净值</th><th>平均收益</th><th>平均分</th>
                <th>最佳</th><th>最差</th>
            </tr>
            {etf_rows}
        </table>
    </div>

    <div class="section">
        <h2>月度统计</h2>
        <table>
            <tr><th>月份</th><th>交易数</th><th>胜率</th><th>平均收益</th><th>月度净值变化</th><th>月末净值</th></tr>
            {month_rows}
        </table>
    </div>

    <div class="section">
        <h2>信号明细（最近60条，按买入日期倒序）</h2>
        <table>
            <tr>
                <th>买入日</th><th>卖出日</th><th>ETF</th><th>算法</th><th>持有</th>
                <th>买入价</th><th>卖出价</th><th>信号分</th><th>胜</th>
                <th>期间最大</th><th>实际收益</th><th>信号理由</th>
            </tr>
            {detail_rows}
        </table>
    </div>

    <div class="footer">
        <p>B1策略回测 | 持有到T+N卖出 | 7ETF组合 | 回测起点{START_DATE}</p>
        <p>每只ETF等权(1/7资金), 各ETF独立交易, 组合净值=7只ETF净值等权平均</p>
        <p>本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
    </div>
</body>
</html>"""
    return html


def main():
    start_time = datetime.now()
    trades, etf_navs, portfolio_nav, target_etfs = run_backtest()
    metrics = calc_metrics(trades, portfolio_nav)
    duration = (datetime.now() - start_time).total_seconds()

    html = generate_html(trades, etf_navs, portfolio_nav, target_etfs, metrics, duration)
    html_path = os.path.join(OUTPUT_DIR, 'b1_backtest_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    json_path = os.path.join(OUTPUT_DIR, 'b1_backtest_result.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {'strategy': 'B1_持有到T+N', 'start_date': START_DATE, 'duration': round(duration, 1)},
            'metrics': metrics,
            'etf_navs': {k: {kk: vv for kk, vv in v.items() if kk != 'nav_curve'} for k, v in etf_navs.items()},
            'portfolio_nav': portfolio_nav,
            'trades': trades,
        }, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"HTML: {html_path}")
    logger.info(f"JSON: {json_path}")

    print(f"\n{'=' * 70}")
    print(f"B1策略回测 (持有到T+N) | 耗时 {duration:.1f}秒")
    print(f"{'=' * 70}")
    print(f"总交易: {metrics['total_trades']} | 胜率: {metrics['win_rate']:.1f}% | 均收益: {metrics['avg_return']:+.2f}%")
    print(f"净值: {metrics['final_nav']:.4f} | CAGR: {metrics['cagr']:.2f}% | 回撤: {metrics['max_dd']:.1f}% | Sharpe: {metrics['sharpe']:.2f}")
    print(f"\n各ETF:")
    for etf in target_etfs:
        d = etf_navs.get(etf.code, {})
        tc = d.get('trade_count', 0)
        wr = d.get('wins', 0) / tc * 100 if tc else 0
        print(f"  {d.get('name',''):12s} | 交易:{tc:3d} | 胜率:{wr:.1f}% | 净值:{d.get('final_nav',1):.4f}")


if __name__ == '__main__':
    main()
