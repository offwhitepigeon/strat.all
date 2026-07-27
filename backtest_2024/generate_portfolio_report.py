# -*- coding: utf-8 -*-
"""
ETF Portfolio Backtest Report Generator
========================================
Runs full backtest with current 20 algorithms, simulates portfolio NAV,
generates HTML report matching backtest_report.html style + new portfolio section.

Portfolio model (capital management):
- Start: 2024-01-01, NAV = 1.0, cash = 1.0
- Each signal: invest position_pct% of current NAV (capped by available capital)
- Positions tracked as "lots", each with entry_date, exit_date (T+N), invested amount
- Capital freed when lot exits at T+N close
- Same ETF: position < 50% allows adding (pyramiding)
- Total portfolio: max 100% invested
- NAV compounds as positions close and P&L is realized
"""
import sys, os, json, logging, math
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backtest_engine import BacktestEngine
from etf_config import ETF_POOL

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGNAL_LEVELS = {
    'WAIT':       (0, 40,  0),
    'WATCH':      (40, 60, 5),
    'LIGHT_BUY':  (60, 75, 15),
    'BUY':        (75, 85, 30),
    'STRONG_BUY': (85, 101, 50),
}

MAX_SINGLE_ETF_PCT = 50  # 单只ETF最大仓位%
MAX_TOTAL_PCT = 100      # 组合最大总仓位%


def get_position_pct(level):
    for lvl, (lo, hi, pos) in SIGNAL_LEVELS.items():
        if lvl == level:
            return pos
    return 0


def simulate_portfolio(signals):
    """
    Simulate portfolio with capital management.

    Rules:
    - NAV starts at 1.0, cash = 1.0 (available capital)
    - Each signal invests position_pct% of NAV, subject to:
      - Single ETF position < 50% (allows pyramiding below cap)
      - Total portfolio position < 100%
      - Available cash >= investment amount
    - Positions exit at T+N (exit_date from backtest), capital released
    - P&L realized at exit, NAV updated
    """
    # Build ETF hold_days lookup
    etf_hold_days = {etf.code: getattr(etf, 'hold_days', 3) for etf in ETF_POOL}

    # Sort signals by date, then by score descending (higher score = higher priority)
    sorted_signals = sorted(signals, key=lambda x: (x['date'], -x['score']))

    nav = 1.0          # Total portfolio NAV (updated on position close)
    locked_capital = 0.0  # Capital locked in open positions (absolute amount)

    # Open lots: list of dicts
    # Each lot: {etf_code, etf_name, entry_date, exit_date, position_pct,
    #            invest_amount, return_pct, algorithm, score, level, buy_price, is_win}
    open_lots = []

    trades = []
    etf_stats = defaultdict(lambda: {'signals': 0, 'wins': 0, 'pnl': 0.0, 'invested': 0.0, 'trades': []})
    skipped_signals = 0

    for sig in sorted_signals:
        current_date = sig['date']

        # 1. Close expired positions (exit_date <= current_date)
        remaining = []
        for lot in open_lots:
            if lot['exit_date'] <= current_date:
                # Close position: realize P&L, release capital
                ret = lot['return_pct'] / 100.0
                pnl = lot['invest_amount'] * ret
                nav += pnl
                locked_capital -= lot['invest_amount']

                trade = {
                    'date': lot['entry_date'],
                    'exit_date': lot['exit_date'],
                    'etf_code': lot['etf_code'],
                    'etf_name': lot['etf_name'],
                    'algorithm': lot.get('algorithm', ''),
                    'score': lot.get('score', 0),
                    'level': lot.get('level', ''),
                    'position_pct': lot['position_pct'],
                    'buy_price': lot.get('buy_price', 0),
                    'invest_amount': round(lot['invest_amount'], 6),
                    'return_pct': lot['return_pct'],
                    'pnl': round(pnl, 6),
                    'nav_after': round(nav, 6),
                    'is_win': lot.get('is_win', False),
                    'is_pyramiding': lot.get('is_pyramiding', False),
                }
                trades.append(trade)

                s = etf_stats[lot['etf_name']]
                s['signals'] += 1
                s['wins'] += 1 if lot.get('is_win') else 0
                s['pnl'] += pnl
                s['invested'] += lot['invest_amount']
                s['trades'].append(trade)
            else:
                remaining.append(lot)
        open_lots = remaining

        # 2. Try to open new position
        pos_pct = get_position_pct(sig['level'])
        if pos_pct == 0:
            continue

        etf_code = sig['etf_code']

        # Check same-ETF position (sum of open lots for this ETF)
        etf_position_pct = sum(l['position_pct'] for l in open_lots if l['etf_code'] == etf_code)
        if etf_position_pct >= MAX_SINGLE_ETF_PCT:
            skipped_signals += 1
            continue

        # Check total portfolio position
        total_position_pct = sum(l['position_pct'] for l in open_lots)
        if total_position_pct >= MAX_TOTAL_PCT:
            skipped_signals += 1
            continue

        # Cap the new position: min of signal position, room in ETF, room in portfolio
        max_new_pct = min(pos_pct, MAX_SINGLE_ETF_PCT - etf_position_pct, MAX_TOTAL_PCT - total_position_pct)
        if max_new_pct <= 0:
            skipped_signals += 1
            continue

        # Available cash (as fraction of NAV)
        available = nav - locked_capital
        if available <= 0:
            skipped_signals += 1
            continue

        # Investment amount = min(desired, available)
        desired_invest = nav * max_new_pct / 100.0
        invest = min(desired_invest, available)
        if invest <= 0:
            skipped_signals += 1
            continue

        # Actual position_pct (may be less if cash-limited)
        actual_pos_pct = invest / nav * 100.0

        # Determine if this is pyramiding (adding to existing position)
        is_pyramiding = etf_position_pct > 0

        # Open new lot
        lot = {
            'etf_code': etf_code,
            'etf_name': sig['etf_name'],
            'entry_date': current_date,
            'exit_date': sig.get('exit_date', current_date),
            'position_pct': actual_pos_pct,
            'invest_amount': invest,
            'return_pct': sig['return_3d'],
            'algorithm': sig['algorithm'],
            'score': sig['score'],
            'level': sig['level'],
            'buy_price': sig['buy_price'],
            'is_win': sig['is_win'],
            'is_pyramiding': is_pyramiding,
        }
        open_lots.append(lot)
        locked_capital += invest

    # Close remaining open positions at end of period
    for lot in open_lots:
        ret = lot['return_pct'] / 100.0
        pnl = lot['invest_amount'] * ret
        nav += pnl
        locked_capital -= lot['invest_amount']

        trade = {
            'date': lot['entry_date'],
            'exit_date': lot['exit_date'],
            'etf_code': lot['etf_code'],
            'etf_name': lot['etf_name'],
            'algorithm': lot.get('algorithm', ''),
            'score': lot.get('score', 0),
            'level': lot.get('level', ''),
            'position_pct': lot['position_pct'],
            'buy_price': lot.get('buy_price', 0),
            'invest_amount': round(lot['invest_amount'], 6),
            'return_pct': lot['return_pct'],
            'pnl': round(pnl, 6),
            'nav_after': round(nav, 6),
            'is_win': lot.get('is_win', False),
            'is_pyramiding': lot.get('is_pyramiding', False),
        }
        trades.append(trade)

        s = etf_stats[lot['etf_name']]
        s['signals'] += 1
        s['wins'] += 1 if lot.get('is_win') else 0
        s['pnl'] += pnl
        s['invested'] += lot['invest_amount']
        s['trades'].append(trade)

    # Sort trades by exit_date for NAV curve
    trades.sort(key=lambda x: x['exit_date'])

    # Calculate metrics
    final_nav = nav
    total_return = (final_nav - 1.0) * 100

    # Max drawdown (using NAV after each closed trade)
    peak = 1.0
    max_dd = 0.0
    for t in trades:
        if t['nav_after'] > peak:
            peak = t['nav_after']
        dd = (t['nav_after'] - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # Sharpe-like ratio (per-trade)
    pnls = [t['pnl'] for t in trades]
    if len(pnls) > 1:
        import numpy as np
        std = np.std(pnls)
        sharpe = (np.mean(pnls) / std * math.sqrt(len(pnls))) if std > 0 else 0
    else:
        sharpe = 0

    # CAGR
    years = (datetime(2026, 7, 22) - datetime(2024, 1, 1)).days / 365.25
    cagr = ((final_nav) ** (1 / years) - 1) * 100 if final_nav > 0 else 0

    # Win rate in portfolio
    win_trades = sum(1 for t in trades if t['is_win'])
    win_rate = win_trades / len(trades) * 100 if trades else 0

    # Pyramiding stats
    pyramid_trades = sum(1 for t in trades if t.get('is_pyramiding', False))

    return {
        'trades': trades,
        'etf_stats': dict(etf_stats),
        'final_nav': final_nav,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'cagr': cagr,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'win_trades': win_trades,
        'skipped_signals': skipped_signals,
        'pyramid_trades': pyramid_trades,
    }


def build_nav_svg(trades, width=900, height=300):
    """Build SVG NAV curve chart"""
    if not trades:
        return "<p>No trades</p>"

    # Extract NAV points
    points = [(t['date'], t['nav_after']) for t in trades]
    # Prepend starting point
    points.insert(0, ('2024-01-01', 1.0))

    n = len(points)
    margin_l, margin_r, margin_t, margin_b = 60, 30, 30, 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    # Y range
    navs = [p[1] for p in points]
    y_min = min(navs) * 0.98
    y_max = max(navs) * 1.02
    if y_max == y_min:
        y_max = y_min + 0.1

    def x_pos(i):
        return margin_l + (i / max(n - 1, 1)) * plot_w

    def y_pos(val):
        return margin_t + (1 - (val - y_min) / (y_max - y_min)) * plot_h

    # Build path
    path_d = ""
    for i, (date, nav) in enumerate(points):
        x = x_pos(i)
        y = y_pos(nav)
        path_d += f"L{x:.1f},{y:.1f} " if i > 0 else f"M{x:.1f},{y:.1f} "

    # Grid lines
    grid_lines = ""
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_t + frac * plot_h
        val = y_max - frac * (y_max - y_min)
        grid_lines += f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" stroke="#ecf0f1" stroke-width="1"/>'
        grid_lines += f'<text x="{margin_l-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#7f8c8d">{val:.3f}</text>'

    # X-axis labels (quarterly)
    x_labels = ""
    quarters = []
    for i, (date, nav) in enumerate(points):
        if date[5:7] in ['01', '04', '07', '10'] and date[:4] >= '2024':
            label = f"{date[:4]}-{date[5:7]}"
            if label not in [q[1] for q in quarters]:
                quarters.append((i, label))
    for i, label in quarters[:12]:
        x = x_pos(i)
        x_labels += f'<text x="{x:.1f}" y="{height-15}" text-anchor="middle" font-size="11" fill="#7f8c8d">{label}</text>'

    # Fill area under curve
    fill_path = path_d + f"L{x_pos(n-1):.1f},{y_pos(y_min):.1f} L{x_pos(0):.1f},{y_pos(y_min):.1f} Z"

    # Baseline at NAV=1.0
    baseline_y = y_pos(1.0)

    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;">
        <defs>
            <linearGradient id="navGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#3498db" stop-opacity="0.3"/>
                <stop offset="100%" stop-color="#3498db" stop-opacity="0.02"/>
            </linearGradient>
        </defs>
        {grid_lines}
        <path d="{fill_path}" fill="url(#navGrad)"/>
        <line x1="{margin_l}" y1="{baseline_y:.1f}" x2="{width-margin_r}" y2="{baseline_y:.1f}" stroke="#e74c3c" stroke-width="1" stroke-dasharray="4,4" opacity="0.5"/>
        <text x="{margin_l-8}" y="{baseline_y+4:.1f}" text-anchor="end" font-size="11" fill="#e74c3c">1.000</text>
        <path d="{path_d}" fill="none" stroke="#3498db" stroke-width="2"/>
        {x_labels}
        <text x="{width/2:.0f}" y="{height-2}" text-anchor="middle" font-size="12" fill="#95a5a6">Signal Date</text>
    </svg>'''
    return svg


def build_etf_contribution_svg(etf_stats, width=900, height=400):
    """Build SVG bar chart for per-ETF P&L contribution"""
    if not etf_stats:
        return "<p>No data</p>"

    # Sort by P&L
    items = sorted(etf_stats.items(), key=lambda x: x[1]['pnl'])
    n = len(items)
    margin_l, margin_r, margin_t, margin_b = 120, 30, 30, 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    max_abs = max(abs(s['pnl']) for _, s in items) or 0.01

    bar_h = plot_h / n * 0.7
    gap = plot_h / n * 0.3

    bars = ""
    for i, (name, stats) in enumerate(items):
        y = margin_t + i * (bar_h + gap) + gap / 2
        pnl = stats['pnl']
        bar_w = abs(pnl) / max_abs * plot_w * 0.45
        if pnl >= 0:
            x = margin_l + plot_w * 0.45
            color = '#27ae60'
        else:
            x = margin_l + plot_w * 0.45 - bar_w
            color = '#e74c3c'
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2"/>'
        bars += f'<text x="{margin_l-8}" y="{y+bar_h/2+4:.1f}" text-anchor="end" font-size="11" fill="#2c3e50">{name}</text>'
        pnl_text = f"{pnl*100:+.2f}%"
        tx = x + bar_w + 5 if pnl >= 0 else x - 5
        ta = "start" if pnl >= 0 else "end"
        bars += f'<text x="{tx:.1f}" y="{y+bar_h/2+4:.1f}" text-anchor="{ta}" font-size="11" fill="{color}" font-weight="bold">{pnl_text}</text>'

    # Center line
    center_x = margin_l + plot_w * 0.45
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;">
        <line x1="{center_x:.1f}" y1="{margin_t}" x2="{center_x:.1f}" y2="{margin_t+plot_h}" stroke="#bdc3c7" stroke-width="1"/>
        {bars}
    </svg>'''
    return svg


def generate_html_report(backtest_result, portfolio):
    """Generate full HTML report with portfolio section"""
    meta = backtest_result['metadata']
    summary = backtest_result['summary']
    stats_by_etf = backtest_result['stats_by_etf']
    signals = backtest_result['signals']

    algo_stats = summary.get('by_algorithm', {})
    level_stats = summary.get('by_level', {})
    score_bins = summary.get('by_score_bin', {})

    # --- Build existing report sections (same as run_backtest.py) ---
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

    monthly_stats = {}
    for sig in signals:
        month = sig['date'][:7]
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

    # --- Build NEW portfolio section ---
    p = portfolio
    nav_color = '#27ae60' if p['total_return'] >= 0 else '#e74c3c'
    dd_color = '#e74c3c'
    sharpe_color = '#27ae60' if p['sharpe'] >= 1 else ('#e67e22' if p['sharpe'] >= 0 else '#e74c3c')
    cagr_color = '#27ae60' if p['cagr'] >= 10 else ('#e67e22' if p['cagr'] >= 0 else '#e74c3c')

    # Portfolio summary cards
    port_cards = f"""
        <div class="card {'success' if p['total_return'] >= 0 else 'alert'}">
            <div class="value" style="color:{nav_color};">{p['final_nav']:.4f}</div>
            <div class="label">最终净值</div>
        </div>
        <div class="card {'success' if p['total_return'] >= 0 else 'alert'}">
            <div class="value" style="color:{nav_color};">{p['total_return']:+.2f}%</div>
            <div class="label">总收益率</div>
        </div>
        <div class="card">
            <div class="value" style="color:{cagr_color};">{p['cagr']:+.2f}%</div>
            <div class="label">年化收益(CAGR)</div>
        </div>
        <div class="card alert">
            <div class="value" style="color:{dd_color};">{p['max_drawdown']:.2f}%</div>
            <div class="label">最大回撤</div>
        </div>
        <div class="card">
            <div class="value" style="color:{sharpe_color};">{p['sharpe']:.3f}</div>
            <div class="label">Sharpe比率</div>
        </div>
        <div class="card {'success' if p['win_rate'] >= 75 else 'warn'}">
            <div class="value">{p['win_rate']:.1f}%</div>
            <div class="label">组合胜率({p['win_trades']}/{p['total_trades']})</div>
        </div>
        <div class="card">
            <div class="value" style="color:#9b59b6;">{p.get('pyramid_trades', 0)}</div>
            <div class="label">金字塔加仓次数</div>
        </div>"""

    # NAV curve SVG
    nav_svg = build_nav_svg(p['trades'])

    # Per-ETF contribution table
    etf_contrib_rows = ""
    for name, stats in sorted(p['etf_stats'].items(), key=lambda x: -x[1]['pnl']):
        s = stats
        wr = s['wins'] / s['signals'] * 100 if s['signals'] > 0 else 0
        wr_color = '#27ae60' if wr >= 75 else ('#e67e22' if wr >= 60 else '#e74c3c')
        pnl_pct = s['pnl'] * 100
        pnl_color = '#27ae60' if pnl_pct >= 0 else '#e74c3c'
        etf_contrib_rows += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{s['signals']}</td>
                <td>{s['wins']}</td>
                <td style="color:{wr_color};font-weight:bold;">{wr:.1f}%</td>
                <td style="color:{pnl_color};font-weight:bold;">{pnl_pct:+.3f}%</td>
                <td>{s['invested']*100:.2f}</td>
                <td style="color:{pnl_color};">{(s['pnl']/s['invested']*100) if s['invested']>0 else 0:+.2f}%</td>
            </tr>"""

    # ETF contribution SVG
    contrib_svg = build_etf_contribution_svg(p['etf_stats'])

    # Portfolio trade detail (recent 30)
    port_detail_rows = ""
    recent_trades = sorted(p['trades'], key=lambda x: x.get('exit_date', x['date']), reverse=True)[:30]
    for t in recent_trades:
        pnl_color = '#27ae60' if t['pnl'] >= 0 else '#e74c3c'
        win_class = 'win' if t['is_win'] else 'loss'
        pyramid_tag = ' <span class="algo-tag" style="background:#9b59b6;color:white;">加仓</span>' if t.get('is_pyramiding') else ''
        port_detail_rows += f"""
            <tr class="{win_class}">
                <td>{t['date']}</td>
                <td>{t.get('exit_date', '')}</td>
                <td><strong>{t['etf_name']}</strong></td>
                <td><span class="algo-tag">{t['algorithm']}</span></td>
                <td>{t['score']:.1f}</td>
                <td><span class="level-badge level-{t['level'].lower()}">{t['level']}</span>{pyramid_tag}</td>
                <td>{t['position_pct']:.1f}%</td>
                <td>{t['buy_price']:.3f}</td>
                <td style="color:{pnl_color};">{t['return_pct']:+.2f}%</td>
                <td style="color:{pnl_color};font-weight:bold;">{t['pnl']*100:+.4f}%</td>
                <td style="font-weight:bold;">{t['nav_after']:.4f}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF多策略回测报告（含组合模拟） | 2024-01-01起</title>
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
        .section .note {{ font-size: 12px; color: #95a5a6; margin-top: 10px; }}
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
        .chart-container {{
            background: white; border-radius: 8px; padding: 15px;
            margin: 10px 0; text-align: center;
        }}
        .footer {{
            text-align: center; padding: 20px; color: #95a5a6;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ETF多策略回测报告（含组合模拟）</h1>
        <div class="meta">
            回测区间: 2024-01-01 ~ 2026-07-22 |
            ETF数量: {meta['etf_count']}只 |
            算法数量: {len(algo_stats)}种 |
            信号阈值: &ge;60分 |
            资金管理: 单仓≤50% / 总仓≤100% / T+N释放资金 |
            胜率定义: T+3/T+5最高价收益 &gt; 0.5% |
            回测耗时: {meta['duration_seconds']}秒
        </div>
    </div>

    <div class="summary-cards">
        <div class="card">
            <div class="value">{summary['total_signals']}</div>
            <div class="label">总信号数</div>
        </div>
        <div class="card success">
            <div class="value" style="color:{total_wr_color};">{summary['total_win_rate']:.1f}%</div>
            <div class="label">总胜率</div>
        </div>
        <div class="card success">
            <div class="value">{summary['total_wins']}</div>
            <div class="label">胜利次数</div>
        </div>
        <div class="card warn">
            <div class="value">{summary['avg_return_3d']:+.2f}%</div>
            <div class="label">平均T+N收盘收益</div>
        </div>
        <div class="card success">
            <div class="value">{summary['avg_max_return']:+.2f}%</div>
            <div class="label">平均T+N最大收益</div>
        </div>
    </div>

    <!-- ===== NEW: Portfolio Simulation Section ===== -->
    <div class="header" style="background: linear-gradient(135deg, #1a1a2e, #2d3561); margin-top: 10px;">
        <h1 style="font-size: 20px;">组合净值模拟（资金管理）</h1>
        <div class="meta">
            初始净值: 1.0000 (2024-01-01) |
            仓位规则: LIGHT_BUY=15% / BUY=30% / STRONG_BUY=50% |
            单仓上限: 50% | 总仓上限: 100% |
            资金释放: T+N卖出后释放（T+3或T+5） |
            加仓规则: 单仓<50%时可补仓（金字塔加仓） |
            卖出: T+N收盘价 |
            复利: 仓位平仓时实现收益
        </div>
    </div>

    <div class="summary-cards">
        {port_cards}
    </div>

    <div class="section">
        <h2>净值曲线</h2>
        <div class="chart-container">
            {nav_svg}
        </div>
        <p class="note">
            说明：净值按仓位平仓日逐笔更新。每笔信号按建议仓位(LIGHT_BUY 15%/BUY 30%/STRONG_BUY 50%)投入当前NAV，
            T+N日以收盘价卖出，收益/亏损计入NAV实现复利。资金在T+N卖出后释放，可用于新信号。
            单只ETF仓位上限50%，组合总仓位上限100%。红色虚线为初始净值1.0基准线。
        </p>
    </div>

    <div class="section">
        <h2>各ETF贡献分析</h2>
        <div class="chart-container">
            {contrib_svg}
        </div>
        <table style="margin-top: 15px;">
            <tr>
                <th>ETF名称</th>
                <th>信号数</th>
                <th>胜利</th>
                <th>胜率</th>
                <th>P&amp;L贡献</th>
                <th>累计投入(%)</th>
                <th>投入回报率</th>
            </tr>
            {etf_contrib_rows}
        </table>
    </div>

    <div class="section">
        <h2>组合交易明细（最近30笔，按平仓日倒序）</h2>
        <table>
            <tr>
                <th>买入日</th>
                <th>卖出日</th>
                <th>ETF</th>
                <th>算法</th>
                <th>信号分</th>
                <th>等级</th>
                <th>仓位</th>
                <th>买入价</th>
                <th>T+N收益</th>
                <th>P&amp;L</th>
                <th>NAV(平仓后)</th>
            </tr>
            {port_detail_rows}
        </table>
    </div>

    <!-- ===== Existing Report Sections ===== -->

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
        <h2>各ETF回测统计</h2>
        <table>
            <tr>
                <th>ETF名称</th>
                <th>算法</th>
                <th>信号数</th>
                <th>胜率</th>
                <th>胜利</th>
                <th>失败</th>
                <th>平均分</th>
                <th>平均收益</th>
                <th>平均最大</th>
                <th>中位收益</th>
                <th>最佳最大</th>
                <th>最差收盘</th>
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
                <th>日期</th><th>ETF</th><th>算法</th><th>买入价</th><th>信号分</th>
                <th>等级</th><th>胜</th><th>达标天数</th><th>最大收益</th><th>收盘收益</th>
                <th>T+N最高</th><th>T+N收盘</th>
            </tr>
            {detail_rows}
        </table>
    </div>

    <div class="footer">
        <p>ETF多策略信号系统 v2.0 | 基于akshare数据 | {len(algo_stats)}种独特算法 × {meta['etf_count']}只ETF</p>
        <p>信号分0-100，越高越推荐买入 | 胜率定义: T+3/T+5（3/5个交易日）最高价收益>0.5%</p>
        <p>组合模拟: 初始净值1.0, 按信号等级仓位买入, T+N收盘卖出, 仓位平仓时复利</p>
        <p>资金管理: 单仓≤50% / 总仓≤100% / T+N释放资金 / 单仓<50%可金字塔加仓</p>
        <p>⚠️ 本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
    </div>
</body>
</html>"""

    return html


def main():
    logger.info("=" * 60)
    logger.info("ETF Portfolio Backtest Report Generator (Capital Management)")
    logger.info("=" * 60)

    # 1. Run backtest (cooldown off, portfolio manages capital)
    engine = BacktestEngine(cooldown_days=0)
    result = engine.run_backtest(signal_threshold=60, start_date='2024-01-01')
    signals = result['signals']
    logger.info(f"Backtest complete: {len(signals)} signals")

    # 2. Simulate portfolio (capital management)
    portfolio = simulate_portfolio(signals)
    logger.info(f"Portfolio: NAV={portfolio['final_nav']:.4f}, "
                f"Return={portfolio['total_return']:+.2f}%, "
                f"MaxDD={portfolio['max_drawdown']:.2f}%, "
                f"Sharpe={portfolio['sharpe']:.3f}, "
                f"CAGR={portfolio['cagr']:.2f}%")
    logger.info(f"  Trades: {portfolio['total_trades']}, "
                f"Skipped: {portfolio.get('skipped_signals', 0)}, "
                f"Pyramiding: {portfolio.get('pyramid_trades', 0)}")

    # 3. Generate HTML
    html = generate_html_report(result, portfolio)
    html_path = os.path.join(OUTPUT_DIR, 'backtest_portfolio_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML report saved: {html_path}")

    # 4. Save portfolio data
    port_json = {
        'final_nav': portfolio['final_nav'],
        'total_return': portfolio['total_return'],
        'max_drawdown': portfolio['max_drawdown'],
        'sharpe': portfolio['sharpe'],
        'cagr': portfolio['cagr'],
        'win_rate': portfolio['win_rate'],
        'total_trades': portfolio['total_trades'],
        'win_trades': portfolio['win_trades'],
        'etf_stats': {k: {kk: vv for kk, vv in v.items() if kk != 'trades'}
                      for k, v in portfolio['etf_stats'].items()},
    }
    json_path = os.path.join(OUTPUT_DIR, 'portfolio_result.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(port_json, f, ensure_ascii=False, indent=2)
    logger.info(f"Portfolio JSON saved: {json_path}")

    # 5. Print summary
    print(f"\n{'='*70}")
    print(f"Portfolio Simulation Summary (Capital Management)")
    print(f"{'='*70}")
    print(f"  Final NAV:      {portfolio['final_nav']:.4f}")
    print(f"  Total Return:   {portfolio['total_return']:+.2f}%")
    print(f"  CAGR:           {portfolio['cagr']:+.2f}%")
    print(f"  Max Drawdown:   {portfolio['max_drawdown']:.2f}%")
    print(f"  Sharpe:         {portfolio['sharpe']:.3f}")
    print(f"  Win Rate:       {portfolio['win_rate']:.1f}% ({portfolio['win_trades']}/{portfolio['total_trades']})")
    print(f"  Skipped:        {portfolio.get('skipped_signals', 0)} signals (capital/position limits)")
    print(f"  Pyramiding:     {portfolio.get('pyramid_trades', 0)} trades (added to existing position)")
    print(f"\n  Top 5 ETF Contributors:")
    for name, s in sorted(portfolio['etf_stats'].items(), key=lambda x: -x[1]['pnl'])[:5]:
        print(f"    {name:20s} P&L: {s['pnl']*100:+.3f}%  Signals: {s['signals']}  Win: {s['wins']}/{s['signals']}")
    print(f"\n  Bottom 5 ETF:")
    for name, s in sorted(portfolio['etf_stats'].items(), key=lambda x: x[1]['pnl'])[:5]:
        print(f"    {name:20s} P&L: {s['pnl']*100:+.3f}%  Signals: {s['signals']}  Win: {s['wins']}/{s['signals']}")


if __name__ == '__main__':
    main()
