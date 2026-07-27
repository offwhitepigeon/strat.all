# -*- coding: utf-8 -*-
"""
卖出策略优化回测 - 7ETF组合
==========================
在2024全年数据上测试多种卖出策略, 选出最佳, 并在2024-至今上验证。

标的: 红利/沪深300/双创50/黄金股/黄金/石油LOF/标普油气 (7只)
初始净值=1, 每只ETF等权(1/7资金), 各ETF独立交易
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
SIGNAL_THRESHOLD = 60


# ============================================================
# 卖出策略定义
# ============================================================
# 每个策略函数: (future_df, buy_price, hold_days) -> (sell_day, sell_price)
# future_df: T+1到T+N的OHLC DataFrame
# sell_day: 实际卖出天数(1~N), sell_price: 卖出价

def strat_hold_to_n(future_df, buy_price, hold_days):
    """B1: 持有到T+N卖出"""
    last = future_df.iloc[-1]
    return hold_days, float(last['close'])

def strat_t1_always(future_df, buy_price, hold_days):
    """E1: T+1无条件卖出"""
    return 1, float(future_df.iloc[0]['close'])

def strat_t2_always(future_df, buy_price, hold_days):
    """E2: T+2无条件卖出"""
    day = min(2, len(future_df))
    return day, float(future_df.iloc[day-1]['close'])

def strat_close_profit(threshold):
    """A组: T+i收盘盈利达threshold则卖出, 否则推迟, 最晚T+N"""
    def _strat(future_df, buy_price, hold_days):
        for j in range(len(future_df)):
            close = float(future_df.iloc[j]['close'])
            ret = (close / buy_price - 1) * 100
            if ret >= threshold:
                return j + 1, close
        last = future_df.iloc[-1]
        return hold_days, float(last['close'])
    return _strat

def strat_high_limit(threshold):
    """C组: T+i日内最高价达threshold%则按limit价卖出, 最晚T+N收盘"""
    limit_price = None  # 动态计算
    def _strat(future_df, buy_price, hold_days):
        _limit = buy_price * (1 + threshold / 100)
        for j in range(len(future_df)):
            high = float(future_df.iloc[j]['high'])
            if high >= _limit:
                return j + 1, _limit  # 限价单成交
        last = future_df.iloc[-1]
        return hold_days, float(last['close'])
    return _strat

def strat_progressive(t1_thresh, t2_thresh):
    """D组: T+1收盘>t1%卖出, T+2收盘>t2%卖出, 否则T+N"""
    def _strat(future_df, buy_price, hold_days):
        if len(future_df) >= 1:
            c1 = float(future_df.iloc[0]['close'])
            if (c1 / buy_price - 1) * 100 >= t1_thresh:
                return 1, c1
        if len(future_df) >= 2:
            c2 = float(future_df.iloc[1]['close'])
            if (c2 / buy_price - 1) * 100 >= t2_thresh:
                return 2, c2
        last = future_df.iloc[-1]
        return hold_days, float(last['close'])
    return _strat

def strat_trailing_stop(activate_thresh):
    """G组: 激活止盈 - T+i收盘盈利>activate%则设止损=买入价, 次日跌破止损则卖出, 最晚T+N"""
    def _strat(future_df, buy_price, hold_days):
        stop_active = False
        for j in range(len(future_df)):
            close = float(future_df.iloc[j]['close'])
            ret = (close / buy_price - 1) * 100
            if not stop_active:
                if ret >= activate_thresh:
                    stop_active = True
                    # 当天不卖, 继续持有等更多收益
                # 未激活, 继续
            else:
                # 止损激活, 跌破买入价卖出
                if close <= buy_price:
                    return j + 1, close
        last = future_df.iloc[-1]
        return hold_days, float(last['close'])
    return _strat

def strat_high_or_close(high_thresh, close_thresh):
    """F组: T+i最高价达high_thresh%限价卖出 OR T+i收盘达close_thresh%卖出, 最晚T+N"""
    def _strat(future_df, buy_price, hold_days):
        _limit = buy_price * (1 + high_thresh / 100)
        for j in range(len(future_df)):
            high = float(future_df.iloc[j]['high'])
            close = float(future_df.iloc[j]['close'])
            if high >= _limit:
                return j + 1, _limit
            if (close / buy_price - 1) * 100 >= close_thresh:
                return j + 1, close
        last = future_df.iloc[-1]
        return hold_days, float(last['close'])
    return _strat


# 策略列表
STRATEGIES = [
    # B组: 持有到期
    ('B1_持有到T+N',        strat_hold_to_n),

    # E组: 固定天数
    ('E1_固定T+1卖出',      strat_t1_always),
    ('E2_固定T+2卖出',      strat_t2_always),

    # A组: 收盘价止盈
    ('A1_收盘盈利0%卖出',   strat_close_profit(0)),
    ('A2_收盘盈利0.3%卖出', strat_close_profit(0.3)),
    ('A3_收盘盈利0.5%卖出', strat_close_profit(0.5)),
    ('A4_收盘盈利1%卖出',   strat_close_profit(1.0)),
    ('A5_收盘盈利1.5%卖出', strat_close_profit(1.5)),

    # C组: 限价单(日内最高价)
    ('C1_限价1%卖出',       strat_high_limit(1.0)),
    ('C2_限价2%卖出',       strat_high_limit(2.0)),
    ('C3_限价3%卖出',       strat_high_limit(3.0)),

    # D组: 渐进式
    ('D1_T1_0.5%_T2_0%',    strat_progressive(0.5, 0)),
    ('D2_T1_1%_T2_0.5%',    strat_progressive(1.0, 0.5)),

    # G组: 移动止损
    ('G1_止盈0.5%后保本',   strat_trailing_stop(0.5)),
    ('G2_止盈1%后保本',     strat_trailing_stop(1.0)),

    # F组: 限价+收盘组合
    ('F1_限价1%或收盘0.5%',  strat_high_or_close(1.0, 0.5)),
    ('F2_限价2%或收盘1%',    strat_high_or_close(2.0, 1.0)),
]


# ============================================================
# 信号生成(共用)
# ============================================================
def generate_signals(target_etfs, start_date, end_date=None):
    """生成信号, 返回 {etf_code: [(buy_date, buy_price, score, future_df, hold_days), ...]}"""
    engine = DataEngine()
    signals_by_etf = {}

    for etf in target_etfs:
        df = engine.get_history_kline(etf.code)
        if df is None or len(df) < 250:
            continue
        df = calc_all_indicators(df)
        algorithm = get_algorithm(etf.algorithm)
        hold_days = getattr(etf, 'hold_days', 3)

        signals = []
        for i in range(60, len(df) - hold_days):
            df_slice = df.iloc[:i + 1]
            try:
                signal = algorithm.calculate(df_slice)
            except Exception:
                continue
            if signal.score < SIGNAL_THRESHOLD:
                continue

            buy_date = df.iloc[i]['date']
            buy_price = float(df.iloc[i]['close'])

            # 日期过滤
            date_str = buy_date.strftime('%Y-%m-%d') if hasattr(buy_date, 'strftime') else str(buy_date)[:10]
            if date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue

            future = df.iloc[i + 1: i + 1 + hold_days]
            if len(future) < 1:
                continue

            signals.append({
                'date': date_str,
                'buy_price': buy_price,
                'score': signal.score,
                'future': future,
                'hold_days': hold_days,
            })

        signals_by_etf[etf.code] = signals
        logger.info(f"{etf.name:12s} | 信号:{len(signals):3d}")

    return signals_by_etf


# ============================================================
# 策略回测
# ============================================================
def run_strategy(signals_by_etf, target_etfs, strat_fn, initial_nav=1.0):
    """对一组信号应用某个策略, 返回 trades列表 和 组合净值曲线"""
    n_etfs = len(target_etfs)
    all_trades = []
    etf_navs = {}

    for etf in target_etfs:
        sigs = signals_by_etf.get(etf.code, [])
        nav = initial_nav
        nav_curve = [(sigs[0]['date'] if sigs else '2024-01-01', nav)]
        trades = []
        for sig in sorted(sigs, key=lambda x: x['date']):
            future = sig['future']
            bp = sig['buy_price']
            hd = sig['hold_days']
            sell_day, sell_price = strat_fn(future, bp, hd)
            ret = (sell_price / bp - 1) * 100
            nav *= (1 + ret / 100)

            sell_date_row = future.iloc[sell_day - 1] if sell_day <= len(future) else future.iloc[-1]
            sell_date = sell_date_row['date']
            sell_date_str = sell_date.strftime('%Y-%m-%d') if hasattr(sell_date, 'strftime') else str(sell_date)[:10]

            trade = {
                'date': sig['date'],
                'sell_date': sell_date_str,
                'etf_code': etf.code,
                'etf_name': etf.name,
                'buy_price': round(bp, 4),
                'sell_price': round(sell_price, 4),
                'sell_day': sell_day,
                'score': round(sig['score'], 1),
                'return_pct': round(ret, 2),
                'is_win': ret > 0.5,
            }
            trades.append(trade)
            nav_curve.append((sell_date_str, nav))

        etf_navs[etf.code] = {
            'name': etf.name,
            'final_nav': nav,
            'trade_count': len(trades),
            'wins': sum(1 for t in trades if t['is_win']),
            'nav_curve': nav_curve,
        }
        all_trades.extend(trades)

    # 组合净值 = 等权平均
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
            etf_nav = initial_nav
            for d, n in data.get('nav_curve', []):
                if d <= date_str:
                    etf_nav = n
                else:
                    break
            navs.append(etf_nav)
        portfolio_nav.append({
            'date': date_str,
            'nav': round(sum(navs) / n_etfs, 4),
        })

    return all_trades, etf_navs, portfolio_nav


def calc_metrics(trades, portfolio_nav, initial_nav=1.0, start_date='2024-01-01'):
    """计算回测指标"""
    n = len(trades)
    if n == 0:
        return {'total_trades': 0}

    wins = sum(1 for t in trades if t['is_win'])
    rets = [t['return_pct'] for t in trades]
    holds = [t['sell_day'] for t in trades]

    final_nav = portfolio_nav[-1]['nav'] if portfolio_nav else initial_nav
    navs = [p['nav'] for p in portfolio_nav] if portfolio_nav else [initial_nav]
    max_nav = max(navs)
    min_nav = min(navs)

    # 最大回撤
    peak = initial_nav
    max_dd = 0
    for p in portfolio_nav:
        if p['nav'] > peak:
            peak = p['nav']
        dd = (p['nav'] / peak - 1) * 100
        if dd < max_dd:
            max_dd = dd

    # 年化
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.now()
    years = max((end_dt - start_dt).days / 365.25, 0.01)
    cagr = (final_nav ** (1 / years) - 1) * 100 if final_nav > 0 else -100

    # Sharpe (按交易, 近似)
    if len(rets) > 1:
        rets_arr = np.array(rets)
        if np.std(rets_arr) > 0:
            sharpe = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(n)
        else:
            sharpe = 0
    else:
        sharpe = 0

    return {
        'total_trades': n,
        'wins': wins,
        'win_rate': round(wins / n * 100, 1),
        'avg_return': round(np.mean(rets), 2),
        'avg_hold': round(np.mean(holds), 1),
        'final_nav': round(final_nav, 4),
        'max_nav': round(max_nav, 4),
        'min_nav': round(min_nav, 4),
        'max_dd': round(max_dd, 1),
        'cagr': round(cagr, 2),
        'sharpe': round(sharpe, 2),
        'total_return': round((final_nav / initial_nav - 1) * 100, 2),
    }


# ============================================================
# HTML报告
# ============================================================
def generate_html(results_2024, results_full, best_name, best_2024, best_full):
    """生成对比HTML报告"""

    # 策略对比表 (2024年)
    rows_2024 = ""
    for name, metrics in sorted(results_2024.items(), key=lambda x: -x[1]['final_nav']):
        is_best = name == best_name
        nav_color = '#27ae60' if metrics['final_nav'] >= 1 else '#e74c3c'
        wr_color = '#27ae60' if metrics['win_rate'] >= 60 else ('#e67e22' if metrics['win_rate'] >= 50 else '#e74c3c')
        row_class = 'best-row' if is_best else ''
        rows_2024 += f"""
            <tr class="{row_class}">
                <td><strong>{name}</strong>{' ★' if is_best else ''}</td>
                <td>{metrics['total_trades']}</td>
                <td style="color:{wr_color};font-weight:bold;">{metrics['win_rate']:.1f}%</td>
                <td style="color:{'#27ae60' if metrics['avg_return'] >= 0 else '#e74c3c'};">{metrics['avg_return']:+.2f}%</td>
                <td>{metrics['avg_hold']:.1f}日</td>
                <td style="color:{nav_color};font-weight:bold;">{metrics['final_nav']:.4f}</td>
                <td style="color:{nav_color};">{metrics['total_return']:+.2f}%</td>
                <td>{metrics['cagr']:.2f}%</td>
                <td style="color:#e74c3c;">{metrics['max_dd']:.1f}%</td>
                <td>{metrics['sharpe']:.2f}</td>
            </tr>"""

    # 策略对比表 (2024-至今)
    rows_full = ""
    for name, metrics in sorted(results_full.items(), key=lambda x: -x[1]['final_nav']):
        is_best = name == best_name
        nav_color = '#27ae60' if metrics['final_nav'] >= 1 else '#e74c3c'
        wr_color = '#27ae60' if metrics['win_rate'] >= 60 else ('#e67e22' if metrics['win_rate'] >= 50 else '#e74c3c')
        row_class = 'best-row' if is_best else ''
        rows_full += f"""
            <tr class="{row_class}">
                <td><strong>{name}</strong>{' ★' if is_best else ''}</td>
                <td>{metrics['total_trades']}</td>
                <td style="color:{wr_color};font-weight:bold;">{metrics['win_rate']:.1f}%</td>
                <td style="color:{'#27ae60' if metrics['avg_return'] >= 0 else '#e74c3c'};">{metrics['avg_return']:+.2f}%</td>
                <td>{metrics['avg_hold']:.1f}日</td>
                <td style="color:{nav_color};font-weight:bold;">{metrics['final_nav']:.4f}</td>
                <td style="color:{nav_color};">{metrics['total_return']:+.2f}%</td>
                <td>{metrics['cagr']:.2f}%</td>
                <td style="color:#e74c3c;">{metrics['max_dd']:.1f}%</td>
                <td>{metrics['sharpe']:.2f}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>卖出策略优化报告 | 7ETF组合</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif; background: #f5f6fa; color: #2c3e50; padding: 20px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 24px; margin-bottom: 8px; }}
        .header .meta {{ opacity: 0.8; font-size: 14px; }}
        .section {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow-x: auto; }}
        .section h2 {{ font-size: 18px; margin-bottom: 15px; color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #f8f9fa; padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6; color: #495057; white-space: nowrap; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #ecf0f1; }}
        tr.best-row {{ background: #fff3cd; }}
        .best-box {{ background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; padding: 25px; border-radius: 12px; margin-bottom: 20px; }}
        .best-box h2 {{ color: white; border: none; }}
        .best-box .strategy-name {{ font-size: 28px; font-weight: bold; margin: 10px 0; }}
        .best-box .metrics {{ display: flex; gap: 30px; flex-wrap: wrap; }}
        .best-box .metric {{ text-align: center; }}
        .best-box .metric .v {{ font-size: 22px; font-weight: bold; }}
        .best-box .metric .l {{ font-size: 12px; opacity: 0.9; }}
        .footer {{ text-align: center; padding: 20px; color: #95a5a6; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>卖出策略优化报告</h1>
        <div class="meta">
            标的: 红利/沪深300/双创50/黄金股/黄金/石油LOF/标普油气(7只) |
            训练期: 2024-01-01 ~ 2024-12-31 | 验证期: 2024-01-01 ~ 至今 |
            策略数: {len(results_2024)}种 |
            初始净值: 1.0000
        </div>
    </div>

    <div class="best-box">
        <h2>最佳策略</h2>
        <div class="strategy-name">{best_name}</div>
        <div class="metrics">
            <div class="metric"><div class="v">{best_2024['final_nav']:.4f}</div><div class="l">2024年净值</div></div>
            <div class="metric"><div class="v">{best_2024['cagr']:.2f}%</div><div class="l">2024年化</div></div>
            <div class="metric"><div class="v">{best_2024['max_dd']:.1f}%</div><div class="l">2024最大回撤</div></div>
            <div class="metric"><div class="v">{best_full['final_nav']:.4f}</div><div class="l">全期净值</div></div>
            <div class="metric"><div class="v">{best_full['cagr']:.2f}%</div><div class="l">全期年化</div></div>
            <div class="metric"><div class="v">{best_full['max_dd']:.1f}%</div><div class="l">全期回撤</div></div>
        </div>
    </div>

    <div class="section">
        <h2>策略对比 - 2024年训练期 (2024-01-01 ~ 2024-12-31, 按净值降序)</h2>
        <table>
            <tr>
                <th>策略</th><th>交易数</th><th>胜率</th><th>平均收益</th><th>平均持有</th>
                <th>最终净值</th><th>总收益</th><th>年化</th><th>最大回撤</th><th>Sharpe</th>
            </tr>
            {rows_2024}
        </table>
    </div>

    <div class="section">
        <h2>策略对比 - 全期验证 (2024-01-01 ~ 至今, 按净值降序)</h2>
        <table>
            <tr>
                <th>策略</th><th>交易数</th><th>胜率</th><th>平均收益</th><th>平均持有</th>
                <th>最终净值</th><th>总收益</th><th>年化</th><th>最大回撤</th><th>Sharpe</th>
            </tr>
            {rows_full}
        </table>
    </div>

    <div class="footer">
        <p>7ETF卖出策略优化 | 训练期2024全年, 验证期2024-至今 | {len(results_2024)}种策略对比</p>
        <p>★ 标记为2024年训练期最佳策略 | 每只ETF等权(1/7资金), 各ETF独立交易</p>
        <p>本报告仅供参考，不构成投资建议。</p>
    </div>
</body>
</html>"""
    return html


# ============================================================
# 主流程
# ============================================================
def main():
    start_time = datetime.now()
    target_etfs = [e for e in ETF_POOL if e.code in TARGET_CODES]
    logger.info(f"目标ETF: {len(target_etfs)}只, 策略数: {len(STRATEGIES)}")

    # 1. 生成信号 (2024-01-01 ~ 至今, 共用)
    signals_full = generate_signals(target_etfs, '2024-01-01')
    # 2024年信号
    signals_2024 = {}
    for code, sigs in signals_full.items():
        signals_2024[code] = [s for s in sigs if s['date'] <= '2024-12-31']

    total_sigs = sum(len(v) for v in signals_full.items())
    total_2024 = sum(len(v) for v in signals_2024.items())
    logger.info(f"信号总数: {sum(len(v) for v in signals_full.values())} (2024年: {sum(len(v) for v in signals_2024.values())})")

    # 2. 对每个策略在两个区间上回测
    results_2024 = {}
    results_full = {}

    for strat_name, strat_fn in STRATEGIES:
        # 2024年训练
        trades_2024, _, nav_2024 = run_strategy(signals_2024, target_etfs, strat_fn, 1.0)
        results_2024[strat_name] = calc_metrics(trades_2024, nav_2024, 1.0, '2024-01-01')

        # 全期验证
        trades_full, _, nav_full = run_strategy(signals_full, target_etfs, strat_fn, 1.0)
        results_full[strat_name] = calc_metrics(trades_full, nav_full, 1.0, '2024-01-01')

        m = results_2024[strat_name]
        logger.info(f"{strat_name:25s} | 2024净值:{m['final_nav']:.4f} CAGR:{m['cagr']:.2f}% 回撤:{m['max_dd']:.1f}% | "
                     f"全期净值:{results_full[strat_name]['final_nav']:.4f}")

    # 3. 选最佳 (按2024年净值)
    best_name = max(results_2024, key=lambda x: results_2024[x]['final_nav'])
    best_2024 = results_2024[best_name]
    best_full = results_full[best_name]
    logger.info(f"\n最佳策略: {best_name}")
    logger.info(f"  2024年: 净值={best_2024['final_nav']:.4f}, CAGR={best_2024['cagr']:.2f}%, 回撤={best_2024['max_dd']:.1f}%")
    logger.info(f"  全期:   净值={best_full['final_nav']:.4f}, CAGR={best_full['cagr']:.2f}%, 回撤={best_full['max_dd']:.1f}%")

    # 4. 生成HTML
    html = generate_html(results_2024, results_full, best_name, best_2024, best_full)
    html_path = os.path.join(OUTPUT_DIR, 'strategy_optimization_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML报告: {html_path}")

    # 5. 保存JSON
    result = {
        'metadata': {
            'strategies': len(STRATEGIES),
            'train_period': '2024-01-01 ~ 2024-12-31',
            'test_period': '2024-01-01 ~ latest',
            'best_strategy': best_name,
        },
        'results_2024': results_2024,
        'results_full': results_full,
    }
    json_path = os.path.join(OUTPUT_DIR, 'strategy_optimization_result.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    # 6. 打印
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 80}")
    print(f"卖出策略优化结果 (耗时 {duration:.1f}秒)")
    print(f"{'=' * 80}")
    print(f"\n2024年训练期 (按净值降序):")
    print(f"{'策略':<26s} {'交易':>4s} {'胜率':>6s} {'均收益':>7s} {'净值':>8s} {'年化':>7s} {'回撤':>7s} {'Sharpe':>7s}")
    print("-" * 80)
    for name, m in sorted(results_2024.items(), key=lambda x: -x[1]['final_nav']):
        marker = '★' if name == best_name else ' '
        print(f"{marker} {name:<24s} {m['total_trades']:4d} {m['win_rate']:5.1f}% {m['avg_return']:+6.2f}% "
              f"{m['final_nav']:8.4f} {m['cagr']:6.2f}% {m['max_dd']:6.1f}% {m['sharpe']:7.2f}")

    print(f"\n全期验证 (2024-至今, 按净值降序):")
    print(f"{'策略':<26s} {'交易':>4s} {'胜率':>6s} {'均收益':>7s} {'净值':>8s} {'年化':>7s} {'回撤':>7s} {'Sharpe':>7s}")
    print("-" * 80)
    for name, m in sorted(results_full.items(), key=lambda x: -x[1]['final_nav']):
        marker = '★' if name == best_name else ' '
        print(f"{marker} {name:<24s} {m['total_trades']:4d} {m['win_rate']:5.1f}% {m['avg_return']:+6.2f}% "
              f"{m['final_nav']:8.4f} {m['cagr']:6.2f}% {m['max_dd']:6.1f}% {m['sharpe']:7.2f}")

    print(f"\n最佳策略: {best_name}")
    print(f"  2024年: 净值={best_2024['final_nav']:.4f} CAGR={best_2024['cagr']:.2f}% 回撤={best_2024['max_dd']:.1f}%")
    print(f"  全期:   净值={best_full['final_nav']:.4f} CAGR={best_full['cagr']:.2f}% 回撤={best_full['max_dd']:.1f}%")


if __name__ == '__main__':
    main()
