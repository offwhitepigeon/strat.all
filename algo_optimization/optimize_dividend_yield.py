# -*- coding: utf-8 -*-
"""
红利ETF - dividend_value 优化 + 股息率因子
==========================================
当前算法: dividend_value (70.2%胜率, +0.21%收益, 47信号)
最佳算法: new_energy_reversal (75.0%胜率, +0.47%收益, 52信号)
问题: dividend_value名为"红利估值型"但无股息率因子

设计3个含股息率的优化变体:
  A. 股息率核心版 - dividend_value基础上用股息率替换底背离(15%)
  B. new_energy+股息率版 - new_energy_reversal基础上加股息率(10%)
  C. 股息率重权版 - 股息率作为第一权重因子(25%)
"""
import sys, os, json, logging
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import (ALGORITHM_MAP, BaseAlgorithm,
                        DividendValueAlgorithm, NewEnergyReversalAlgorithm)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'algo_optimization')
ETF_CODE = 'sh510880'
ETF_NAME = '红利ETF'
START_DATE = '2024-01-01'
THRESHOLD = 60


# ======================================================================
# 股息率数据加载与计算
# ======================================================================
def load_dividend_history():
    """从akshare加载红利ETF分红历史"""
    try:
        ak = __import__('akshare')
        df = ak.fund_etf_dividend_sina(symbol='sh510880')
        if df is None or df.empty:
            logger.warning("无法获取分红历史")
            return None
        # 列名: 除息日, 累计分红
        cols = df.columns.tolist()
        df.columns = ['ex_date', 'cum_dividend']
        df['ex_date'] = pd.to_datetime(df['ex_date'])
        df = df.sort_values('ex_date').reset_index(drop=True)
        # 计算每次分红金额
        df['dividend'] = df['cum_dividend'].diff().fillna(df['cum_dividend'].iloc[0])
        logger.info(f"分红历史: {len(df)}次, {df['ex_date'].iloc[0].strftime('%Y-%m-%d')}~{df['ex_date'].iloc[-1].strftime('%Y-%m-%d')}")
        return df
    except Exception as e:
        logger.error(f"加载分红历史失败: {e}")
        return None


def add_dividend_yield(df, div_df):
    """为K线数据添加股息率列"""
    if div_df is None:
        logger.warning("无分红数据, 股息率设为0")
        df['dividend_yield'] = 0.0
        df['dividend_yield_pct'] = 50.0
        return df

    # 为每个交易日计算TTM分红(最近一次年度分红的金额)
    div_records = []
    for _, row in div_df.iterrows():
        div_records.append((row['ex_date'], row['dividend']))

    ttm_divs = []
    for _, krow in df.iterrows():
        date = krow['date']
        # 找到最近一次分红(在当前日期之前或当天)
        ttm = 0
        for ex_date, div_amt in div_records:
            if ex_date <= date:
                ttm = div_amt
            else:
                break
        ttm_divs.append(ttm)

    df['ttm_dividend'] = ttm_divs
    # 股息率 = TTM分红 / 收盘价 * 100
    df['dividend_yield'] = np.where(df['close'] > 0,
                                     df['ttm_dividend'] / df['close'] * 100, 0)

    # 250日滚动百分位
    df['dividend_yield_pct'] = df['dividend_yield'].rolling(
        window=250, min_periods=60).rank(pct=True) * 100
    df['dividend_yield_pct'] = df['dividend_yield_pct'].fillna(50)

    # 统计
    recent = df[df['date'] >= START_DATE]
    if len(recent) > 0:
        logger.info(f"股息率: 范围{recent['dividend_yield'].min():.2f}%~{recent['dividend_yield'].max():.2f}%, "
                     f"均值{recent['dividend_yield'].mean():.2f}%, "
                     f"百分位范围{recent['dividend_yield_pct'].min():.0f}~{recent['dividend_yield_pct'].max():.0f}")

    return df


# ======================================================================
# 变体A: 股息率核心版 - dividend_value基础上用股息率替换底背离
# ======================================================================
class DividendValueYieldCore(BaseAlgorithm):
    """
    dividend_value 股息率核心版
    - F1-F4保持原版(RSI 28% + MA200 22% + Z-score 20% + 布林带 15%)
    - F5替换: 股息率分位(15%) - 替代底背离, 真正的红利价值因子
      股息率在过去250日中的百分位 >= 90 → 15分, >= 80 → 12, >= 60 → 8, >= 40 → 4
    """
    name = "dividend_value_yield_core"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        dy_pct = float(last.get('dividend_yield_pct', 50))

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma200': dev_ma200, 'bb_percent_b': bb_pctb,
            'dividend_yield_pct': dy_pct,
        })

        # F1: RSI超卖 (28%)
        if rsi < 25:
            s1 = 28
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s1 = 23
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s1 = 17
            reasons.append(f"RSI={rsi:.0f}，偏低")
        elif rsi < 45:
            s1 = 9
        else:
            s1 = 0
        score += s1

        # F2: MA200偏离度 (22%)
        if dev_ma200 < -12:
            s2 = 22
            reasons.append(f"40周均线偏离{dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -9:
            s2 = 19
            reasons.append(f"偏离{dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -8:
            s2 = 17
            reasons.append(f"偏离{dev_ma200:.1f}%，明显超卖")
        elif dev_ma200 < -5:
            s2 = 10
        elif dev_ma200 < -3:
            s2 = 6
        elif dev_ma200 < 0:
            s2 = 3
        else:
            s2 = 0
        score += s2

        # F3: Z-Score (20%)
        if zscore < -2:
            s3 = 20
            reasons.append(f"Z-score={zscore:.1f}，严重低于均值")
        elif zscore < -1.8:
            s3 = 18
            reasons.append(f"Z-score={zscore:.1f}，明显低于均值")
        elif zscore < -1.5:
            s3 = 15
        elif zscore < -1:
            s3 = 10
        else:
            s3 = 0
        score += s3

        # F4: 布林带位置 (15%)
        if bb_pctb < 0.05:
            s4 = 15
            reasons.append("触及布林带下轨")
        elif bb_pctb < 0.15:
            s4 = 10
        elif bb_pctb < 0.3:
            s4 = 5
        else:
            s4 = 0
        score += s4

        # F5: 股息率分位 (15%, 替代底背离) - 核心新增
        if dy_pct >= 90:
            s5 = 15
            reasons.append(f"股息率分位{dy_pct:.0f}，历史高位")
        elif dy_pct >= 80:
            s5 = 12
            reasons.append(f"股息率分位{dy_pct:.0f}，明显偏高")
        elif dy_pct >= 60:
            s5 = 8
            reasons.append(f"股息率分位{dy_pct:.0f}，偏高")
        elif dy_pct >= 40:
            s5 = 4
        else:
            s5 = 0
        score += s5

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 变体B: new_energy_reversal + 股息率
# ======================================================================
class DividendValueYieldEnhanced(BaseAlgorithm):
    """
    new_energy_reversal + 股息率版
    - 基于new_energy_reversal(红利ETF最优算法), 新增股息率因子(10%)
    - F1 RSI(20%)+F2 MA200(20%)+F3 Z-score(15%)+F4 BB(10%)
    - F5 量能(10%)+F6 KDJ(10%)+F7 动量(5%)+F8 股息率(10%)
    """
    name = "dividend_value_yield_enhanced"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        bb_pctb = float(last.get('bb_percent_b', 0.5))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        kdj_j = float(last.get('kdj_j', 50))
        dev_ma60 = float(last.get('dev_ma60', 0))
        dy_pct = float(last.get('dividend_yield_pct', 50))

        rsi_prev = None
        try:
            if len(df) >= 2 and 'rsi_14' in df.columns:
                rsi_prev = float(df.iloc[-2]['rsi_14'])
        except (IndexError, KeyError, TypeError, ValueError):
            pass

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma200': dev_ma200, 'bb_percent_b': bb_pctb,
            'vol_ratio': vol_ratio, 'kdj_j': kdj_j,
            'dev_ma60': dev_ma60, 'dividend_yield_pct': dy_pct,
        })

        # F1: RSI超卖 (20%)
        if rsi < 25:
            s1 = 20
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s1 = 17
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s1 = 12
            reasons.append(f"RSI={rsi:.0f}，偏低")
        elif rsi < 45:
            s1 = 7
        else:
            s1 = 0
        score += s1

        # F2: MA200偏离度 (20%)
        if dev_ma200 < -12:
            s2 = 20
            reasons.append(f"偏离{dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -9:
            s2 = 18
        elif dev_ma200 < -8:
            s2 = 15
            reasons.append(f"偏离{dev_ma200:.1f}%，明显超卖")
        elif dev_ma200 < -5:
            s2 = 10
        elif dev_ma200 < -3:
            s2 = 6
        elif dev_ma200 < 0:
            s2 = 3
        else:
            s2 = 0
        score += s2

        # F3: Z-Score (15%)
        if zscore < -2:
            s3 = 15
            reasons.append(f"Z-score={zscore:.1f}，严重低于均值")
        elif zscore < -1.8:
            s3 = 13
        elif zscore < -1.5:
            s3 = 11
        elif zscore < -1:
            s3 = 7
        else:
            s3 = 0
        score += s3

        # F4: 布林带 (10%)
        if bb_pctb < 0.05:
            s4 = 10
            reasons.append("布林带下轨")
        elif bb_pctb < 0.15:
            s4 = 7
        elif bb_pctb < 0.3:
            s4 = 3
        else:
            s4 = 0
        score += s4

        # F5: 量能确认 (10%)
        if vol_ratio > 2.0:
            s5 = 10
            reasons.append(f"量比{vol_ratio:.1f}，恐慌放量")
        elif vol_ratio > 1.5:
            s5 = 7
        elif vol_ratio > 1.2:
            s5 = 4
        else:
            s5 = 0
        score += s5

        # F6: KDJ超卖 (10%)
        if kdj_j < 0:
            s6 = 10
            reasons.append(f"KDJ J={kdj_j:.0f}，极度超卖")
        elif kdj_j < 10:
            s6 = 8
            reasons.append(f"KDJ J={kdj_j:.0f}，严重超卖")
        elif kdj_j < 20:
            s6 = 5
        elif kdj_j < 30:
            s6 = 2
        else:
            s6 = 0
        score += s6

        # F7: 短期动量 (5%)
        if rsi_prev is not None and rsi > rsi_prev and rsi < 45:
            s7 = 5
            reasons.append(f"RSI回升，动量转正")
        else:
            s7 = 0
        score += s7

        # F8: 股息率分位 (10%, 新增)
        if dy_pct >= 90:
            s8 = 10
            reasons.append(f"股息率分位{dy_pct:.0f}，历史高位")
        elif dy_pct >= 80:
            s8 = 8
            reasons.append(f"股息率分位{dy_pct:.0f}，明显偏高")
        elif dy_pct >= 60:
            s8 = 5
        elif dy_pct >= 40:
            s8 = 2
        else:
            s8 = 0
        score += s8

        # 硬过滤: MA60极端偏离
        if dev_ma60 < -15:
            score = min(score, 59)
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%，封顶")

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 变体C: 股息率重权版 - 股息率作为第一权重因子
# ======================================================================
class DividendValueYieldHeavy(BaseAlgorithm):
    """
    股息率重权版 - 股息率作为核心因子(25%)
    - F1: 股息率分位 (25%, 核心因子)
    - F2: RSI超卖 (25%)
    - F3: MA200偏离度 (20%)
    - F4: KDJ超卖 (10%)
    - F5: 量能确认 (10%)
    - F6: Z-Score (10%)
    - 硬过滤: dev_ma60 < -15% → 封顶59
    """
    name = "dividend_value_yield_heavy"

    def _calc_signal(self, df, last, price, indicators, extra_data=None):
        score = 0
        reasons = []

        rsi = float(last.get('rsi_14', 50))
        zscore = float(last.get('zscore_20', 0))
        dev_ma200 = float(last.get('dev_ma200', 0))
        vol_ratio = float(last.get('vol_ratio_20', 1))
        kdj_j = float(last.get('kdj_j', 50))
        dev_ma60 = float(last.get('dev_ma60', 0))
        dy_pct = float(last.get('dividend_yield_pct', 50))
        dy = float(last.get('dividend_yield', 0))

        indicators.update({
            'rsi_14': rsi, 'zscore_20': zscore,
            'dev_ma200': dev_ma200, 'vol_ratio': vol_ratio,
            'kdj_j': kdj_j, 'dev_ma60': dev_ma60,
            'dividend_yield_pct': dy_pct, 'dividend_yield': dy,
        })

        # F1: 股息率分位 (25%, 核心因子)
        if dy_pct >= 95:
            s1 = 25
            reasons.append(f"股息率分位{dy_pct:.0f}(收益率{dy:.2f}%)，历史极高")
        elif dy_pct >= 90:
            s1 = 22
            reasons.append(f"股息率分位{dy_pct:.0f}(收益率{dy:.2f}%)，历史高位")
        elif dy_pct >= 80:
            s1 = 18
            reasons.append(f"股息率分位{dy_pct:.0f}，明显偏高")
        elif dy_pct >= 60:
            s1 = 12
            reasons.append(f"股息率分位{dy_pct:.0f}，偏高")
        elif dy_pct >= 40:
            s1 = 6
        else:
            s1 = 0
        score += s1

        # F2: RSI超卖 (25%)
        if rsi < 25:
            s2 = 25
            reasons.append(f"RSI={rsi:.0f}，极度超卖")
        elif rsi < 30:
            s2 = 20
            reasons.append(f"RSI={rsi:.0f}，超卖")
        elif rsi < 35:
            s2 = 15
            reasons.append(f"RSI={rsi:.0f}，偏低")
        elif rsi < 45:
            s2 = 8
        else:
            s2 = 0
        score += s2

        # F3: MA200偏离度 (20%)
        if dev_ma200 < -12:
            s3 = 20
            reasons.append(f"偏离{dev_ma200:.1f}%，严重超卖")
        elif dev_ma200 < -9:
            s3 = 18
        elif dev_ma200 < -8:
            s3 = 15
            reasons.append(f"偏离{dev_ma200:.1f}%，明显超卖")
        elif dev_ma200 < -5:
            s3 = 10
        elif dev_ma200 < -3:
            s3 = 6
        elif dev_ma200 < 0:
            s3 = 3
        else:
            s3 = 0
        score += s3

        # F4: KDJ超卖 (10%)
        if kdj_j < 0:
            s4 = 10
            reasons.append(f"KDJ J={kdj_j:.0f}，极度超卖")
        elif kdj_j < 10:
            s4 = 8
            reasons.append(f"KDJ J={kdj_j:.0f}，严重超卖")
        elif kdj_j < 20:
            s4 = 5
        elif kdj_j < 30:
            s4 = 2
        else:
            s4 = 0
        score += s4

        # F5: 量能确认 (10%)
        if vol_ratio > 2.0:
            s5 = 10
            reasons.append(f"量比{vol_ratio:.1f}，恐慌放量")
        elif vol_ratio > 1.5:
            s5 = 7
        elif vol_ratio > 1.2:
            s5 = 4
        else:
            s5 = 0
        score += s5

        # F6: Z-Score (10%)
        if zscore < -2:
            s6 = 10
            reasons.append(f"Z-score={zscore:.1f}，严重低于均值")
        elif zscore < -1.5:
            s6 = 7
        elif zscore < -1:
            s6 = 5
        else:
            s6 = 0
        score += s6

        # 硬过滤: MA60极端偏离
        if dev_ma60 < -15:
            score = min(score, 59)
            reasons.append(f"偏离MA60 {dev_ma60:.1f}%，封顶")

        return self._build_result(min(score, 100), reasons, indicators)


# ======================================================================
# 回测函数
# ======================================================================
def run_backtest(df, algo, threshold=THRESHOLD):
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
        if signal.score < threshold:
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
            'date': date_str, 'score': round(signal.score, 1),
            'level': signal.level, 'buy_price': round(buy_price, 4),
            'max_return': round(max_ret, 2), 'close_return': round(close_ret, 2),
            'is_win': is_win, 'days_to_win': days_to_win,
            'reasons': signal.reasons[:3],
        })

    n = len(signals)
    if n == 0:
        return {'signals': 0, 'wins': 0, 'win_rate': 0, 'avg_return': 0,
                'avg_max_return': 0, 'avg_score': 0, 'best': 0, 'worst': 0,
                'sharpe': 0, 'std': 0, 'signal_list': []}

    wins = sum(1 for s in signals if s['is_win'])
    rets = [s['close_return'] for s in signals]
    max_rets = [s['max_return'] for s in signals]
    scores = [s['score'] for s in signals]
    std = np.std(rets) if len(rets) > 1 else 0
    sharpe = (np.mean(rets) / std * np.sqrt(len(rets))) if std > 0 else 0

    return {
        'signals': n, 'wins': wins, 'win_rate': round(wins / n * 100, 1),
        'avg_return': round(np.mean(rets), 2),
        'avg_max_return': round(np.mean(max_rets), 2),
        'avg_score': round(np.mean(scores), 1),
        'best': round(max(max_rets), 2), 'worst': round(min(rets), 2),
        'sharpe': round(sharpe, 3), 'std': round(std, 2),
        'signal_list': signals,
    }


def generate_html(results, df_info):
    sorted_results = sorted(results.items(), key=lambda x: (-x[1]['win_rate'], -x[1]['signals']))

    summary_rows = ""
    for name, r in sorted_results:
        wr = r['win_rate']
        ret = r['avg_return']
        maxret = r['avg_max_return']
        wr_c = '#27ae60' if wr >= 75 else ('#e67e22' if wr >= 60 else '#e74c3c')
        ret_c = '#27ae60' if ret >= 0 else '#e74c3c'
        max_c = '#27ae60' if maxret >= 0 else '#e74c3c'
        marker = ''
        if 'dividend_value (原版)' in name:
            marker = ' [原版]'
        if 'new_energy' in name:
            marker = ' [最佳基准]'
        summary_rows += f"""
            <tr>
                <td><strong>{name}</strong>{marker}</td>
                <td>{r['signals']}</td><td>{r['wins']}</td>
                <td style="color:{wr_c};font-weight:bold;">{wr:.1f}%</td>
                <td style="color:{ret_c};">{ret:+.2f}%</td>
                <td style="color:{max_c};font-weight:bold;">{maxret:+.2f}%</td>
                <td>{r['avg_score']:.1f}</td>
                <td>{r['sharpe']:.3f}</td>
                <td style="color:#27ae60;">{r['best']:+.2f}%</td>
                <td style="color:#e74c3c;">{r['worst']:+.2f}%</td>
            </tr>"""

    # 逐年对比
    yearly_data = {}
    for name, r in results.items():
        for s in r['signal_list']:
            year = s['date'][:4]
            key = f"{name}_{year}"
            if key not in yearly_data:
                yearly_data[key] = {'algo': name, 'year': year, 'signals': 0, 'wins': 0, 'rets': []}
            yearly_data[key]['signals'] += 1
            if s['is_win']:
                yearly_data[key]['wins'] += 1
            yearly_data[key]['rets'].append(s['close_return'])

    yearly_rows = ""
    years = sorted(set(s['date'][:4] for r in results.values() for s in r['signal_list']))
    for name, _ in sorted_results:
        for year in years:
            key = f"{name}_{year}"
            d = yearly_data.get(key)
            if d and d['signals'] > 0:
                wr = d['wins'] / d['signals'] * 100
                avg_ret = np.mean(d['rets'])
                wr_c = '#27ae60' if wr >= 75 else ('#e67e22' if wr >= 60 else '#e74c3c')
                yearly_rows += f"""<tr><td>{name}</td><td>{year}</td><td>{d['signals']}</td><td style="color:{wr_c};">{wr:.0f}%</td><td>{avg_ret:+.2f}%</td></tr>"""

    # 信号明细
    candidates = [(n, v) for n, v in results.items() if v['signals'] >= 5 and '原版' not in n and '最佳' not in n]
    best_name = max(candidates, key=lambda x: (x[1]['win_rate'], x[1]['avg_max_return']))[0] if candidates else 'dividend_value (原版)'

    detail_rows = ""
    for algo_name in ['dividend_value (原版)', best_name]:
        r = results.get(algo_name, {})
        for s in r.get('signal_list', []):
            win_class = 'win' if s['is_win'] else 'loss'
            ret_c = '#27ae60' if s['close_return'] >= 0 else '#e74c3c'
            max_c = '#27ae60' if s['max_return'] >= 0 else '#e74c3c'
            reasons = '; '.join(s.get('reasons', [])) if s.get('reasons') else '-'
            detail_rows += f"""<tr class="{win_class}"><td>{algo_name}</td><td>{s['date']}</td><td>{s['score']:.1f}</td><td>{s['buy_price']:.3f}</td><td class="win-cell">{'Y' if s['is_win'] else 'N'}</td><td style="color:{max_c};font-weight:bold;">{s['max_return']:+.2f}%</td><td style="color:{ret_c};">{s['close_return']:+.2f}%</td><td>{s['days_to_win']}d</td><td class="reasons-cell">{reasons}</td></tr>"""

    # 结论
    orig = results.get('dividend_value (原版)', {})
    ner = results.get('new_energy_reversal (最佳基准)', {})
    if candidates:
        best_r = results[best_name]
        if best_r['win_rate'] > max(orig.get('win_rate', 0), ner.get('win_rate', 0)):
            conclusion = f"最优变体: {best_name} (胜率{best_r['win_rate']:.1f}%), 超越原版({orig.get('win_rate',0):.1f}%)和new_energy({ner.get('win_rate',0):.1f}%), 建议采用"
            conclusion_color = '#27ae60'
        else:
            conclusion = f"变体{best_name}胜率{best_r['win_rate']:.1f}%, 未超越new_energy({ner.get('win_rate',0):.1f}%), 但包含股息率因子, 综合评估"
            conclusion_color = '#e67e22'
    else:
        conclusion = "无有效变体"
        conclusion_color = '#3498db'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>红利ETF 股息率优化对比</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif; background:#f5f6fa; color:#2c3e50; padding:20px; line-height:1.6; }}
.header {{ background:linear-gradient(135deg,#1a1a2e,#16213e); color:white; padding:25px; border-radius:12px; margin-bottom:20px; }}
.header h1 {{ font-size:22px; margin-bottom:8px; }}
.header .meta {{ opacity:0.8; font-size:13px; }}
.conclusion {{ background:{conclusion_color}; color:white; padding:15px 20px; border-radius:10px; margin-bottom:20px; font-size:15px; }}
.section {{ background:white; border-radius:12px; padding:20px; margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,0.08); overflow-x:auto; }}
.section h2 {{ font-size:17px; margin-bottom:15px; border-bottom:2px solid #ecf0f1; padding-bottom:10px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ background:#f8f9fa; padding:10px; text-align:left; border-bottom:2px solid #dee2e6; white-space:nowrap; }}
td {{ padding:8px 10px; border-bottom:1px solid #ecf0f1; }}
tr.win {{ background:#f0fff4; }} tr.loss {{ background:#fff5f5; }}
.win-cell {{ font-weight:bold; text-align:center; }}
tr.win .win-cell {{ color:#27ae60; }} tr.loss .win-cell {{ color:#e74c3c; }}
.reasons-cell {{ font-size:11px; color:#555; max-width:350px; }}
.footer {{ text-align:center; padding:20px; color:#95a5a6; font-size:12px; }}
</style></head><body>
<div class="header"><h1>红利ETF 股息率因子优化对比</h1>
<div class="meta">ETF: {ETF_CODE} {ETF_NAME} | 数据: {df_info} | 回测: {START_DATE}~最新 | 阈值: &ge;{THRESHOLD}分 | 胜率: T+3最高价&gt;0.5%</div></div>
<div class="conclusion">{conclusion}</div>
<div class="section"><h2>算法对比摘要 (按胜率降序)</h2>
<table><tr><th>算法</th><th>信号</th><th>胜利</th><th>胜率</th><th>均收益</th><th>均最大</th><th>均分</th><th>Sharpe</th><th>最佳</th><th>最差</th></tr>
{summary_rows}</table></div>
<div class="section"><h2>逐年对比</h2>
<table><tr><th>算法</th><th>年份</th><th>信号</th><th>胜率</th><th>均收益</th></tr>
{yearly_rows}</table></div>
<div class="section"><h2>原版 vs 最优变体 - 信号明细</h2>
<table><tr><th>算法</th><th>日期</th><th>信号分</th><th>买入价</th><th>胜</th><th>最大收益</th><th>收盘收益</th><th>达标天数</th><th>理由</th></tr>
{detail_rows}</table></div>
<div class="footer"><p>红利ETF算法优化 | 股息率因子 | T+3胜率回测</p><p>本报告仅供参考,不构成投资建议。</p></div>
</body></html>"""
    return html


def main():
    logger.info(f"开始 {ETF_NAME} 股息率优化对比...")

    engine = DataEngine()
    df = engine.get_history_kline(ETF_CODE)
    if df is None:
        logger.error(f"无法获取 {ETF_CODE} 数据")
        return

    df_info = f"{len(df)}条, {df['date'].iloc[0].strftime('%Y-%m-%d')}~{df['date'].iloc[-1].strftime('%Y-%m-%d')}"
    logger.info(f"{ETF_NAME} 数据: {df_info}")

    # 加载股息率数据
    div_df = load_dividend_history()
    df = add_dividend_yield(df, div_df)

    # 计算技术指标
    df = calc_all_indicators(df)

    # 运行回测
    algorithms = {
        'dividend_value (原版)': DividendValueAlgorithm(),
        'new_energy_reversal (最佳基准)': NewEnergyReversalAlgorithm(),
        'A. 股息率核心版': DividendValueYieldCore(),
        'B. new_energy+股息率': DividendValueYieldEnhanced(),
        'C. 股息率重权版': DividendValueYieldHeavy(),
    }

    results = {}
    for name, algo in algorithms.items():
        logger.info(f"  回测 {name}...")
        r = run_backtest(df, algo)
        results[name] = r
        if r['signals'] > 0:
            logger.info(f"  {name:30s} | 信号:{r['signals']:3d} | 胜率:{r['win_rate']:.1f}% | 收益:{r['avg_return']:+.2f}% | 最大:{r['avg_max_return']:+.2f}% | Sharpe:{r['sharpe']:.3f}")
        else:
            logger.info(f"  {name:30s} | 信号:  0")

    # 保存JSON
    json_data = {k: {kk: vv for kk, vv in v.items() if kk != 'signal_list'} for k, v in results.items()}
    json_path = os.path.join(OUTPUT_DIR, '红利ETF_股息率优化.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON: {json_path}")

    # 生成HTML
    html = generate_html(results, df_info)
    html_path = os.path.join(OUTPUT_DIR, '红利ETF_股息率优化.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML: {html_path}")

    # 打印摘要
    print(f"\n{'=' * 95}")
    print(f"红利ETF 股息率优化对比 (按胜率降序)")
    print(f"{'=' * 95}")
    print(f"{'算法':<32s} {'信号':>4s} {'胜率':>6s} {'均收益':>7s} {'均最大':>7s} {'Sharpe':>7s}")
    print("-" * 95)
    for name, r in sorted(results.items(), key=lambda x: (-x[1]['win_rate'], -x[1]['signals'])):
        marker = ''
        if '原版' in name:
            marker = ' <- 原版'
        elif '最佳' in name:
            marker = ' <- 最佳'
        print(f"{name:<32s} {r['signals']:4d} {r['win_rate']:5.1f}% {r['avg_return']:+6.2f}% {r['avg_max_return']:+6.2f}% {r['sharpe']:6.3f}{marker}")


if __name__ == '__main__':
    main()
