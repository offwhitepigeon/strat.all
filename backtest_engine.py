# -*- coding: utf-8 -*-
"""
T+3胜率回测引擎
================

核心目标：验证每日14:45发出的买入信号在T+3（3个交易日内）的胜率

胜率定义：
- T+3胜 = 买入后3个交易日内的最高价 > 买入价 + 0.5%
- 胜率 = 胜的次数 / 总信号次数

回测流程：
1. 对每只ETF，加载历史K线数据
2. 逐日计算技术指标和信号
3. 当信号分 >= 阈值时，记录"买入"
4. 检查未来3日最高价是否满足胜利条件
5. 统计各算法/各ETF的胜率

输出：
- 各算法的胜率统计
- 各ETF的胜率统计
- 信号分布
- 最优信号阈值
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

from etf_config import ETF_POOL, ETFTarget
from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import get_algorithm

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """单条信号记录"""
    date: str               # 信号日期
    exit_date: str = ''     # T+N退出日期（资金释放日）
    etf_code: str = ''      # ETF代码
    etf_name: str = ''      # ETF名称
    algorithm: str = ''     # 算法
    buy_price: float = 0.0  # 买入价格（收盘价）
    score: float = 0.0      # 信号分
    level: str = ''         # 信号等级
    action: str = ''        # 操作建议
    # T+3结果
    high_3d: float = 0.0    # 未来3日最高价
    max_return_3d: float = 0.0  # T+3最大收益率%
    is_win: bool = False    # 是否胜利
    hold_days_to_win: int = 0  # 达到胜利条件所需天数
    close_3d: float = 0.0  # T+3收盘价
    return_3d: float = 0.0  # T+3收盘收益率%


@dataclass
class BacktestStats:
    """回测统计"""
    algorithm: str
    etf_name: str
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_score: float = 0.0
    avg_return_3d: float = 0.0
    avg_max_return: float = 0.0
    median_return_3d: float = 0.0
    best_return: float = 0.0
    worst_return: float = 0.0


class BacktestEngine:
    """T+3胜率回测引擎"""

    def __init__(self, cache_dir: str = None, cooldown_days: int = 0):
        """初始化回测引擎

        Args:
            cache_dir: 缓存目录
            cooldown_days: 信号冷却期（交易日），同一ETF发信号后N日内不再发新信号。
                          默认0=关闭，由组合级资金管理控制仓位；可选设为3/5作为额外安全网。
        """
        self.data_engine = DataEngine(cache_dir=cache_dir)
        self.etf_pool = ETF_POOL
        self.win_threshold = 0.5  # 胜利条件：收益>0.5%
        self.hold_days = 3         # 持有天数
        self.cooldown_days = cooldown_days  # 信号冷却期（可选安全网）

        logger.info(f"回测引擎初始化，胜率阈值={self.win_threshold}%，持有{self.hold_days}日，冷却{self.cooldown_days}日")

    def run_backtest(
        self,
        signal_threshold: int = 60,
        start_date: str = None,
        end_date: str = None,
        min_data_days: int = 250
    ) -> Dict[str, Any]:
        """
        执行完整回测

        Args:
            signal_threshold: 信号阈值，只有>=此分才记录信号
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            min_data_days: 最少需要的历史数据天数

        Returns:
            完整回测结果
        """
        start_time = datetime.now()
        logger.info(f"\n{'='*70}")
        logger.info(f"开始T+3胜率回测")
        logger.info(f"{'='*70}")
        logger.info(f"信号阈值: >={signal_threshold}分")
        logger.info(f"胜率定义: T+3最高价收益 > {self.win_threshold}%")
        logger.info(f"数据范围: {start_date or '最早'} ~ {end_date or '最新'}\n")

        all_signals: List[SignalRecord] = []
        all_stats: List[BacktestStats] = []

        for i, etf in enumerate(self.etf_pool, 1):
            logger.info(f"[{i}/{len(self.etf_pool)}] 回测 {etf.code} ({etf.name}) - {etf.algorithm}")

            # 获取历史数据
            df = self.data_engine.get_history_kline(etf.code)
            if df is None or len(df) < min_data_days:
                logger.warning(f"  数据不足({0 if df is None else len(df)}条)，跳过")
                continue

            # 计算指标
            df = calc_all_indicators(df)

            # 过滤日期范围
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]

            if len(df) < 60:
                logger.warning(f"  过滤后数据不足，跳过")
                continue

            # 逐日回测
            signals = self._backtest_single_etf(df, etf, signal_threshold)
            all_signals.extend(signals)

            # 统计
            stats = self._calc_stats(signals, etf.algorithm, etf.name)
            all_stats.append(stats)

            if stats.total_signals > 0:
                logger.info(f"  信号数:{stats.total_signals} | "
                           f"胜率:{stats.win_rate:.1%} | "
                           f"平均收益:{stats.avg_return_3d:+.2f}% | "
                           f"平均最大收益:{stats.avg_max_return:+.2f}%")
            else:
                logger.info(f"  无有效信号")

        # 汇总统计
        summary = self._calc_summary(all_signals, all_stats)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"\n{'='*70}")
        logger.info(f"回测完成，耗时{duration:.1f}秒")
        logger.info(f"总信号数: {len(all_signals)}")
        if all_signals:
            total_wins = sum(1 for s in all_signals if s.is_win)
            logger.info(f"总胜率: {total_wins}/{len(all_signals)} = {total_wins/len(all_signals):.1%}")
        logger.info(f"{'='*70}\n")

        return {
            'metadata': {
                'run_time': start_time.isoformat(),
                'duration_seconds': round(duration, 1),
                'signal_threshold': signal_threshold,
                'win_threshold': self.win_threshold,
                'hold_days': 'per_etf (default=3, oil=5)',
                'cooldown_days': self.cooldown_days,
                'capital_management': 'portfolio' if self.cooldown_days == 0 else f'cooldown_{self.cooldown_days}d',
                'start_date': start_date,
                'end_date': end_date,
                'etf_count': len(all_stats),
            },
            'summary': summary,
            'stats_by_etf': [asdict(s) for s in all_stats],
            'signals': [asdict(s) for s in all_signals],
        }

    def _backtest_single_etf(
        self,
        df: pd.DataFrame,
        etf: ETFTarget,
        threshold: int
    ) -> List[SignalRecord]:
        """
        对单只ETF进行逐日回测

        策略：
        - 从第60个交易日开始（确保有足够指标数据）
        - 每天计算信号
        - 当信号>=阈值时记录买入
        - 信号冷却：同一ETF发信号后cooldown_days日内不再发新信号
        - 检查未来N日是否满足胜利条件（N=etf.hold_days, 默认3, 石油用5）
        """
        signals = []
        algorithm = get_algorithm(etf.algorithm)
        hold_days = getattr(etf, 'hold_days', 3)
        last_signal_idx = -self.cooldown_days  # 初始化为负值，允许第一个信号

        # 确保有未来N日数据
        max_idx = len(df) - hold_days - 1

        for i in range(60, max_idx):
            # 信号冷却检查：上次信号后N日内跳过
            if i - last_signal_idx < self.cooldown_days:
                continue

            # 截取到当日的数据
            df_slice = df.iloc[:i+1]

            # 计算信号
            try:
                signal = algorithm.calculate(df_slice)
            except Exception as e:
                continue

            # 只有信号>=阈值才记录
            if signal.score < threshold:
                continue

            # 获取买入价格（当日收盘价）
            buy_price = float(df.iloc[i]['close'])
            buy_date = df.iloc[i]['date']

            # 获取未来N日数据
            future = df.iloc[i+1:i+1+hold_days]

            if len(future) < 1:
                continue

            # 未来N日最高价
            future_high = float(future['high'].max())
            future_max_return = (future_high / buy_price - 1) * 100

            # 未来N日收盘价和收益
            future_close = float(future['close'].iloc[-1])
            future_return = (future_close / buy_price - 1) * 100

            # 判断是否胜利
            is_win = future_max_return > self.win_threshold

            # 计算达到胜利所需天数
            days_to_win = 0
            if is_win:
                for j, row in future.iterrows():
                    day_high = float(row['high'])
                    day_return = (day_high / buy_price - 1) * 100
                    if day_return > self.win_threshold:
                        days_to_win = list(future.index).index(j) + 1
                        break

            # 日期格式化
            if hasattr(buy_date, 'strftime'):
                date_str = buy_date.strftime('%Y-%m-%d')
            else:
                date_str = str(buy_date)[:10]

            # T+N退出日期
            exit_dt = future['date'].iloc[-1]
            if hasattr(exit_dt, 'strftime'):
                exit_date_str = exit_dt.strftime('%Y-%m-%d')
            else:
                exit_date_str = str(exit_dt)[:10]

            signals.append(SignalRecord(
                date=date_str,
                exit_date=exit_date_str,
                etf_code=etf.code,
                etf_name=etf.name,
                algorithm=etf.algorithm,
                buy_price=round(buy_price, 4),
                score=round(signal.score, 1),
                level=signal.level,
                action=signal.action,
                high_3d=round(future_high, 4),
                max_return_3d=round(future_max_return, 2),
                is_win=is_win,
                hold_days_to_win=days_to_win,
                close_3d=round(future_close, 4),
                return_3d=round(future_return, 2),
            ))
            # 更新上次信号索引（触发冷却）
            last_signal_idx = i

        return signals

    def _calc_stats(
        self,
        signals: List[SignalRecord],
        algorithm: str,
        etf_name: str
    ) -> BacktestStats:
        """计算单只ETF的回测统计"""
        stats = BacktestStats(algorithm=algorithm, etf_name=etf_name)

        if not signals:
            return stats

        stats.total_signals = len(signals)
        stats.wins = sum(1 for s in signals if s.is_win)
        stats.losses = stats.total_signals - stats.wins
        stats.win_rate = stats.wins / stats.total_signals if stats.total_signals > 0 else 0

        scores = [s.score for s in signals]
        returns_3d = [s.return_3d for s in signals]
        max_returns = [s.max_return_3d for s in signals]

        stats.avg_score = round(np.mean(scores), 1)
        stats.avg_return_3d = round(np.mean(returns_3d), 2)
        stats.avg_max_return = round(np.mean(max_returns), 2)
        stats.median_return_3d = round(np.median(returns_3d), 2)
        stats.best_return = round(max(max_returns), 2)
        stats.worst_return = round(min(returns_3d), 2)

        return stats

    def _calc_summary(
        self,
        all_signals: List[SignalRecord],
        all_stats: List[BacktestStats]
    ) -> Dict:
        """计算全局汇总"""
        if not all_signals:
            return {'total_signals': 0, 'total_win_rate': 0}

        total_wins = sum(1 for s in all_signals if s.is_win)
        total_signals = len(all_signals)

        # 按算法分组统计
        algo_stats = {}
        for sig in all_signals:
            if sig.algorithm not in algo_stats:
                algo_stats[sig.algorithm] = {'signals': [], 'wins': 0}
            algo_stats[sig.algorithm]['signals'].append(sig)
            if sig.is_win:
                algo_stats[sig.algorithm]['wins'] += 1

        algo_summary = {}
        for algo, data in algo_stats.items():
            sigs = data['signals']
            wins = data['wins']
            returns = [s.return_3d for s in sigs]
            max_returns = [s.max_return_3d for s in sigs]
            algo_summary[algo] = {
                'total_signals': len(sigs),
                'wins': wins,
                'win_rate': round(wins / len(sigs) * 100, 1) if sigs else 0,
                'avg_return_3d': round(np.mean(returns), 2) if returns else 0,
                'avg_max_return': round(np.mean(max_returns), 2) if max_returns else 0,
                'etfs': list(set(s.etf_name for s in sigs)),
            }

        # 按信号等级分组
        level_stats = {}
        for sig in all_signals:
            if sig.level not in level_stats:
                level_stats[sig.level] = {'signals': [], 'wins': 0}
            level_stats[sig.level]['signals'].append(sig)
            if sig.is_win:
                level_stats[sig.level]['wins'] += 1

        level_summary = {}
        for level, data in level_stats.items():
            sigs = data['signals']
            wins = data['wins']
            returns = [s.return_3d for s in sigs]
            level_summary[level] = {
                'total_signals': len(sigs),
                'wins': wins,
                'win_rate': round(wins / len(sigs) * 100, 1) if sigs else 0,
                'avg_return_3d': round(np.mean(returns), 2) if returns else 0,
            }

        # 按得分区间分组
        score_bins = [(60, 70), (70, 80), (80, 90), (90, 101)]
        score_summary = {}
        for lo, hi in score_bins:
            bin_sigs = [s for s in all_signals if lo <= s.score < hi]
            if bin_sigs:
                bin_wins = sum(1 for s in bin_sigs if s.is_win)
                returns = [s.return_3d for s in bin_sigs]
                score_summary[f'{lo}-{hi}'] = {
                    'total_signals': len(bin_sigs),
                    'wins': bin_wins,
                    'win_rate': round(bin_wins / len(bin_sigs) * 100, 1),
                    'avg_return_3d': round(np.mean(returns), 2),
                }

        return {
            'total_signals': total_signals,
            'total_wins': total_wins,
            'total_win_rate': round(total_wins / total_signals * 100, 1) if total_signals else 0,
            'avg_return_3d': round(np.mean([s.return_3d for s in all_signals]), 2),
            'avg_max_return': round(np.mean([s.max_return_3d for s in all_signals]), 2),
            'by_algorithm': algo_summary,
            'by_level': level_summary,
            'by_score_bin': score_summary,
        }

    def find_optimal_threshold(self) -> Dict[str, int]:
        """
        寻找各算法的最优信号阈值

        测试不同阈值(50/60/70/80/90)，找到胜率最高的阈值
        """
        logger.info(f"\n{'='*70}")
        logger.info("寻找各算法最优信号阈值...")
        logger.info(f"{'='*70}\n")

        thresholds = [50, 55, 60, 65, 70, 75, 80, 85, 90]
        results = {}

        for threshold in thresholds:
            logger.info(f"测试阈值 {threshold}...")
            bt_result = self.run_backtest(signal_threshold=threshold)
            summary = bt_result['summary']

            if summary['total_signals'] > 0:
                results[threshold] = {
                    'total_signals': summary['total_signals'],
                    'win_rate': summary['total_win_rate'],
                    'avg_return': summary['avg_return_3d'],
                }
                logger.info(f"  信号数:{summary['total_signals']} | "
                           f"胜率:{summary['total_win_rate']}% | "
                           f"平均收益:{summary['avg_return_3d']}%")

        # 找出各算法在各阈值下的胜率
        algo_optimal = {}
        for algo_name in set(etf.algorithm for etf in self.etf_pool):
            best_wr = 0
            best_th = 60
            best_count = 0

            for threshold, result in results.items():
                # 重新跑该阈值的结果
                bt = self.run_backtest(signal_threshold=threshold)
                for stat in bt['stats_by_etf']:
                    if stat['algorithm'] == algo_name and stat['total_signals'] > 0:
                        if stat['win_rate'] > best_wr:
                            best_wr = stat['win_rate']
                            best_th = threshold
                            best_count = stat['total_signals']

            algo_optimal[algo_name] = {
                'threshold': best_th,
                'win_rate': round(best_wr * 100, 1),
                'signal_count': best_count,
            }

        return algo_optimal


def main():
    """运行回测"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )

    engine = BacktestEngine()

    # 运行回测
    result = engine.run_backtest(
        signal_threshold=60,
        start_date='2024-01-01',  # 回测最近1年半
    )

    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'backtest')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"\n回测结果已保存: {output_file}")

    # 打印摘要
    summary = result['summary']
    print(f"\n{'='*70}")
    print(f"T+3胜率回测摘要")
    print(f"{'='*70}")
    print(f"总信号数: {summary['total_signals']}")
    print(f"总胜率: {summary['total_win_rate']}%")
    print(f"平均收益: {summary['avg_return_3d']}%")
    print(f"平均最大收益: {summary['avg_max_return']}%")

    print(f"\n按算法统计:")
    for algo, stats in sorted(summary.get('by_algorithm', {}).items(),
                                key=lambda x: -x[1]['win_rate']):
        print(f"  {algo:25s} | 信号:{stats['total_signals']:3d} | "
              f"胜率:{stats['win_rate']:5.1f}% | "
              f"平均收益:{stats['avg_return_3d']:+.2f}% | "
              f"ETFs: {', '.join(stats['etfs'])}")

    print(f"\n按信号等级统计:")
    for level, stats in summary.get('by_level', {}).items():
        print(f"  {level:12s} | 信号:{stats['total_signals']:3d} | "
              f"胜率:{stats['win_rate']:5.1f}%")

    print(f"\n按得分区间统计:")
    for bin_name, stats in summary.get('by_score_bin', {}).items():
        print(f"  {bin_name:8s} | 信号:{stats['total_signals']:3d} | "
              f"胜率:{stats['win_rate']:5.1f}% | "
              f"平均收益:{stats['avg_return_3d']:+.2f}%")


if __name__ == '__main__':
    main()
