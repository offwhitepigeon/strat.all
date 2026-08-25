# -*- coding: utf-8 -*-
"""
信号引擎 - 信号生成器
=====================

整合数据获取、指标计算和算法分发，生成每日交易信号

工作流程：
1. 获取所有ETF历史K线数据
2. 计算技术指标
3. 根据ETF的算法类型选择对应算法
4. 生成信号（信号分0-100、操作建议、仓位%）
5. 如果有实时价格，用14:45价格替换收盘价计算

纯信号系统：不包含资金管理/仓位控制逻辑，仅输出原始信号。
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from etf_config import ETF_POOL, ETFTarget
from data_engine import DataEngine
from indicators import calc_all_indicators
from algorithms import get_algorithm, SignalResult, get_signal_level

logger = logging.getLogger(__name__)


class SignalEngine:
    """信号生成引擎 — 纯信号输出，无资金管理"""

    def __init__(self, cache_dir: str = None):
        """
        初始化信号引擎

        Args:
            cache_dir: 数据缓存目录
        """
        self.data_engine = DataEngine(cache_dir=cache_dir)
        self.etf_pool = ETF_POOL

        # 预加载历史数据缓存
        self._history_cache: Dict[str, pd.DataFrame] = {}
        self._indicators_cache: Dict[str, pd.DataFrame] = {}

        logger.info(f"信号引擎初始化完成，共{len(self.etf_pool)}只ETF")

    def load_history_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有ETF历史数据并计算指标

        Returns:
            {etf_code: 指标DataFrame} 字典
        """
        logger.info(f"\n{'='*60}")
        logger.info("开始加载历史数据和计算指标...")
        logger.info(f"{'='*60}\n")

        result = {}

        for i, etf in enumerate(self.etf_pool, 1):
            logger.info(f"[{i}/{len(self.etf_pool)}] {etf.code} ({etf.name}) - 算法: {etf.algorithm}")

            # 获取历史K线
            df = self.data_engine.get_history_kline(etf.code)

            if df is None or len(df) < 60:
                logger.warning(f"  {etf.name} 数据不足，跳过")
                continue

            # 计算所有指标
            df_with_indicators = calc_all_indicators(df)

            # 红利ETF额外加载股息率数据
            if etf.code == 'sh510880':
                df_with_indicators = self.data_engine.get_dividend_yield(etf.code, df_with_indicators)

            # 缓存
            self._history_cache[etf.code] = df
            self._indicators_cache[etf.code] = df_with_indicators
            result[etf.code] = df_with_indicators

            last_date = df['date'].iloc[-1]
            last_close = df['close'].iloc[-1]
            logger.info(f"  数据: {len(df)}条, 最近日期={last_date.strftime('%Y-%m-%d')}, "
                       f"收盘={last_close:.3f}")

        logger.info(f"\n历史数据加载完成，成功{len(result)}/{len(self.etf_pool)}只\n")
        return result

    def generate_signals(
        self,
        realtime_prices: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        为所有ETF生成交易信号

        Args:
            realtime_prices: 实时数据字典 {symbol: price或quote_dict}
                             如果value是float: 仅价格（回测/简化模式）
                             如果value是dict: 含price, discount_rate等完整行情
            use_cache: 是否使用缓存的历史数据

        Returns:
            信号结果列表
        """
        # 加载历史数据
        if use_cache and self._indicators_cache:
            indicators_data = self._indicators_cache
        else:
            indicators_data = self.load_history_data()

        if not indicators_data:
            logger.error("无可用历史数据")
            return []

        signals = []
        run_time = datetime.now()

        logger.info(f"\n{'='*60}")
        logger.info(f"开始生成交易信号 | {run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}\n")

        for etf in self.etf_pool:
            if etf.code not in indicators_data:
                # 尝试用symbol匹配
                df = None
                for cached_code, cached_df in indicators_data.items():
                    if etf.symbol in cached_code or cached_code in etf.code:
                        df = cached_df
                        break
                if df is None:
                    continue
            else:
                df = indicators_data[etf.code]

            try:
                # 获取当前价格和额外数据（如折价率）
                current_price = None
                extra_data = None

                if realtime_prices and etf.symbol in realtime_prices:
                    rt_val = realtime_prices[etf.symbol]
                    if isinstance(rt_val, dict):
                        # 完整行情模式（含折价率等）
                        current_price = rt_val.get('price')
                        discount_rate = rt_val.get('discount_rate', 0)
                        extra_data = {'discount_rate': discount_rate}
                    else:
                        # 简单价格模式
                        current_price = float(rt_val)

                if current_price is None:
                    current_price = float(df['close'].iloc[-1])

                # 获取算法实例
                algorithm = get_algorithm(etf.algorithm)

                # 计算信号（传入extra_data）
                signal = algorithm.calculate(df, current_price=current_price,
                                           extra_data=extra_data)

                # 构建结果
                last_date = df['date'].iloc[-1]
                last_close = float(df['close'].iloc[-1])

                signal_dict = {
                    'etf_code': etf.code,
                    'etf_symbol': etf.symbol,
                    'etf_name': etf.name,
                    'sector': etf.sector,
                    'market': etf.market,
                    'algorithm': etf.algorithm,
                    'description': etf.description,

                    # 价格信息
                    'current_price': round(current_price, 4),
                    'last_close': round(last_close, 4),
                    'last_date': last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else str(last_date),

                    # 信号
                    'score': signal.score,
                    'level': signal.level,
                    'action': signal.action,
                    'position_pct': signal.position_pct,
                    'reasons': signal.reasons,
                    'indicators': signal.indicators,

                    # 运行信息
                    'run_time': run_time.strftime('%Y-%m-%d %H:%M:%S'),
                }

                signals.append(signal_dict)

                # 日志
                emoji = {
                    'STRONG_BUY': '🔴', 'BUY': '🟢', 'LIGHT_BUY': '🟡',
                    'WATCH': '⚪', 'WAIT': '⚫'
                }.get(signal.level, '⚫')

                logger.info(
                    f"  {emoji} {etf.name:12s} | "
                    f"算法:{etf.algorithm:20s} | "
                    f"分:{signal.score:5.1f} | "
                    f"{signal.action}"
                )

            except Exception as e:
                logger.error(f"  {etf.name} 信号计算失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # 按信号分排序
        signals.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"信号生成完成，共{len(signals)}只ETF")

        # 统计
        buy_signals = [s for s in signals if s['score'] >= 60]
        strong_signals = [s for s in signals if s['score'] >= 75]
        logger.info(f"  有效信号(>=60分): {len(buy_signals)}只")
        logger.info(f"  强信号(>=75分): {len(strong_signals)}只")
        if strong_signals:
            logger.info(f"  最强信号: {strong_signals[0]['etf_name']} ({strong_signals[0]['score']:.1f}分)")
        logger.info(f"{'='*60}\n")

        return signals

    def get_realtime_prices(self) -> Dict[str, Any]:
        """
        获取所有ETF的实时行情（含折价率等完整数据）

        Returns:
            {symbol: quote_dict} 字典，每个quote_dict含price, discount_rate等
        """
        try:
            quotes = self.data_engine.batch_get_realtime(
                [etf.symbol for etf in self.etf_pool]
            )

            logger.info(f"获取实时行情成功，共{len(quotes)}只")
            return quotes

        except Exception as e:
            logger.warning(f"获取实时行情失败，将使用收盘价: {e}")
            return {}

    def run_daily(self, use_realtime: bool = True) -> List[Dict[str, Any]]:
        """
        执行每日信号生成流程

        Args:
            use_realtime: 是否使用实时价格（14:45）

        Returns:
            信号列表
        """
        run_time = datetime.now()
        logger.info(f"\n{'#'*60}")
        logger.info(f"# 每日信号生成 | {run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'#'*60}\n")

        # 获取实时行情（含折价率）
        realtime_data = {}
        if use_realtime:
            realtime_data = self.get_realtime_prices()

        # 生成信号
        signals = self.generate_signals(realtime_prices=realtime_data)

        return signals


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        handlers=[logging.StreamHandler()]
    )

    engine = SignalEngine()
    signals = engine.run_daily(use_realtime=True)

    print(f"\n{'='*60}")
    print(f"信号汇总（共{len(signals)}只）:")
    print(f"{'='*60}")
    for s in signals:
        print(f"  {s['etf_name']:12s} | {s['algorithm']:20s} | "
              f"分:{s['score']:5.1f} | {s['action']} | "
              f"仓位:{s['position_pct']}%")
