# -*- coding: utf-8 -*-
"""
数据获取引擎 - 基于akshare
============================

功能：
1. 获取ETF实时行情（14:45价格）
2. 获取历史K线数据（用于回测和指标计算）
3. 数据缓存机制
4. 批量获取所有ETF数据

数据源：
- fund_etf_spot_em(): 实时行情（东方财富）
- fund_etf_hist_sina(): 历史K线（新浪，稳定可靠）
"""

import os
import json
import logging
import time
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataEngine:
    """数据获取引擎"""

    def __init__(self, cache_dir: str = None):
        """
        初始化数据引擎

        Args:
            cache_dir: 缓存目录
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 内存缓存
        self._spot_cache = None
        self._spot_cache_time = None
        self._kline_cache = {}

        logger.info(f"数据引擎初始化完成，缓存目录: {cache_dir}")

    def _import_akshare(self):
        """延迟导入akshare"""
        try:
            import akshare as ak
            return ak
        except ImportError:
            raise ImportError("akshare未安装，请运行: pip install akshare")

    def get_realtime_quotes(self, use_cache: bool = True) -> pd.DataFrame:
        """
        获取所有ETF实时行情

        Returns:
            DataFrame: 实时行情数据
        """
        # 5分钟缓存
        if use_cache and self._spot_cache is not None and self._spot_cache_time is not None:
            age = (datetime.now() - self._spot_cache_time).total_seconds()
            if age < 300:  # 5分钟
                return self._spot_cache

        ak = self._import_akshare()

        for attempt in range(3):
            try:
                df = ak.fund_etf_spot_em()
                # 重命名列
                df = df.rename(columns={
                    '代码': 'code',
                    '名称': 'name',
                    '最新价': 'price',
                    '涨跌幅': 'change_pct',
                    '涨跌额': 'change_amt',
                    '开盘价': 'open',
                    '最高价': 'high',
                    '最低价': 'low',
                    '昨收': 'prev_close',
                    '成交量': 'volume',
                    '成交额': 'turnover',
                    '换手率': 'turnover_rate',
                    '振幅': 'amplitude',
                    '量比': 'volume_ratio',
                    '委比': 'order_ratio',
                    '基金折价率': 'discount_rate',
                })

                self._spot_cache = df
                self._spot_cache_time = datetime.now()
                logger.info(f"获取实时行情成功，共{len(df)}只ETF")
                return df

            except Exception as e:
                logger.warning(f"获取实时行情失败(尝试{attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(2)

        raise RuntimeError("获取实时行情失败，请检查网络连接")

    def get_etf_realtime(self, code: str) -> Optional[Dict]:
        """
        获取单只ETF实时行情

        Args:
            code: 6位代码（如 512890）

        Returns:
            行情字典
        """
        try:
            df = self.get_realtime_quotes()
            row = df[df['code'] == code]
            if row.empty:
                logger.warning(f"未找到ETF: {code}")
                return None

            r = row.iloc[0]
            return {
                'code': code,
                'name': r.get('name', ''),
                'price': float(r.get('price', 0)),
                'change_pct': float(r.get('change_pct', 0)),
                'open': float(r.get('open', 0)),
                'high': float(r.get('high', 0)),
                'low': float(r.get('low', 0)),
                'prev_close': float(r.get('prev_close', 0)),
                'volume': float(r.get('volume', 0)),
                'turnover': float(r.get('turnover', 0)),
                'turnover_rate': float(r.get('turnover_rate', 0)),
                'volume_ratio': float(r.get('volume_ratio', 0)),
                'discount_rate': float(r.get('discount_rate', 0)) if 'discount_rate' in row.columns else 0.0,
            }
        except Exception as e:
            logger.error(f"获取ETF {code} 实时行情失败: {e}")
            return None

    def get_history_kline(
        self,
        code: str,
        period: str = 'daily',
        adjust: str = 'qfq'
    ) -> Optional[pd.DataFrame]:
        """
        获取ETF历史K线数据

        使用 fund_etf_hist_sina（新浪数据源，稳定可靠）

        Args:
            code: sh/sz+6位代码（如 sh512890）
            period: 周期 daily/weekly/monthly
            adjust: 复权类型 qfq前复费/hfq后复权/''不复权

        Returns:
            DataFrame: K线数据
        """
        # 检查内存缓存
        cache_key = f"{code}_{period}_{adjust}"
        if cache_key in self._kline_cache:
            cached = self._kline_cache[cache_key]
            if (datetime.now() - cached['time']).total_seconds() < 3600:  # 1小时缓存
                return cached['data'].copy()

        # 检查磁盘缓存（当日有效）
        disk_cache_path = os.path.join(
            self.cache_dir,
            f"kline_{code}_{period}_{adjust}.pkl"
        )
        if os.path.exists(disk_cache_path):
            mtime = datetime.fromtimestamp(os.path.getmtime(disk_cache_path))
            if mtime.date() == datetime.now().date():
                try:
                    with open(disk_cache_path, 'rb') as f:
                        data = pickle.load(f)
                    self._kline_cache[cache_key] = {'data': data, 'time': mtime}
                    logger.debug(f"从磁盘缓存加载 {code} K线数据: {len(data)}条")
                    return data.copy()
                except Exception as e:
                    logger.debug(f"磁盘缓存加载失败: {e}")

        # 从API获取
        ak = self._import_akshare()

        for attempt in range(3):
            try:
                # fund_etf_hist_sina 的symbol格式: sh512890 或 sz159901
                df = ak.fund_etf_hist_sina(symbol=code)

                if df is None or df.empty:
                    logger.warning(f"获取 {code} K线数据为空")
                    return None

                # fund_etf_hist_sina 返回的列名:
                # date, open, high, low, close, volume, amount, postVol, postAmt
                # 注意: Sina返回的已经是前复权数据

                # 确保日期格式
                df['date'] = pd.to_datetime(df['date'])

                # 按日期排序
                df = df.sort_values('date').reset_index(drop=True)

                # 添加额外字段
                df['change_pct'] = df['close'].pct_change() * 100

                # 缓存
                self._kline_cache[cache_key] = {'data': df, 'time': datetime.now()}

                # 保存到磁盘
                try:
                    with open(disk_cache_path, 'wb') as f:
                        pickle.dump(df, f)
                except Exception as e:
                    logger.debug(f"磁盘缓存保存失败: {e}")

                logger.info(f"获取 {code} K线数据成功: {len(df)}条, "
                           f"范围 {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ "
                           f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}")

                return df

            except Exception as e:
                logger.warning(f"获取 {code} K线失败(尝试{attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(2)

        logger.error(f"获取 {code} K线数据最终失败")
        return None

    def get_dividend_yield(self, code: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        为K线数据添加股息率列（仅适用于有分红历史的ETF）

        Args:
            code: ETF代码（如 sh510880）
            df: K线DataFrame

        Returns:
            添加了 ttm_dividend, dividend_yield, dividend_yield_pct 列的df
        """
        try:
            ak = self._import_akshare()
            div_df = ak.fund_etf_dividend_sina(symbol=code)
            if div_df is None or div_df.empty:
                logger.warning(f"无分红历史: {code}")
                df['dividend_yield'] = 0.0
                df['dividend_yield_pct'] = 50.0
                return df

            # 列名: 除息日, 累计分红
            div_df.columns = ['ex_date', 'cum_dividend']
            div_df['ex_date'] = pd.to_datetime(div_df['ex_date'])
            div_df = div_df.sort_values('ex_date').reset_index(drop=True)
            div_df['dividend'] = div_df['cum_dividend'].diff().fillna(div_df['cum_dividend'].iloc[0])

            # 为每个交易日计算TTM分红
            div_records = [(row['ex_date'], row['dividend']) for _, row in div_df.iterrows()]
            ttm_divs = []
            for _, krow in df.iterrows():
                date = krow['date']
                ttm = 0
                for ex_date, div_amt in div_records:
                    if ex_date <= date:
                        ttm = div_amt
                    else:
                        break
                ttm_divs.append(ttm)

            df = df.copy()
            df['ttm_dividend'] = ttm_divs
            df['dividend_yield'] = np.where(df['close'] > 0,
                                             df['ttm_dividend'] / df['close'] * 100, 0)
            # 250日滚动百分位
            df['dividend_yield_pct'] = df['dividend_yield'].rolling(
                window=250, min_periods=60).rank(pct=True) * 100
            df['dividend_yield_pct'] = df['dividend_yield_pct'].fillna(50)

            logger.info(f"股息率加载完成: {code}, "
                       f"范围{df['dividend_yield'].min():.2f}%~{df['dividend_yield'].max():.2f}%")
            return df

        except Exception as e:
            logger.warning(f"加载股息率失败({code}): {e}")
            df['dividend_yield'] = 0.0
            df['dividend_yield_pct'] = 50.0
            return df

    def get_history_by_symbol(self, symbol: str, period: str = 'daily') -> Optional[pd.DataFrame]:
        """
        通过6位数字代码获取历史K线

        Args:
            symbol: 6位代码（如 512890）
            period: 周期

        Returns:
            DataFrame
        """
        # 自动判断sh/sz前缀
        if symbol.startswith(('5', '6', '9')):
            code = f"sh{symbol}"
        else:
            code = f"sz{symbol}"

        return self.get_history_kline(code, period)

    def batch_get_realtime(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        批量获取多只ETF实时行情

        Args:
            symbols: 6位代码列表

        Returns:
            {code: quote_dict} 字典
        """
        results = {}
        df = self.get_realtime_quotes()

        for symbol in symbols:
            row = df[df['code'] == symbol]
            if not row.empty:
                r = row.iloc[0]
                results[symbol] = {
                    'code': symbol,
                    'name': r.get('name', ''),
                    'price': float(r.get('price', 0)),
                    'change_pct': float(r.get('change_pct', 0)),
                    'open': float(r.get('open', 0)),
                    'high': float(r.get('high', 0)),
                    'low': float(r.get('low', 0)),
                    'prev_close': float(r.get('prev_close', 0)),
                    'volume': float(r.get('volume', 0)),
                    'turnover': float(r.get('turnover', 0)),
                    'turnover_rate': float(r.get('turnover_rate', 0)),
                    'volume_ratio': float(r.get('volume_ratio', 0)),
                    'discount_rate': float(r.get('discount_rate', 0)) if 'discount_rate' in df.columns else 0.0,
                }
            else:
                logger.warning(f"未找到ETF: {symbol}")

        return results

    def batch_get_history(
        self,
        codes: List[str],
        period: str = 'daily'
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取历史K线数据

        Args:
            codes: sh/sz+6位代码列表
            period: 周期

        Returns:
            {code: DataFrame} 字典
        """
        results = {}
        for i, code in enumerate(codes, 1):
            logger.info(f"  [{i}/{len(codes)}] 获取 {code} 历史数据...")
            df = self.get_history_kline(code, period)
            if df is not None:
                results[code] = df
            else:
                logger.warning(f"  {code} 获取失败")
            # 避免请求过快
            if i < len(codes):
                time.sleep(0.3)

        return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    engine = DataEngine()

    # 测试实时行情
    print("\n=== 测试实时行情 ===")
    quote = engine.get_etf_realtime('513100')
    if quote:
        print(f"513100 纳指ETF: 价格={quote['price']}, 涨跌={quote['change_pct']}%")

    # 测试历史K线
    print("\n=== 测试历史K线 ===")
    df = engine.get_history_kline('sh513100')
    if df is not None:
        print(f"获取 {len(df)} 条K线")
        print(f"列名: {df.columns.tolist()}")
        print(df.tail(3).to_string())
