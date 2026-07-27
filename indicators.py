# -*- coding: utf-8 -*-
"""
技术指标库
==========

所有算法共用的技术指标计算函数
指标全部基于pandas DataFrame中的OHLCV数据

包含指标：
- RSI（相对强弱指数）
- MACD（指数平滑异同移动平均线）
- Bollinger Bands（布林带）
- ATR（平均真实波幅）
- Z-Score（标准化偏离）
- 移动平均线（MA5/10/20/60/200）
- 连续涨跌天数
- 成交量比
- 布林带宽度和%B
- 价格动量
- KDJ随机指标
- 支撑/阻力位
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算RSI（相对强弱指数）

    RSI < 30: 超卖
    RSI > 70: 超买
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 处理全涨或全跌的情况
    rsi = rsi.fillna(100).clip(0, 100)

    return rsi


def calc_rsi_fast(close: pd.Series, period: int = 7) -> pd.Series:
    """快速RSI（7日），更敏感"""
    return calc_rsi(close, period)


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算MACD

    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calc_bollinger(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    计算布林带

    Returns:
        (upper, middle, lower, bandwidth, percent_b)
        bandwidth = (upper - lower) / middle
        percent_b = (close - lower) / (upper - lower)
    """
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()

    upper = middle + num_std * std
    lower = middle - num_std * std

    bandwidth = (upper - lower) / middle
    percent_b = (close - lower) / (upper - lower).replace(0, np.nan)

    return upper, middle, lower, bandwidth, percent_b


def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """计算ATR（平均真实波幅）"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calc_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """
    计算Z-Score（标准化偏离度）

    Z < -2: 严重低于均值
    Z > 2: 严重高于均值
    """
    mean = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    zscore = (close - mean) / std.replace(0, np.nan)
    return zscore.fillna(0)


def calc_moving_averages(close: pd.Series) -> Dict[str, pd.Series]:
    """计算多周期移动平均线"""
    return {
        'ma5': close.rolling(window=5).mean(),
        'ma10': close.rolling(window=10).mean(),
        'ma20': close.rolling(window=20).mean(),
        'ma60': close.rolling(window=60).mean(),
        'ma120': close.rolling(window=120).mean(),
        'ma200': close.rolling(window=200).mean(),
        'ma40w': close.rolling(window=200).mean(),  # 40周≈200日
    }


def calc_consecutive_days(close: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    计算连续上涨/下跌天数

    Returns:
        (consecutive_up, consecutive_down)
    """
    change = close.diff()
    is_up = (change > 0).astype(int)
    is_down = (change < 0).astype(int)

    # 计算连续上涨天数
    consec_up = pd.Series(index=close.index, dtype=int)
    consec_down = pd.Series(index=close.index, dtype=int)

    for i in range(len(close)):
        if i == 0:
            consec_up.iloc[i] = 0
            consec_down.iloc[i] = 0
        else:
            if is_up.iloc[i]:
                consec_up.iloc[i] = consec_up.iloc[i-1] + 1
                consec_down.iloc[i] = 0
            elif is_down.iloc[i]:
                consec_down.iloc[i] = consec_down.iloc[i-1] + 1
                consec_up.iloc[i] = 0
            else:
                consec_up.iloc[i] = 0
                consec_down.iloc[i] = 0

    return consec_up, consec_down


def calc_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    计算量比（当前成交量/过去N日平均成交量）
    """
    avg_vol = volume.rolling(window=window).mean()
    return (volume / avg_vol.replace(0, np.nan)).fillna(1)


def calc_momentum(close: pd.Series, window: int = 20) -> pd.Series:
    """
    计算动量（过去N日收益率%）
    """
    return (close / close.shift(window) - 1) * 100


def calc_kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算KDJ随机指标

    Returns:
        (k, d, j)
    """
    low_min = low.rolling(window=n).min()
    high_max = high.rolling(window=n).max()

    rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100

    k = rsv.ewm(com=m1-1, adjust=False).mean()
    d = k.ewm(com=m2-1, adjust=False).mean()
    j = 3 * k - 2 * d

    return k.fillna(50), d.fillna(50), j.fillna(50)


def calc_support_resistance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 60
) -> Tuple[float, float]:
    """
    计算支撑位和阻力位

    基于过去window周期的最低价和最高价

    Returns:
        (support, resistance)
    """
    support = float(low.rolling(window=window).min().iloc[-1])
    resistance = float(high.rolling(window=window).max().iloc[-1])
    return support, resistance


def calc_price_deviation(close: pd.Series, ma: pd.Series) -> pd.Series:
    """
    计算价格偏离度（价格相对均线的偏离百分比）

    正值=价格在均线上方（超买方向）
    负值=价格在均线下方（超卖方向）
    """
    return (close - ma) / ma.replace(0, np.nan) * 100


def calc_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """
    计算波动率（N日收益率标准差 * sqrt(252)）
    """
    returns = close.pct_change()
    vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
    return vol


def calc_atr_percent(close: pd.Series, atr: pd.Series) -> pd.Series:
    """
    计算ATR百分比（ATR占价格的百分比）
    用于衡量相对波动率
    """
    return (atr / close * 100).replace(0, np.nan)


def calc_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    一次性计算所有技术指标，附加到原DataFrame

    Args:
        df: 原始K线数据（包含 date, open, high, low, close, volume）

    Returns:
        附加了所有指标的DataFrame
    """
    result = df.copy()

    close = result['close']
    high = result['high']
    low = result['low']
    volume = result['volume']

    # RSI
    result['rsi_14'] = calc_rsi(close, 14)
    result['rsi_7'] = calc_rsi_fast(close, 7)

    # MACD
    macd, signal, hist = calc_macd(close)
    result['macd'] = macd
    result['macd_signal'] = signal
    result['macd_hist'] = hist

    # 布林带
    bb_upper, bb_middle, bb_lower, bb_bw, bb_pctb = calc_bollinger(close)
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_middle
    result['bb_lower'] = bb_lower
    result['bb_bandwidth'] = bb_bw
    result['bb_percent_b'] = bb_pctb

    # ATR
    result['atr_14'] = calc_atr(high, low, close, 14)
    result['atr_pct'] = calc_atr_percent(close, result['atr_14'])

    # Z-Score
    result['zscore_20'] = calc_zscore(close, 20)

    # 移动平均线
    for ma_name, ma_series in calc_moving_averages(close).items():
        result[ma_name] = ma_series

    # 偏离度
    result['dev_ma20'] = calc_price_deviation(close, result['ma20'])
    result['dev_ma60'] = calc_price_deviation(close, result['ma60'])
    result['dev_ma200'] = calc_price_deviation(close, result['ma200'])

    # 连续涨跌天数
    consec_up, consec_down = calc_consecutive_days(close)
    result['consec_up'] = consec_up
    result['consec_down'] = consec_down

    # 量比
    result['vol_ratio_20'] = calc_volume_ratio(volume, 20)

    # 动量
    result['momentum_5'] = calc_momentum(close, 5)
    result['momentum_10'] = calc_momentum(close, 10)
    result['momentum_20'] = calc_momentum(close, 20)

    # KDJ
    k, d, j = calc_kdj(high, low, close)
    result['kdj_k'] = k
    result['kdj_d'] = d
    result['kdj_j'] = j

    # 波动率
    result['volatility_20'] = calc_volatility(close, 20)

    # 3日最高价（用于T+3胜率计算，只在回测中使用）
    result['high_3d'] = high.rolling(window=3).max().shift(-3)  # 未来3日最高价
    # 3日收益率（未来3日最高价相对当前收盘的收益率）
    result['return_3d_max'] = (result['high_3d'] / close - 1) * 100

    return result
