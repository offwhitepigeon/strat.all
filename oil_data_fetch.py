# -*- coding: utf-8 -*-
"""
石油LOF与标普油气ETF数据获取 + COMEX原油价格
=============================================
标的:
  - sz162411: 华宝标普油气LOF (标普油气)
  - sz161129: 广发石油LOF (石油基金)
  - COMEX WTI原油价格
"""
import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
import numpy as np
from data_engine import DataEngine

de = DataEngine()

# 1. 获取ETF/LOF数据
print("=" * 60)
print("1. ETF/LOF数据获取")
print("=" * 60)

df_162411 = de.get_history_kline("sz162411")
df_161129 = de.get_history_kline("sz161129")

for name, df in [("sz162411 标普油气LOF", df_162411), ("sz161129 石油LOF", df_161129)]:
    if df is not None:
        print(f"  {name}: {len(df)}条, {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"    列: {df.columns.tolist()}")
    else:
        print(f"  {name}: 获取失败")

# 2. 获取COMEX WTI原油价格
print(f"\n{'=' * 60}")
print("2. COMEX WTI原油价格获取")
print("=" * 60)

oil_price = None
oil_source = None

# 方法1: akshare futures_foreign_hist
try:
    import akshare as ak
    print("  尝试 futures_foreign_hist(symbol='CL')...")
    oil_df = ak.futures_foreign_hist(symbol="CL")
    if oil_df is not None and not oil_df.empty:
        print(f"  成功! {len(oil_df)}条, 列: {oil_df.columns.tolist()}")
        print(f"  日期范围: {oil_df['日期'].iloc[0]} ~ {oil_df['日期'].iloc[-1]}" if '日期' in oil_df.columns else f"  前3行:\n{oil_df.head(3)}")
        oil_price = oil_df
        oil_source = "futures_foreign_hist"
except Exception as e:
    print(f"  futures_foreign_hist失败: {e}")

# 方法2: energy_oil_hist
if oil_price is None:
    try:
        import akshare as ak
        print("\n  尝试 energy_oil_hist...")
        # 尝试不同的symbol
        for sym in ["WTI", "Brent", "原油"]:
            try:
                oil_df = ak.energy_oil_hist(symbol=sym)
                if oil_df is not None and not oil_df.empty:
                    print(f"  energy_oil_hist(symbol={sym}) 成功! {len(oil_df)}条")
                    oil_price = oil_df
                    oil_source = f"energy_oil_hist({sym})"
                    break
            except:
                pass
    except Exception as e:
        print(f"  energy_oil_hist失败: {e}")

# 方法3: futures_global_em
if oil_price is None:
    try:
        import akshare as ak
        print("\n  尝试 futures_global_em...")
        oil_df = ak.futures_global_em()
        if oil_df is not None and not oil_df.empty:
            print(f"  成功! {len(oil_df)}条, 列: {oil_df.columns.tolist()}")
            # 筛选原油相关
            oil_related = oil_df[oil_df['名称'].str.contains('原油|WTI|布伦特|Brent', na=False)] if '名称' in oil_df.columns else pd.DataFrame()
            if not oil_related.empty:
                print(f"  原油相关: {oil_related['名称'].unique()}")
            oil_price = oil_df
            oil_source = "futures_global_em"
    except Exception as e:
        print(f"  futures_global_em失败: {e}")

# 方法4: 使用国际期货数据
if oil_price is None:
    try:
        import akshare as ak
        print("\n  尝试 index_zh_a_hist (使用国际油价指数)...")
        # 尝试获取国际油价
        oil_df = ak.index_us_stock_sina(symbol="CL")
        if oil_df is not None and not oil_df.empty:
            print(f"  成功! {len(oil_df)}条")
            oil_price = oil_df
            oil_source = "index_us_stock_sina"
    except Exception as e:
        print(f"  index_us_stock_sina失败: {e}")

# 方法5: futures_hf_spot
if oil_price is None:
    try:
        import akshare as ak
        print("\n  尝试 futures_hf_spot...")
        oil_df = ak.futures_hf_spot()
        if oil_df is not None and not oil_df.empty:
            print(f"  成功! {len(oil_df)}条, 列: {oil_df.columns.tolist()}")
            oil_price = oil_df
            oil_source = "futures_hf_spot"
    except Exception as e:
        print(f"  futures_hf_spot失败: {e}")

if oil_price is not None:
    print(f"\n  最终数据源: {oil_source}")
    print(f"  数据形状: {oil_price.shape}")
    print(f"  列名: {oil_price.columns.tolist()}")
    print(f"\n  前5行:")
    print(oil_price.head().to_string())
    print(f"\n  后5行:")
    print(oil_price.tail().to_string())

    # 保存原油数据
    import os
    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    oil_price.to_pickle(os.path.join(cache_dir, 'comex_oil_price.pkl'))
    print(f"\n  原油数据已保存到 data/cache/comex_oil_price.pkl")
else:
    print("\n  所有方法均失败，尝试获取新浪财经国际油价...")
    try:
        import akshare as ak
        # 方法6: sina 国际期货
        oil_df = ak.futures_foreign_sub(symbol="CL")
        if oil_df is not None and not oil_df.empty:
            print(f"  futures_foreign_sub 成功! {len(oil_df)}条")
            oil_price = oil_df
            oil_source = "futures_foreign_sub"
    except Exception as e:
        print(f"  futures_foreign_sub失败: {e}")

    if oil_price is None:
        # 方法7: 直接获取现货价格
        try:
            import akshare as ak
            print("\n  尝试获取现货价格指数...")
            # 尝试使用 oil_spot 或类似函数
            for func_name in ['oil_spot', 'energy_spot', 'commodity_spot']:
                if hasattr(ak, func_name):
                    try:
                        oil_df = getattr(ak, func_name)()
                        if oil_df is not None and not oil_df.empty:
                            print(f"  ak.{func_name}() 成功! {len(oil_df)}条")
                            oil_price = oil_df
                            oil_source = func_name
                            break
                    except:
                        pass
        except Exception as e:
            print(f"  现货价格获取失败: {e}")

    if oil_price is None:
        print("\n  无法获取COMEX原油价格数据")
        print("  将使用ETF自身数据作为替代分析")

print(f"\n{'=' * 60}")
print("数据获取完成")
print("=" * 60)
