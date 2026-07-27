# -*- coding: utf-8 -*-
# ETF盘中信号自动运行脚本 (不依赖QoderWork)
#
# 电脑开机/登录后自动运行, 交易日盘中定时生成信号并按需发邮件。
#
# 运行时间表(交易日):
#   9:45  首次运行  --conditional-mail (有>=60分信号才发邮件)
#   10:15 / 10:45 / 11:15 / 13:15 / 13:45 / 14:15  --conditional-mail
#   14:45 最后一次  固定发完整邮件+HTML附件
#
# 非交易日(周末/节假日)自动退出, 不执行任何操作。

import os
import sys
import time
import subprocess
import logging
from datetime import datetime

# ===== 配置 =====
PYTHON = r"C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
PROJECT_DIR = r"D:\workspace\strat.all"
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

# 运行时间表: (hour, minute, is_final)
SCHEDULE = [
    (9, 45, False),
    (10, 15, False),
    (10, 45, False),
    (11, 15, False),
    (13, 15, False),
    (13, 45, False),
    (14, 15, False),
    (14, 45, True),   # 最后一次, 固定发邮件
]

# ===== 日志 =====
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f'intraday_{datetime.now().strftime("%Y%m%d")}.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def is_trading_day():
    """
    判断今天是否为交易日
    1. 先查akshare交易日历(权威)
    2. 网络失败则退化为工作日判断(周一~周五)
    """
    now = datetime.now()

    # 周末直接返回False
    if now.weekday() >= 5:
        weekday_name = '六' if now.weekday() == 5 else '日'
        logger.info(f"今天是周{weekday_name}，非交易日")
        return False

    # 查akshare交易日历
    try:
        import akshare as ak
        trade_dates = ak.tool_trade_date_hist_sina()
        today_str = now.strftime('%Y-%m-%d')
        date_col = trade_dates['trade_date']
        if date_col.dtype == 'object':
            date_list = date_col.astype(str).tolist()
        else:
            date_list = date_col.dt.strftime('%Y-%m-%d').tolist()

        if today_str in date_list:
            logger.info(f"今天是交易日 ({today_str})")
            return True
        else:
            logger.info(f"今天{today_str}在交易日历中标记为非交易日(节假日)")
            return False
    except Exception as e:
        logger.warning(f"获取交易日历失败({e})，按工作日判断")
        return True  # 工作日默认为交易日


def run_signal(is_final=False):
    """运行信号生成"""
    cmd = [PYTHON, '-X', 'utf8', 'daily_run.py']
    if not is_final:
        cmd.append('--conditional-mail')

    logger.info(f"执行: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=300,  # 5分钟超时
        )
        if result.stdout:
            for line in result.stdout.strip().split('\n')[-20:]:
                logger.info(f"  {line}")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-10:]:
                logger.warning(f"  {line}")
        if result.returncode == 0:
            logger.info("运行成功")
        else:
            logger.error(f"运行失败, 返回码={result.returncode}")
    except subprocess.TimeoutExpired:
        logger.error("运行超时(5分钟)")
    except Exception as e:
        logger.error(f"运行异常: {e}")


def main():
    logger.info("=" * 50)
    logger.info("ETF盘中信号自动运行 启动")
    logger.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    # 1. 判断交易日
    if not is_trading_day():
        logger.info("非交易日, 程序退出")
        return

    # 2. 遍历时间表
    now = datetime.now()
    has_pending = False

    for hour, minute, is_final in SCHEDULE:
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if now > target:
            label = "最终" if is_final else "盘中"
            logger.info(f"  {hour:02d}:{minute:02d} ({label}) 已过, 跳过")
            continue

        has_pending = True
        wait_sec = (target - now).total_seconds()
        label = "最终发邮件" if is_final else "条件发邮件"
        logger.info(f"  等待 {hour:02d}:{minute:02d} ({label}), 还需 {wait_sec:.0f}秒 ({wait_sec/60:.1f}分钟)")

        if wait_sec > 0:
            time.sleep(wait_sec)

        logger.info(f"  >>> {hour:02d}:{minute:02d} 开始运行 ({label})")
        run_signal(is_final)

        now = datetime.now()

    if not has_pending:
        logger.info("今日所有运行时间已过, 程序退出")

    logger.info("ETF盘中信号自动运行 结束")


if __name__ == '__main__':
    main()
