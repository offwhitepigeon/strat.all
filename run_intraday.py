# -*- coding: utf-8 -*-
# ETF盘中信号自动运行脚本 (已弃用)
# ====================================
#
# 此脚本已被简化调度方式替代。
# 当前使用 Windows 任务计划每天 14:45 直接运行 daily_run.py。
#
# 如需恢复盘中多时间点运行，请安装旧版任务计划:
#   powershell -ExecutionPolicy Bypass -File setup_schedule_legacy.ps1
#
# 保留此文件仅供参考，不再被定时任务调用。

import os
import sys
import logging
from datetime import datetime

PYTHON = r"C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
PROJECT_DIR = r"D:\workspace\strat.all"
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

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

logger.warning("=" * 50)
logger.warning("run_intraday.py 已弃用")
logger.warning("当前调度方式: Windows任务计划每天14:45直接运行 daily_run.py")
logger.warning("如需使用此脚本，请手动执行: python run_intraday.py")
logger.warning("=" * 50)


def main():
    logger.info("此脚本已弃用，请使用 daily_run.py + Windows任务计划(14:45)")
    return


if __name__ == '__main__':
    main()
