# -*- coding: utf-8 -*-
"""
主运行脚本 - ETF多策略信号系统 v2.0
=====================================

每日14:45运行，基于当日14:45价格发出买入信号

功能：
1. 获取22只ETF的实时行情（14:45价格）
2. 加载历史K线数据计算技术指标
3. 根据每只ETF的算法类型生成信号
4. 输出信号分(0-100)、操作建议、仓位百分比
5. 生成HTML可视化报告
6. 保存历史数据供回测

使用方法：
    python main.py                    # 立即运行一次
    python main.py --schedule        # 定时模式（等待到14:45运行）
    python main.py --time 14:45     # 指定运行时间
    python main.py --backtest        # 运行回测
    python main.py --backtest --threshold 65  # 指定回测阈值

作者：AI量化团队
日期：2026-07-21
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 导入自定义模块
from signal_engine import SignalEngine
from backtest_engine import BacktestEngine
from report_generator import ReportGenerator

# 配置日志
def setup_logging(log_dir: str = 'logs'):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"etf_signal_{datetime.now().strftime('%Y%m%d')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )

    return logging.getLogger(__name__)


logger = setup_logging()


class ETFSignalSystem:
    """
    ETF多策略信号系统

    整合信号引擎、回测引擎和报告生成器
    """

    def __init__(self):
        """初始化系统"""
        self.signal_engine = SignalEngine()
        self.report_gen = ReportGenerator()

        # 数据存储目录
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.history_dir = os.path.join(self.data_dir, 'history')
        os.makedirs(self.history_dir, exist_ok=True)

        logger.info("ETF多策略信号系统初始化完成")

    def run_daily_analysis(self, use_realtime: bool = True) -> Dict:
        """
        执行每日分析流程

        Args:
            use_realtime: 是否使用实时价格（14:45）

        Returns:
            完整的分析结果字典
        """
        run_time = datetime.now()
        logger.info(f"\n{'#'*70}")
        logger.info(f"# ETF多策略信号系统 | {run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"# 22只ETF × 20种算法 | 胜率目标: T+3")
        logger.info(f"{'#'*70}\n")

        result = {
            'run_time': run_time.isoformat(),
            'date': run_time.strftime('%Y-%m-%d'),
            'total_etfs': len(self.signal_engine.etf_pool),
            'signals': [],
            'errors': [],
        }

        # ===== 第一步：生成信号 =====
        logger.info("📡 [1/3] 获取数据并生成信号...")
        try:
            signals = self.signal_engine.run_daily(use_realtime=use_realtime)
            result['signals'] = signals

            # 统计
            buy_count = sum(1 for s in signals if s['score'] >= 60)
            strong_count = sum(1 for s in signals if s['score'] >= 75)
            logger.info(f"  信号生成完成: {len(signals)}只ETF, "
                       f"有效信号{buy_count}只, 强信号{strong_count}只")

        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            result['errors'].append(f"信号生成失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        # ===== 第二步：加载回测统计 =====
        logger.info("\n📊 [2/3] 加载历史回测统计...")
        backtest_summary = self._load_backtest_summary()
        result['backtest_summary'] = backtest_summary

        # ===== 第三步：生成HTML报告 =====
        logger.info("\n📝 [3/3] 生成HTML报告...")
        try:
            report_path = self.report_gen.generate_signal_report(
                signals=result['signals'],
                backtest_summary=backtest_summary,
                timestamp=run_time
            )
            result['report_path'] = report_path
            logger.info(f"  ✅ 报告已生成: {report_path}")
        except Exception as e:
            logger.error(f"  ❌ 报告生成失败: {e}")
            result['errors'].append(f"报告生成失败: {e}")

        # ===== 保存历史数据 =====
        try:
            history_path = self._save_history(result)
            result['history_path'] = history_path
            logger.info(f"  💾 历史数据已保存: {history_path}")
        except Exception as e:
            logger.error(f"  ❌ 历史数据保存失败: {e}")

        # ===== 打印汇总 =====
        self._print_summary(result)

        return result

    def _load_backtest_summary(self) -> Optional[Dict]:
        """加载最近的回测结果摘要"""
        try:
            backtest_dir = os.path.join(self.data_dir, 'backtest')
            if not os.path.exists(backtest_dir):
                return None

            # 找最新的回测文件
            files = [f for f in os.listdir(backtest_dir) if f.startswith('backtest_')]
            if not files:
                return None

            files.sort(reverse=True)
            latest_file = os.path.join(backtest_dir, files[0])

            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"  加载回测结果: {latest_file}")
            return data.get('summary', {})

        except Exception as e:
            logger.warning(f"加载回测统计失败: {e}")
            return None

    def _save_history(self, result: Dict) -> str:
        """保存历史数据到JSON文件"""
        date_str = result['date'].replace('-', '')
        time_str = result.get('run_time', '')[:19].split('T')[-1].replace(':', '')[:6] if 'run_time' in result else '000000'
        filename = f"signal_{date_str}_{time_str}.json"
        filepath = os.path.join(self.history_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        return filepath

    def _print_summary(self, result: Dict):
        """打印分析结果摘要"""
        logger.info(f"\n{'='*70}")
        logger.info("📊 每日信号摘要")
        logger.info(f"{'='*70}")

        signals = result.get('signals', [])
        if not signals:
            logger.warning("  无有效信号")
            return

        logger.info(f"\n⏰ 运行时间: {result['run_time']}")
        logger.info(f"📈 ETF总数: {len(signals)}")

        # 按信号分排序显示
        buy_signals = [s for s in signals if s['score'] >= 60]
        if buy_signals:
            logger.info(f"\n🎯 有效买入信号 ({len(buy_signals)}只):")
            for s in buy_signals:
                emoji = {
                    'STRONG_BUY': '🔴', 'BUY': '🟢', 'LIGHT_BUY': '🟡',
                }.get(s['level'], '⚪')

                logger.info(
                    f"  {emoji} {s['etf_name']:12s} | "
                    f"分:{s['score']:5.1f} | "
                    f"{s['action']} | "
                    f"仓位:{s['position_pct']}% | "
                    f"{s['algorithm']}"
                )

        # 强信号
        strong = [s for s in signals if s['score'] >= 75]
        if strong:
            logger.info(f"\n⭐ 重点推荐 ({len(strong)}只强信号):")
            for s in strong[:3]:
                logger.info(f"  ★ {s['etf_name']} ({s['etf_code'].upper()})")
                logger.info(f"    → 信号分: {s['score']:.1f} | 操作: {s['action']} | 仓位: {s['position_pct']}%")
                if s.get('reasons'):
                    logger.info(f"    → 理由: {'; '.join(s['reasons'][:2])}")

        # 报告路径
        if result.get('report_path'):
            logger.info(f"\n📄 HTML报告: {result['report_path']}")

        logger.info(f"\n{'='*70}")
        logger.info("✅ 每日分析完成！")
        logger.info(f"{'='*70}\n")


def run_backtest(threshold: int = 60):
    """运行回测"""
    logger.info("\n" + "="*70)
    logger.info("🔍 运行T+3胜率回测")
    logger.info("="*70 + "\n")

    engine = BacktestEngine()
    result = engine.run_backtest(
        signal_threshold=threshold,
        start_date='2024-01-01',
    )

    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'backtest')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'backtest_{datetime.now().strftime("%Y%m%d")}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"\n回测结果已保存: {output_file}")
    return result


def wait_for_target_time(target_time_str: str = '14:45'):
    """
    等待到目标时间

    Args:
        target_time_str: 目标时间字符串（HH:MM格式）
    """
    now = datetime.now()
    target_hour, target_minute = map(int, target_time_str.split(':'))

    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

    # 如果今天的时间已经过了，则等待明天
    if now >= target:
        target += timedelta(days=1)
        logger.info(f"今天的{target_time_str}已过，将等待明天同一时间...")

    wait_seconds = (target - now).total_seconds()

    if wait_seconds > 0:
        hours = int(wait_seconds // 3600)
        minutes = int((wait_seconds % 3600) // 60)

        logger.info(f"⏰ 定时模式已启动")
        logger.info(f"   目标时间: {target.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   等待时长: 约 {hours}小时{minutes}分钟")
        logger.info("   (按 Ctrl+C 取消)\n")

        import time
        time.sleep(wait_seconds)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ETF多策略信号系统 v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python main.py                    # 立即执行一次信号生成
  python main.py --schedule         # 定时模式（等待到14:45执行）
  python main.py --time 14:30       # 指定时间执行
  python main.py --backtest          # 运行T+3胜率回测
  python main.py --backtest --threshold 65  # 指定回测信号阈值

系统特点：
  - 22只ETF覆盖不同行业/概念/大盘指数
  - 20种独特算法，不同标的对应不同算法
  - 每日14:45基于实时价格发出信号
  - 追求T+3胜率（3日内最高价收益>0.5%）
  - 信号分0-100，越高越推荐买入
        """
    )

    parser.add_argument(
        '--schedule', '-s',
        action='store_true',
        help='启用定时模式（等待到14:45执行）'
    )

    parser.add_argument(
        '--time', '-t',
        type=str,
        default='14:45',
        help='指定运行时间（格式 HH:MM），默认 14:45'
    )

    parser.add_argument(
        '--backtest', '-b',
        action='store_true',
        help='运行T+3胜率回测'
    )

    parser.add_argument(
        '--threshold',
        type=int,
        default=60,
        help='回测信号阈值（默认60）'
    )

    parser.add_argument(
        '--no-realtime',
        action='store_true',
        help='不使用实时价格，用收盘价代替'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    try:
        # 回测模式
        if args.backtest:
            run_backtest(threshold=args.threshold)
            return

        # 定时模式
        if args.schedule:
            wait_for_target_time(args.time)

        # 正常执行
        logger.info("\n🚀 启动ETF多策略信号系统...\n")
        system = ETFSignalSystem()
        result = system.run_daily_analysis(use_realtime=not args.no_realtime)

        # 返回退出码
        if result.get('errors'):
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("\n\n⛔ 用户中断执行")
        sys.exit(130)

    except Exception as e:
        logger.critical(f"\n💥 系统错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
