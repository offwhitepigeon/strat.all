# -*- coding: utf-8 -*-
"""
每日定时运行脚本 - ETF信号系统 + 邮件推送
==========================================

功能：
1. 运行ETF多策略信号系统（28只ETF × 14种算法）
2. 生成HTML信号报告
3. 将信号摘要 + HTML报告附件发送至邮箱

使用方法：
    python daily_run.py                 # 立即运行一次
    python daily_run.py --schedule      # 等待到14:45自动运行
    python daily_run.py --no-realtime   # 不使用实时价格（用收盘价）
    python daily_run.py --no-mail       # 只运行不发邮件

配置文件: mail_config.json
"""

import os
import sys
import json
import time
import smtplib
import logging
import argparse
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# 确保UTF-8输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ===== 路径设置 =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'mail_config.json')

# ===== 日志配置 =====
log_dir = os.path.join(SCRIPT_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f'daily_run_{datetime.now().strftime("%Y%m%d")}.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """加载邮箱配置"""
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"配置文件不存在: {CONFIG_PATH}")
        return None

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if '在此填入' in config.get('smtp_password', ''):
        logger.warning("SMTP授权码未配置！请编辑 mail_config.json 填入授权码")
        return None

    return config


def run_analysis(use_realtime: bool = True) -> dict:
    """
    运行ETF信号分析

    Returns:
        分析结果字典（含signals、report_path等）
    """
    # 切换到项目目录
    os.chdir(SCRIPT_DIR)

    # 导入项目模块
    from signal_engine import SignalEngine
    from report_generator import ReportGenerator
    from backtest_engine import BacktestEngine

    run_time = datetime.now()
    logger.info(f"{'='*60}")
    logger.info(f"ETF信号系统启动 | {run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"22只ETF × 20种算法 | 胜率目标: T+3")
    logger.info(f"{'='*60}")

    # 1. 生成信号
    logger.info("[1/2] 获取数据并生成信号...")
    engine = SignalEngine()
    signals = engine.run_daily(use_realtime=use_realtime)

    # 统计
    buy_signals = [s for s in signals if s['score'] >= 60]
    strong_signals = [s for s in signals if s['score'] >= 75]
    logger.info(f"  信号生成完成: {len(signals)}只ETF, "
                f"有效信号{len(buy_signals)}只, 强信号{len(strong_signals)}只")

    # 2. 生成HTML报告
    logger.info("[2/2] 生成HTML报告...")
    report_gen = ReportGenerator()

    # 加载回测统计
    backtest_summary = None
    backtest_dir = os.path.join(SCRIPT_DIR, 'data', 'backtest')
    if os.path.exists(backtest_dir):
        files = sorted([f for f in os.listdir(backtest_dir)
                        if f.startswith('backtest_') and f.endswith('.json')],
                      reverse=True)
        if files:
            with open(os.path.join(backtest_dir, files[0]), 'r', encoding='utf-8') as f:
                bt_data = json.load(f)
                backtest_summary = bt_data.get('summary', {})

    report_path = report_gen.generate_signal_report(
        signals=signals,
        backtest_summary=backtest_summary,
        timestamp=run_time
    )
    logger.info(f"  报告已生成: {report_path}")

    # 3. 保存历史数据
    history_dir = os.path.join(SCRIPT_DIR, 'data', 'history')
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f'signal_{run_time.strftime("%Y%m%d_%H%M%S")}.json')

    result = {
        'run_time': run_time.isoformat(),
        'date': run_time.strftime('%Y-%m-%d'),
        'signals': signals,
        'report_path': report_path,
        'history_path': history_path,
        'total_etfs': len(signals),
        'buy_count': len(buy_signals),
        'strong_count': len(strong_signals),
    }

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"  历史数据已保存: {history_path}")

    return result


def build_email_html(result: dict) -> str:
    """
    构建邮件正文HTML（信号摘要）

    Args:
        result: 分析结果

    Returns:
        HTML格式的邮件正文
    """
    signals = result.get('signals', [])
    run_time = result.get('run_time', '')

    # 按信号分排序
    sorted_signals = sorted(signals, key=lambda x: x['score'], reverse=True)

    # 统计
    total = len(sorted_signals)
    buy_signals = [s for s in sorted_signals if s['score'] >= 60]
    strong_signals = [s for s in sorted_signals if s['score'] >= 75]

    # 颜色映射
    level_colors = {
        'STRONG_BUY': '#e74c3c',
        'BUY': '#27ae60',
        'LIGHT_BUY': '#f39c12',
        'WATCH': '#95a5a6',
        'WAIT': '#7f8c8d',
    }

    # 构建信号表格行
    rows_html = ''
    for s in sorted_signals:
        color = level_colors.get(s['level'], '#7f8c8d')
        reasons = '; '.join(s.get('reasons', [])[:2]) if s.get('reasons') else '-'

        # 关键指标
        indicators = s.get('indicators', {})
        ind_str = ''
        if 'rsi_14' in indicators:
            ind_str += f"RSI:{indicators['rsi_14']:.0f} "
        if 'discount_rate' in indicators:
            ind_str += f"折价率:{indicators['discount_rate']:.1f}% "

        rows_html += f'''
        <tr style="color: {'#fff' if s['score'] >= 75 else '#333'}; background: {color if s['score'] >= 60 else 'transparent'};">
            <td style="padding: 6px; text-align: center;"><b>{s['score']:.0f}</b></td>
            <td style="padding: 6px;">{s['level']}</td>
            <td style="padding: 6px;">{s['etf_name']}</td>
            <td style="padding: 6px;">{s['algorithm']}</td>
            <td style="padding: 6px; text-align: right;">{s['current_price']:.3f}</td>
            <td style="padding: 6px;">{s['action']}</td>
            <td style="padding: 6px; text-align: center;">{s['position_pct']}%</td>
            <td style="padding: 6px; font-size: 12px;">{ind_str}</td>
        </tr>'''

    # 构建强信号卡片
    strong_html = ''
    if strong_signals:
        cards = ''
        for s in strong_signals[:5]:
            color = level_colors.get(s['level'], '#7f8c8d')
            reasons = '; '.join(s.get('reasons', [])[:3]) if s.get('reasons') else ''
            cards += f'''
            <div style="display: inline-block; width: 45%; margin: 5px; padding: 12px;
                        background: {color}; color: white; border-radius: 8px;">
                <div style="font-size: 16px; font-weight: bold;">{s['etf_name']} ({s['etf_code']})</div>
                <div style="font-size: 14px; margin: 4px 0;">信号分: {s['score']:.0f} | {s['action']} | 仓位: {s['position_pct']}%</div>
                <div style="font-size: 12px;">{reasons}</div>
            </div>'''
        strong_html = f'<div style="margin: 15px 0;"><h3>⭐ 重点推荐 ({len(strong_signals)}只)</h3>{cards}</div>'

    # 完整HTML
    html = f'''<html><body style="font-family: 'Microsoft YaHei', Arial, sans-serif; background: #f5f5f5; padding: 20px;">

    <div style="max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px;">
        <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
            ETF多策略信号报告
        </h2>
        <p style="color: #7f8c8d; font-size: 13px;">
            生成时间: {run_time[:19].replace('T', ' ')} | 22只ETF × 20种算法 | 胜率目标: T+3
        </p>

        <div style="display: flex; margin: 15px 0;">
            <div style="flex:1; text-align: center; padding: 10px; background: #ecf0f1; border-radius: 6px; margin: 0 5px;">
                <div style="font-size: 24px; font-weight: bold; color: #2c3e50;">{total}</div>
                <div style="font-size: 12px; color: #7f8c8d;">ETF总数</div>
            </div>
            <div style="flex:1; text-align: center; padding: 10px; background: #e8f6f3; border-radius: 6px; margin: 0 5px;">
                <div style="font-size: 24px; font-weight: bold; color: #27ae60;">{len(buy_signals)}</div>
                <div style="font-size: 12px; color: #7f8c8d;">有效信号(≥60分)</div>
            </div>
            <div style="flex:1; text-align: center; padding: 10px; background: #fdedec; border-radius: 6px; margin: 0 5px;">
                <div style="font-size: 24px; font-weight: bold; color: #e74c3c;">{len(strong_signals)}</div>
                <div style="font-size: 12px; color: #7f8c8d;">强信号(≥75分)</div>
            </div>
        </div>

        {strong_html}

        <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 15px;">
            <thead>
                <tr style="background: #2c3e50; color: white;">
                    <th style="padding: 8px;">信号分</th>
                    <th style="padding: 8px;">等级</th>
                    <th style="padding: 8px;">ETF名称</th>
                    <th style="padding: 8px;">算法</th>
                    <th style="padding: 8px;">当前价</th>
                    <th style="padding: 8px;">操作</th>
                    <th style="padding: 8px;">仓位</th>
                    <th style="padding: 8px;">关键指标</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <div style="margin-top: 15px; padding: 10px; background: #fef9e7; border-radius: 6px; font-size: 12px; color: #7f8c8d;">
            <p>信号分0-100: WAIT(0-40) | WATCH(40-60) | LIGHT_BUY(60-75) | BUY(75-85) | STRONG_BUY(85-100)</p>
            <p>胜率定义: T+3（3个交易日）最高价收益 &gt; 0.5%</p>
            <p>详细HTML报告见附件</p>
            <p style="color: #e74c3c;">本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
        </div>
    </div>

    </body></html>'''

    return html


def send_email(result: dict, config: dict) -> bool:
    """
    发送邮件

    Args:
        result: 分析结果
        config: 邮箱配置

    Returns:
        是否发送成功
    """
    try:
        smtp_server = config['smtp_server']
        smtp_port = config['smtp_port']
        email_from = config['email_from']
        email_to = config['email_to']
        smtp_password = config['smtp_password']

        # 邮件主题
        buy_count = result.get('buy_count', 0)
        strong_count = result.get('strong_count', 0)
        date_str = result.get('date', datetime.now().strftime('%Y-%m-%d'))

        if strong_count > 0:
            subject = f'ETF信号 {date_str} | {strong_count}只强信号 | {buy_count}只有效'
        elif buy_count > 0:
            subject = f'ETF信号 {date_str} | {buy_count}只有效信号'
        else:
            subject = f'ETF信号 {date_str} | 今日无买入信号'

        # 构建邮件
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = email_to
        msg['Subject'] = subject

        # 邮件正文（HTML摘要）
        html_body = build_email_html(result)
        msg.attach(MIMEText(html_body, 'html', 'utf-8'))

        # 添加HTML报告附件
        report_path = result.get('report_path', '')
        if report_path and os.path.exists(report_path):
            with open(report_path, 'rb') as f:
                attach = MIMEBase('application', 'octet-stream')
                attach.set_payload(f.read())
                encoders.encode_base64(attach)
                attach.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename=f'ETF信号报告_{date_str}.html'
                )
                msg.attach(attach)
            logger.info(f"  附件: ETF信号报告_{date_str}.html")

        # 发送
        logger.info(f"  正在发送邮件至 {email_to}...")

        if smtp_port == 465:
            # SSL
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
        else:
            # TLS
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.starttls()

        server.login(email_from, smtp_password)
        server.sendmail(email_from, [email_to], msg.as_string())
        server.quit()

        logger.info(f"  邮件发送成功!")
        logger.info(f"  主题: {subject}")
        return True

    except Exception as e:
        logger.error(f"  邮件发送失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def wait_for_target_time(target_time_str: str = '14:45'):
    """等待到目标时间"""
    now = datetime.now()
    target_hour, target_minute = map(int, target_time_str.split(':'))

    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

    if now >= target:
        target += timedelta(days=1)

    wait_seconds = (target - now).total_seconds()
    hours = int(wait_seconds // 3600)
    minutes = int((wait_seconds % 3600) // 60)

    logger.info(f"定时模式已启动")
    logger.info(f"  目标时间: {target.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  等待时长: 约 {hours}小时{minutes}分钟")
    logger.info(f"  (按 Ctrl+C 取消)")

    time.sleep(wait_seconds)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ETF信号系统 - 每日运行 + 邮件推送')
    parser.add_argument('--schedule', '-s', action='store_true', help='定时模式（等待到14:45执行）')
    parser.add_argument('--time', '-t', type=str, default='14:45', help='运行时间（HH:MM）')
    parser.add_argument('--no-realtime', action='store_true', help='不使用实时价格')
    parser.add_argument('--no-mail', action='store_true', help='只运行不发邮件')
    parser.add_argument('--conditional-mail', action='store_true', help='仅有有效信号(≥60分)时才发邮件')

    args = parser.parse_args()

    try:
        # 定时模式
        if args.schedule:
            wait_for_target_time(args.time)

        # 运行分析
        logger.info("启动ETF信号系统...")
        result = run_analysis(use_realtime=not args.no_realtime)

        # 发送邮件
        if not args.no_mail:
            buy_count = result.get('buy_count', 0)
            if args.conditional_mail and buy_count == 0:
                logger.info("无有效信号(≥60分)，跳过邮件发送")
            else:
                config = load_config()
                if config:
                    send_email(result, config)
                else:
                    logger.warning("邮箱配置无效，跳过邮件发送")

        # 打印摘要
        signals = result.get('signals', [])
        buy_signals = [s for s in signals if s['score'] >= 60]
        logger.info(f"\n{'='*60}")
        logger.info(f"运行完成!")
        logger.info(f"  ETF总数: {len(signals)}")
        logger.info(f"  有效信号: {len(buy_signals)}只")
        if buy_signals:
            logger.info(f"  买入信号:")
            for s in sorted(buy_signals, key=lambda x: x['score'], reverse=True):
                logger.info(f"    {s['etf_name']:12s} | 分:{s['score']:5.1f} | "
                          f"{s['action']} | 仓位:{s['position_pct']}%")
        logger.info(f"  HTML报告: {result.get('report_path', '')}")
        logger.info(f"{'='*60}")

    except KeyboardInterrupt:
        logger.info("\n用户中断执行")
        sys.exit(130)
    except Exception as e:
        logger.error(f"运行错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
