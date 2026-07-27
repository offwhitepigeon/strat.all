# -*- coding: utf-8 -*-
"""
报告生成器 - HTML信号报告
=========================

生成每日交易信号的可视化HTML报告：
1. 信号汇总表（按信号分排序）
2. 各算法表现统计
3. 重点关注标的（强信号）
4. 技术指标详情
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ReportGenerator:
    """HTML报告生成器"""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'reports')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _filter_backtest_summary(self, backtest_summary: Dict) -> Dict:
        """
        过滤回测摘要：只保留当前ETF_POOL中的算法和ETF
        
        防止已删除的ETF（如军工、红利低波等）仍显示在报告底部
        """
        try:
            from etf_config import ETF_POOL
            
            current_algos = set(etf.algorithm for etf in ETF_POOL)
            current_etf_names = set(etf.name for etf in ETF_POOL)
            
            by_algorithm = backtest_summary.get('by_algorithm', {})
            filtered_by_algo = {}
            
            total_signals = 0
            total_wins = 0
            weighted_return = 0.0
            weighted_max_return = 0.0
            
            for algo, stats in by_algorithm.items():
                # 过滤掉不再使用的算法
                if algo not in current_algos:
                    continue
                # 过滤etfs列表，只保留当前ETF
                filtered_etfs = [e for e in stats.get('etfs', []) if e in current_etf_names]
                if not filtered_etfs:
                    continue
                
                filtered_stats = dict(stats)
                filtered_stats['etfs'] = filtered_etfs
                filtered_by_algo[algo] = filtered_stats
                
                # 累加用于重新计算总数
                sig_count = stats.get('total_signals', 0)
                total_signals += sig_count
                total_wins += int(sig_count * stats.get('win_rate', 0) / 100)
                weighted_return += sig_count * stats.get('avg_return_3d', 0)
                weighted_max_return += sig_count * stats.get('avg_max_return', 0)
            
            result = dict(backtest_summary)
            result['by_algorithm'] = filtered_by_algo
            
            # 重新计算汇总数据
            if total_signals > 0:
                result['total_signals'] = total_signals
                result['total_win_rate'] = round(total_wins / total_signals * 100, 1)
                result['avg_return_3d'] = round(weighted_return / total_signals, 2)
                result['avg_max_return'] = round(weighted_max_return / total_signals, 2)
            
            filtered_count = len(by_algorithm) - len(filtered_by_algo)
            if filtered_count > 0:
                logger.info(f"  回测数据过滤: 移除{filtered_count}个已弃用算法")
            
            return result
        except Exception as e:
            logger.warning(f"过滤回测数据失败: {e}")
            return backtest_summary

    def generate_signal_report(
        self,
        signals: List[Dict],
        backtest_summary: Optional[Dict] = None,
        timestamp: datetime = None
    ) -> str:
        """
        生成每日信号HTML报告
        
        文件名格式: signal_report_YYYYMMDD_HHMMSS.html
        每次生成都保存为新文件，不覆盖已有报告
        """
        if timestamp is None:
            timestamp = datetime.now()

        report_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        date_str = timestamp.strftime('%Y%m%d')
        time_str = timestamp.strftime('%H%M%S')

        # 过滤回测数据：移除已删除的ETF和算法
        if backtest_summary:
            backtest_summary = self._filter_backtest_summary(backtest_summary)

        total = len(signals)
        buy_signals = [s for s in signals if s['score'] >= 60]
        strong_signals = [s for s in signals if s['score'] >= 75]
        strong_buy_signals = [s for s in signals if s['score'] >= 85]

        html = self._build_html(
            signals, report_time, total,
            buy_signals, strong_signals, strong_buy_signals,
            backtest_summary
        )

        filename = f'signal_report_{date_str}_{time_str}.html'
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML报告已生成: {filepath}")
        return filepath

    def _build_html(
        self,
        signals: List[Dict],
        report_time: str,
        total: int,
        buy_signals: list,
        strong_signals: list,
        strong_buy_signals: list,
        backtest_summary: Optional[Dict]
    ) -> str:
        """构建HTML内容"""

        rows_html = ""
        for s in signals:
            score = s['score']
            level = s['level']
            action = s['action']

            if score >= 85:
                row_class = 'strong-buy'
                score_color = '#e74c3c'
            elif score >= 75:
                row_class = 'buy'
                score_color = '#27ae60'
            elif score >= 60:
                row_class = 'light-buy'
                score_color = '#f39c12'
            elif score >= 40:
                row_class = 'watch'
                score_color = '#7f8c8d'
            else:
                row_class = 'wait'
                score_color = '#bdc3c7'

            reasons = s.get('reasons', [])[:3]
            reasons_html = '<br>'.join(f'• {r}' for r in reasons) if reasons else '—'

            ind = s.get('indicators', {})
            ind_html = ""
            if ind:
                key_inds = []
                if 'rsi_14' in ind:
                    key_inds.append(f"RSI:{ind['rsi_14']:.0f}")
                if 'zscore_20' in ind:
                    key_inds.append(f"Z:{ind['zscore_20']:.1f}")
                if 'bb_percent_b' in ind:
                    key_inds.append(f"BB:{ind['bb_percent_b']:.2f}")
                if 'dev_ma20' in ind:
                    key_inds.append(f"偏离MA20:{ind['dev_ma20']:+.1f}%")
                if 'momentum_20' in ind:
                    key_inds.append(f"动量:{ind['momentum_20']:+.1f}%")
                if 'consec_down' in ind and ind['consec_down'] > 0:
                    key_inds.append(f"连跌:{ind['consec_down']}日")
                if 'vol_ratio' in ind:
                    key_inds.append(f"量比:{ind['vol_ratio']:.1f}")
                ind_html = ' | '.join(key_inds)

            rows_html += f"""
            <tr class="{row_class}">
                <td class="score-cell" style="color:{score_color};font-weight:bold;font-size:16px;">{score:.1f}</td>
                <td><span class="level-badge level-{level.lower()}">{level}</span></td>
                <td class="name-cell">
                    <strong>{s['etf_name']}</strong>
                    <div class="etf-code">{s['etf_code'].upper()} | {s['sector']}</div>
                </td>
                <td><span class="algo-tag">{s['algorithm']}</span></td>
                <td class="price-cell">{s['current_price']:.3f}</td>
                <td class="action-cell"><strong>{action}</strong></td>
                <td class="position-cell">{s['position_pct']}%</td>
                <td class="indicators-cell">{ind_html}</td>
                <td class="reasons-cell">{reasons_html}</td>
            </tr>"""

        backtest_html = ""
        if backtest_summary:
            bt = backtest_summary
            backtest_html = f"""
            <div class="backtest-section">
                <h2>📊 历史回测统计</h2>
                <div class="bt-summary">
                    <div class="bt-card">
                        <div class="bt-value">{bt.get('total_signals', 0)}</div>
                        <div class="bt-label">历史信号总数</div>
                    </div>
                    <div class="bt-card">
                        <div class="bt-value bt-win">{bt.get('total_win_rate', 0)}%</div>
                        <div class="bt-label">T+3总胜率</div>
                    </div>
                    <div class="bt-card">
                        <div class="bt-value">{bt.get('avg_return_3d', 0):+.2f}%</div>
                        <div class="bt-label">平均T+3收益</div>
                    </div>
                    <div class="bt-card">
                        <div class="bt-value">{bt.get('avg_max_return', 0):+.2f}%</div>
                        <div class="bt-label">平均T+3最大收益</div>
                    </div>
                </div>
                <table class="bt-table">
                    <tr><th>算法</th><th>信号数</th><th>胜率</th><th>平均收益</th><th>平均最大收益</th><th>适用ETF</th></tr>
                    {self._build_backtest_rows(bt.get('by_algorithm', {}))}
                </table>
            </div>"""

        strong_html = ""
        if strong_signals:
            strong_list = ""
            for s in strong_signals[:5]:
                strong_list += f"""
                <div class="strong-signal-item">
                    <span class="strong-score">{s['score']:.1f}</span>
                    <span class="strong-name">{s['etf_name']}</span>
                    <span class="strong-action">{s['action']}</span>
                    <span class="strong-position">仓位{s['position_pct']}%</span>
                </div>"""
            strong_html = f"""
            <div class="strong-signals-section">
                <h2>🎯 今日重点关注（强信号≥75分）</h2>
                {strong_list}
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF多策略信号报告 | {report_time}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif;
            background: #f5f6fa; color: #2c3e50; padding: 20px; line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px;
        }}
        .header h1 {{ font-size: 24px; margin-bottom: 8px; }}
        .header .meta {{ opacity: 0.8; font-size: 14px; }}
        .summary-cards {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
        .card {{
            background: white; padding: 20px; border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1; min-width: 150px;
            text-align: center;
        }}
        .card .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .card .label {{ font-size: 13px; color: #7f8c8d; margin-top: 5px; }}
        .card.alert .value {{ color: #e74c3c; }}
        .card.success .value {{ color: #27ae60; }}
        .section {{
            background: white; border-radius: 12px; padding: 20px;
            margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow-x: auto;
        }}
        .section h2 {{ font-size: 18px; margin-bottom: 15px; color: #2c3e50;
                       border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #f8f9fa; padding: 10px; text-align: left;
             border-bottom: 2px solid #dee2e6; color: #495057; white-space: nowrap; }}
        td {{ padding: 10px; border-bottom: 1px solid #ecf0f1; vertical-align: top; }}
        tr.strong-buy {{ background: #fff5f5; }}
        tr.buy {{ background: #f0fff4; }}
        tr.light-buy {{ background: #fffef5; }}
        tr.watch {{ background: #f8f9fa; }}
        .score-cell {{ font-size: 16px; font-weight: bold; }}
        .name-cell .etf-code {{ font-size: 11px; color: #95a5a6; margin-top: 2px; }}
        .level-badge {{
            padding: 3px 10px; border-radius: 12px; font-size: 11px;
            font-weight: bold; color: white;
        }}
        .level-strong_buy {{ background: #e74c3c; }}
        .level-buy {{ background: #27ae60; }}
        .level-light_buy {{ background: #f39c12; }}
        .level-watch {{ background: #7f8c8d; }}
        .level-wait {{ background: #bdc3c7; }}
        .algo-tag {{
            background: #ecf0f1; padding: 2px 8px; border-radius: 4px;
            font-size: 11px; color: #2c3e50;
        }}
        .price-cell {{ font-family: 'Courier New', monospace; font-size: 14px; }}
        .action-cell {{ font-weight: bold; }}
        .position-cell {{ font-weight: bold; color: #e67e22; }}
        .indicators-cell {{ font-size: 11px; color: #7f8c8d; }}
        .reasons-cell {{ font-size: 11px; color: #555; max-width: 300px; }}
        .strong-signals-section {{ margin-bottom: 20px; }}
        .strong-signal-item {{
            background: white; padding: 12px 15px; border-radius: 8px;
            margin-bottom: 8px; display: flex; align-items: center; gap: 15px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06); border-left: 4px solid #e74c3c;
        }}
        .strong-score {{
            font-size: 20px; font-weight: bold; color: #e74c3c;
            min-width: 50px;
        }}
        .strong-name {{ font-weight: bold; min-width: 100px; }}
        .strong-action {{ color: #27ae60; font-weight: bold; }}
        .backtest-section h2 {{ margin-bottom: 15px; }}
        .bt-summary {{ display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; }}
        .bt-card {{
            background: #f8f9fa; padding: 15px; border-radius: 8px;
            text-align: center; flex: 1; min-width: 120px;
        }}
        .bt-card .bt-value {{ font-size: 22px; font-weight: bold; color: #2c3e50; }}
        .bt-card .bt-win {{ color: #27ae60; }}
        .bt-card .bt-label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
        .bt-table {{ font-size: 12px; }}
        .bt-table th {{ background: #f8f9fa; }}
        .footer {{
            text-align: center; padding: 20px; color: #95a5a6;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ETF多策略信号报告</h1>
        <div class="meta">
            生成时间: {report_time} | 策略: 20种算法 × 22只ETF | 胜率目标: T+3
            <br>数据源: akshare | 运行时间: 14:45
        </div>
    </div>

    <div class="summary-cards">
        <div class="card">
            <div class="value">{total}</div>
            <div class="label">ETF总数</div>
        </div>
        <div class="card success">
            <div class="value">{len(buy_signals)}</div>
            <div class="label">有效信号(≥60分)</div>
        </div>
        <div class="card alert">
            <div class="value">{len(strong_signals)}</div>
            <div class="label">强信号(≥75分)</div>
        </div>
        <div class="card alert">
            <div class="value">{len(strong_buy_signals)}</div>
            <div class="label">极强信号(≥85分)</div>
        </div>
    </div>

    {strong_html}

    <div class="section">
        <h2>📋 全部信号（按信号分排序）</h2>
        <table>
            <tr>
                <th>信号分</th>
                <th>等级</th>
                <th>ETF名称</th>
                <th>算法</th>
                <th>当前价</th>
                <th>操作</th>
                <th>建议仓位</th>
                <th>关键指标</th>
                <th>信号理由</th>
            </tr>
            {rows_html}
        </table>
    </div>

    {backtest_html}

    <div class="footer">
        <p>ETF多策略信号系统 v2.0 | 基于akshare数据 | 20种独特算法 × 22只ETF</p>
        <p>信号分0-100，越高越推荐买入 | 胜率定义: T+3（3个交易日）最高价收益>0.5%</p>
        <p>⚠️ 本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
    </div>
</body>
</html>"""
        return html

    def _build_backtest_rows(self, by_algorithm: Dict) -> str:
        """构建回测统计表格行"""
        rows = ""
        for algo, stats in sorted(by_algorithm.items(), key=lambda x: -x[1]['win_rate']):
            etfs = ', '.join(stats.get('etfs', []))
            rows += f"""
            <tr>
                <td><span class="algo-tag">{algo}</span></td>
                <td>{stats['total_signals']}</td>
                <td style="color:{'#27ae60' if stats['win_rate'] >= 75 else '#e74c3c'};font-weight:bold;">{stats['win_rate']}%</td>
                <td>{stats['avg_return_3d']:+.2f}%</td>
                <td>{stats['avg_max_return']:+.2f}%</td>
                <td>{etfs}</td>
            </tr>"""
        return rows
