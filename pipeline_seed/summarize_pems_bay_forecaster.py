"""Summarize the audited FORECAST VALIDATION; do not imply held-out alert gains."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import numpy as np
from freeze_external_validation import sha256
from train_pems_bay_forecaster import read_json, write_json

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/pems-bay-forecaster')
    args = parser.parse_args()
    output = args.output.resolve()
    result, audit, config, fingerprints = (read_json(output/name) for name in
        ('results.json', 'audit.json', 'config.json', 'input_fingerprints.json'))
    if audit['audit_status'] != 'passed' or audit['result_sha256'] != sha256(output/'results.json'):
        raise ValueError('Require a passed audit of the current training result.')
    if audit['audit_script_sha256'] != sha256(ROOT/'audit_pems_bay_forecaster.py'):
        raise ValueError('Audit source changed.')
    for name, digest in config['source_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Training source changed: {name}')
    for name, digest in result['artifact_sha256'].items():
        if sha256(output/name) != digest:
            raise ValueError(f'Training artifact changed: {name}')
    logs, metrics = result['model_selection'], result['validation']
    ensemble, persistence = metrics['three_seed_ensemble'], metrics['last_visible_persistence']
    relative = (persistence['mae_mph']-ensemble['mae_mph'])/persistence['mae_mph']
    lines = ['# PEMS-BAY三种子缺测图TCN：预测验证结果', '',
             '2026-10-11。本阶段完成预测器训练、验证选权重与审核。数据范围仅预测训练/预测验证；83日评测尾段尚未解封。以下为参与模型选择的验证描述，不能解释成未见测试泛化或交通告警改善。', '',
             '## 固定预算与实际执行', '',
             f'训练2017-01-01至02-11，共{fingerprints["train_windows"]:,}个连续窗口；预测验证02-12至02-18，共{fingerprints["validation_windows"]:,}个窗口。固定24站、过去12步预测未来3步、5分钟采样。均值/标准差与正相关图只用前42天，旧绑定与原验证方案SHA保持。', '',
             f'每种子25轮、batch128、每轮{fingerprints["optimizer_updates_per_epoch"]}次更新，共{25*fingerprints["optimizer_updates_per_epoch"]:,}次。Adam0.001、mask有效目标的归一化L1、梯度截断5及1线程确定性CPU设置；模型验证mask种子991固定。实际优化更新和25次验证调用均记录并审核，不根据中途表现加轮次或只选一个种子。', '',
             '归一化训练L1与下表的mph MAE不同：验证将每站误差乘回训练标准差，再按自然有效目标汇总。人工缺测只影响输入，不删未来真值分母。', '',
             '| 方法 | 选定轮次 | 验证MAE（mph） | 5/10/15分钟MAE（mph） |',
             '| --- | ---: | ---: | --- |']
    for log, metric in zip(logs, metrics['by_seed']):
        horizon = '/'.join(f'{value:.4f}' for value in metric['mae_mph_by_horizon'])
        lines.append(f'| TCN种子{log["seed"]} | {log["selected_epoch"]} | {metric["mae_mph"]:.4f} | {horizon} |')
    for label, metric in [('三种子平均预测', ensemble), ('最近可见速度持久性参考', persistence)]:
        horizon = '/'.join(f'{value:.4f}' for value in metric['mae_mph_by_horizon'])
        lines.append(f'| {label} | — | {metric["mae_mph"]:.4f} | {horizon} |')
    lines += ['', f'三种子平均相对持久性参考的验证MAE变化为{relative*100:.2f}%降低；如果该数为负则表示增加。有效目标{ensemble["valid_targets"]:,}个，所有方法相同。', '',
              '## 可以怎样解释', '',
              '这个比较说明当前缺测图TCN在固定人工mask的预测验证期具有可用的速度估计能力，后续可冻结三模型生成风险层特征。验证期用于选训练轮次，因此不能称独立确认，也没有按告警目标验证。', '',
              '预测MAE较小不保证新事件通知更早、误报更少或检测更多；先前METR-LA实验已展示这种偏差。当前未计算PEMS-BAY的风险训练标签、告警预算、阈值或主指标。不能只凭预测器结果宣称论文方法创新。', '',
              '结构采用已有图邻域聚合、门控因果时间卷积、节点嵌入和持久性残差，并非官方DCRNN或STGCN复现。三种子全部保留，平均预测不代表零样本跨网络能力。', '',
              '## 通过的审核', '',
              '- 训练前保存源码/协议/数据/绑定SHA，完成后所有来源及权重/日志/验证缓存SHA一致。',
              '- 真实速度索引拦截仅允许预测训练和预测验证两分区；读取器继续拒绝评测。审核另用两笔同分区切片独立核对输入/目标边界，不读取风险、校准或评测速度。',
              '- 训练/验证窗口与固定mask指纹重建，独立逐窗口时间检查、首/中/末窗口原始读数及自然有效mask核对。',
              '- 三模型实际25轮更新、每轮验证日志、最小验证MAE选轮次及平局较早规则、保存权重和图缓冲区一致。',
              '- 保存权重重建全部验证预测，数值差0；三种子平均、持久性参考、整体和各步MAE及另一路误差求和核对。',
              '- 将输入mask后的数值改为10000，三个选定模型输出差均0。', '',
              '审核范围是预测训练/选权重，不含风险层或告警性能，也没有重训bootstrap区间。', '',
              '## 下一步和PyTorch理解', '',
              '下一步冻结这三份权重，读取既定风险训练/验证分区，按51维可见特征建立持续低速与起点监督，拟合强线性/校准对照和三种子风险MLP。风险训练需排除03-12时间间隔对应的跨越窗口；完整校准选择保存并审核后才有资格解封评测。', '',
              '这一步的PyTorch流程是：TensorDataset中的历史/自然mask/未来目标/有效mask/日历→DataLoader按种子打乱→人工mask→模型前向→有效目标L1→backward→截断梯度→Adam.step。每轮model.eval与no_grad验证，保存最小验证MAE的state_dict；后续风险层使用冻结预测，不继续反向更新预测器。', '',
              f'实际运行墙钟约{result["elapsed_seconds"]:.1f}秒，受CPU环境影响，不推断完整325站网络耗时。结果/每轮日志/预测缓存/审核及图保存在本目录；未发布GitHub。', '']
    report = output/'report.zh-CN.md'
    text = '\n'.join(lines)
    if report.exists() and report.read_text(encoding='utf-8') != text:
        raise ValueError('Existing report differs; preserve it and version changes explicitly.')
    if not report.exists():
        report.write_text(text, encoding='utf-8')
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for log in logs:
        curve = [row['model_mae'] for row in log['epochs']]
        line, = axes[0].plot(range(1, 26), curve, label=f'Seed {log["seed"]}', linewidth=1.6)
        axes[0].scatter([log['selected_epoch']], [log['validation_mae']], color=line.get_color(), s=30, zorder=3)
    axes[0].axhline(persistence['mae_mph'], color='gray', linestyle='--', label='Persistence')
    axes[0].set(xlabel='Training epoch', ylabel='Masked validation MAE (mph)', title='Fixed 25-epoch selection')
    axes[0].legend(fontsize=8, loc='upper right')
    x = np.arange(3)
    axes[1].bar(x-.18, persistence['mae_mph_by_horizon'], .36, label='Persistence', color='#9099a3')
    axes[1].bar(x+.18, ensemble['mae_mph_by_horizon'], .36, label='Three-seed TCN', color='#387bbd')
    axes[1].set(xticks=x, xticklabels=['5 min', '10 min', '15 min'], ylabel='MAE (mph)', title='Prediction validation only')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
    fig.suptitle('PEMS-BAY: forecast train/validation; held-out evaluation remains sealed', fontsize=11)
    fig.tight_layout()
    # Reuse already-created figures so repeated verification never changes a frozen image.
    for suffix in ('png', 'svg'):
        path = output/f'summary.{suffix}'
        if not path.exists():
            fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    sources = [output/'config.json', output/'results.json', output/'audit.json', report,
               output/'summary.png', output/'summary.svg', Path(__file__).resolve(),
               ROOT/'audit_pems_bay_forecaster.py']
    freeze = {'stage': 'forecaster_weights_selected_on_prediction_validation',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'evaluation_values_read': False, 'held_out_evaluation_executed': False,
              'risk_training_executed': False, 'calibration_executed': False,
              'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}}
    path = output/'forecast_implementation_freeze.json'
    if path.exists():
        if read_json(path)['source_sha256'] != freeze['source_sha256']:
            raise ValueError('Frozen forecast source/artifact changed.')
    else:
        write_json(path, freeze)
    print(json.dumps({'forecast_audit': 'passed', 'ensemble_validation_mae_mph': ensemble['mae_mph'],
                      'persistence_validation_mae_mph': persistence['mae_mph'],
                      'relative_validation_mae_reduction': relative, 'evaluation_values_read': False,
                      'report': str(report)}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
