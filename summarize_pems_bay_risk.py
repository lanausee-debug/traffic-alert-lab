"""Summarize an audited risk milestone and preserve its separate SHA freeze."""
from datetime import datetime, timezone
from pathlib import Path
from train_pems_bay_risk import ROOT, OUTPUT, verify_forecaster
from train_pems_bay_forecaster import read_json, write_json
from freeze_external_validation import sha256


def main():
    verify_forecaster()
    result=read_json(OUTPUT/'results.json'); logs=read_json(OUTPUT/'model_selection.json')
    audit=read_json(OUTPUT/'audit.json')
    assert audit['passed'] and audit['results_sha256']==sha256(OUTPUT/'results.json')
    for name,digest in read_json(OUTPUT/'config.json')['source_sha256'].items(): assert sha256(ROOT/name)==digest,name
    for name,digest in result['artifact_sha256'].items(): assert sha256(OUTPUT/name)==digest,name
    lines=['# PEMS-BAY风险层训练与验证', '', '日期：2026-10-11。七组预定方法已完成；这是风险权重选择验证，不是独立测试，也没有告警阈值或事件服务结论。', '',
           '固定前阶段三份TCN权重、24站、相关图和故障组。速度值只读02-19–03-18风险训练及03-19–03-25风险验证；03-26–04-08校准及04-09–06-30共83日评测继续封存。', '',
           '| 分区 | 在线起点 | 共同有效点（不重复情景） | 三情景训练/验证行 | Y阳性 | Z阳性 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for partition,row in result['partitions'].items():
        lines.append(f'| {partition} | {row["origins"]} | {row["unique_common_six_reading_points"]} | {row["pooled_rows"]} | {row["unique_persistent_positives"]} | {row["unique_onset_positives"]} |')
    lines += ['', '三情景同一个自然监督标签是重复观测，不是三倍独立事件。在线起点不按未来自然有效性过滤；共同六读数条件仅用于监督。时间缺口不能跨接，后续告警回放须分段处理。', '']
    for panel,title in (('persistent_target_Y','持续低速Y：同目标验证'),('onset_target_Z','未来起点Z：同目标验证')):
        lines += [f'## {title}', '', '| 方法 | 验证BCE | Brier |', '| --- | ---: | ---: |']
        for method,metrics in result['validation'][panel].items():
            lines.append(f'| {method} | {metrics["bce"]:.6f} | {metrics["brier"]:.6f} |')
        lines += ['']
    lines += ['不能跨Y/Z目标比较BCE；阳性比例很低时，小损失不自动代表有效预警。验证参与选择权重，因此以上不是独立泛化评价。速度规则没有概率损失，不把负mph分数当概率。', '',
              '## 优化与审核', '', '| MLP监督 | 种子 | 选中更新步 | 实际总更新 | 验证BCE |', '| --- | ---: | ---: | ---: | ---: |']
    for method in ('persistent_mlp','onset_mlp'):
        for rec in logs[method]:
            lines.append(f'| {method} | {rec["seed"]} | {rec["selected_step"]} | {rec["actual_optimizer_updates"]} | {rec["validation_bce"]:.6f} |')
    lines += ['', '两监督同种子初始化和有放回抽样索引SHA完全一致，每模型batch4096、1350次Adam更新，225/450/900/1350按自身目标BCE选权重。所有三种子概率平均，未选择最好种子。', '',
              '| 线性模型 | 选定alpha | 最大绝对梯度 |', '| --- | ---: | ---: |']
    for name,gradient in audit['selected_linear_max_abs_gradient'].items():
        rec=logs['group_calibration'][name[-1]] if name.startswith('group') else logs[name]
        lines.append(f'| {name} | {rec["selected_alpha"]} | {gradient:.3g} |')
    for g,rec in logs['group_calibration'].items():
        if 'fallback' in rec: lines.append(f'\n分组{g}回退全局：{", ".join(rec["reasons"])}；训练{rec["training_rows"]}行、验证{rec["validation_rows"]}行。')
    lines += ['', f'复用既有缓存/权重的完成调用墙钟{result["elapsed_seconds"]:.1f}秒；这不含首次构造缓存和训练。各分区准备耗时及各模型训练耗时见分区/模型日志。', '',
              '首次分数保存触及情景内空可用性分组，旧评分函数无法拼接空批次。`complete_pems_bay_risk.py`仅给零行输入返回空float32向量，其他分数调用原函数；没有样本被加减，没有重训或更改选择。`empty_group_adapter.json`记录原因、实现SHA及11份已有权重SHA，完成和审核均核对权重不变。', '',
              '审核通过：来源/权重/结果SHA，独立Y/Z转移定义与共同池，逐窗口时间边界，训练池标准化，模型选择与线性收敛，更新计数与初始化/抽样索引，各七方法全部分数精确重建，隐藏值及未来速度扰动不变性。抽样重推预测最大误差'+f'{audit["forecast_sample_maximum_difference_mph"]:.3g} mph。', '',
              '## PyTorch在本步骤的用途', '',
              '`RiskModel(51, True)`用`nn.Linear`和`nn.ReLU`建立51→32→16→1网络，输入是每个站每个时刻的51个可见特征，输出一个logit。`BCEWithLogits`直接用logit训练，避免先sigmoid造成数值问题。Y/Z更换的是监督含义，网络和预算相同。', '',
              '`loss.backward()`计算梯度，`Adam.step()`更新权重；有放回`torch.randint`固定每次4096行。`eval()`/`torch.no_grad()`只做验证和推理。`state_dict`保存选定权重，三种子`sigmoid`概率平均用于后续阈值扫描。逻辑回归只有一层Linear，使用LBFGS和梯度审核，确保简单对照充分优化。', '',
              '## 下一步', '', '只在独立校准分区建立误报预算，七方法统一粗细网格选择阈值/确认次数；先解决缺口分段与事件计分审核，保存完整选择和执行冻结后，才实现评测解封。固定两项主比较和97.5%区间判据不变。本阶段不能支持“起点MLP整体更好”或算法原创宣称。']
    path=OUTPUT/'report.zh-CN.md'; content='\n'.join(lines)+'\n'
    if path.exists(): assert path.read_text(encoding='utf-8')==content
    else: path.write_text(content,encoding='utf-8')
    paths=[OUTPUT/'config.json',OUTPUT/'results.json',OUTPUT/'audit.json',path,Path(__file__).resolve(),
           ROOT/'complete_pems_bay_risk.py',OUTPUT/'empty_group_adapter.json']
    freeze={'stage':'risk_weights_selected_on_risk_validation', 'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'risk_training_executed':True,'calibration_executed':False,'evaluation_values_read':False,
            'final_evaluation_unlock_authorized':False,
            'source_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in paths}}
    freeze_path=OUTPUT/'risk_implementation_freeze.json'
    if freeze_path.exists():
        prior=read_json(freeze_path); freeze['frozen_at_utc']=prior['frozen_at_utc']; assert prior==freeze
    else: write_json(freeze_path,freeze)
    print({'risk_stage_freeze':'passed','validation':result['validation'],'evaluation_values_read':False})


if __name__=='__main__': main()
