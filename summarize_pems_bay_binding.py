"""Report the audited acquisition/reader milestone, without running models."""
from datetime import datetime, timezone
import json
from pathlib import Path
from freeze_external_validation import verify, sha256

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/pems-bay-binding'


def read(name):
    return json.loads((OUTPUT/name).read_text(encoding='utf-8'))


def main():
    verify()
    binding, metadata, quality, audit = (read(name) for name in
        ('binding.json', 'metadata.json', 'training_quality.json', 'reader_audit.json'))
    for name, digest in binding['source_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Bound source changed: {name}')
    for name, digest in binding['artifact_sha256'].items():
        if sha256(OUTPUT/name) != digest:
            raise ValueError(f'Bound artifact changed: {name}')
    if audit['audit_status'] != 'passed' or audit['evaluation_values_read'] or audit['validation_executed']:
        raise ValueError('This report requires a passed reader-only audit with evaluation still sealed.')
    if audit['audit_script_sha256'] != sha256(ROOT/'audit_pems_bay_reader.py'):
        raise ValueError('Audit implementation changed since passing.')
    if audit['windows_script_sha256'] != sha256(ROOT/'pems_bay_windows.py'):
        raise ValueError('Window implementation changed since passing.')
    lines = ['# PEMS-BAY数据绑定与分区读取审核', '', '2026-10-11。完成数据获取、元数据绑定、训练段质量门槛和读取/窗口实现审核。尚未训练新模型、校准告警或评测，不能作为论文收益结果。', '',
             '## 来源与字节身份', '',
             '文件从[DCRNN作者公开目录](https://github.com/liyaguang/DCRNN#data-preparation)中对应Google Drive文件下载。大小135,930,936字节；SHA-256与固定HuggingFace镜像版本的LFS对象相同。镜像校验值不是作者单独公布的官方校验值。下载只处理文件字节，未把速度数组载入内存。', '',
             f'内容SHA-256：`{binding["dataset_sha256"]}`。来源记录`data/pems-bay-provenance.json`及`data/source-metadata/pems-bay-discovery.json`。', '',
             '速度单位采用[GRIN作者ICLR 2022报告第12页](https://iclr.cc/media/iclr-2022/Slides/5891.pdf)对PEMS-BAY的mph说明；不是从评测速度值推断。镜像许可标签不替代原始数据分发条件核验，本项目未授权公开再打包。', '',
             '## 只看元数据得到的时间边界', '',
             f'文件矩阵{metadata["shape"][0]:,}×{metadata["shape"][1]}，首末时间2017-01-01 00:00至2017-06-30 23:55，涉及181个原生时间轴日历日。没有擅自转换时区。时间轴一处2017-03-12 01:55→03:00的65分钟间隔；可能与夏令时相符，但文件未提供时区证据，不能把原因当已核实。', '',
             '| 分区 | 元数据日期范围（含末日） | 行数 | 用途/访问状态 |',
             '| --- | --- | ---: | --- |']
    names = {'forecast_train': ('预测器训练', '已读取24站训练速度'),
             'forecast_validation': ('预测器验证', '未读取速度'),
             'risk_train': ('风险训练', '未读取速度'),
             'risk_validation': ('风险验证', '未读取速度'),
             'calibration': ('告警校准', '未读取速度'),
             'evaluation': ('一次评测', '读取器始终拒绝速度访问')}
    for name, row in metadata['partitions'].items():
        label, status = names[name]
        lines.append(f'| {label} | {row["first_timestamp"][:10]}—{row["last_timestamp"][:10]} | {row["rows"]:,} | {status} |')
    lines += ['', '评测尾段为83个日历日；这来自时间戳，并没有查看它的自然缺测率、事件数量、拥堵频率或告警效果。', '',
              '## 固定站点与训练统计', '',
              '按冻结盐值与站ID的SHA-256选择24站，再按ID排列，未根据交通值筛站。前42天各站自然有效率均为100%，超过95%门槛；没有替换站点。有效率仅指有限且>0的文件读数，不证明现场设备无故障，也不证明数据未被提供者处理。', '',
              '选中ID：`'+', '.join(metadata['selected_ids'])+'`。', '',
              '保存仅由2017-01-01至02-11选中站读数计算的均值、标准差、正相关图与12个三站相关故障组。相关组不代表已核实的连续道路故障；这里只验证24站方案，不能代表325站全网。', '',
              '## 实现与审核范围', '',
              '- `pems_bay_reader.py`：先读ID/时间戳及矩阵头，按明确半开区间切片，只开放评测之前的命名分区；评测访问当前无解封接口。不是操作系统或加密级保护。',
              '- `pems_bay_windows.py`：在单分区内构造连续5分钟的12步历史/3步预测窗口；风险标签确认可用同一时间检查扩至5步。预测器适配器只接受预测训练/验证两个分区。',
              '- `audit_pems_bay_reader.py`：拦截真实HDF5速度索引，核对元数据零速度读取、训练切片仅[0,12096)×24站、阻止评测/跨界请求、独立哈希选站及训练统计精确重建。',
              '- 合成HDF5测试覆盖乱序列恢复、评测尾段数值投毒不影响训练统计/图/组、训练质量失败不替换站、错误列/重复时间拒绝；独立逐窗口循环核对时间间隔与边界，未来数值扰动不改变当前输入。合成数据仅用于实现审核。', '',
              '全部检查通过。这个“通过”只涵盖来源绑定、读取器、训练统计与窗口适配器；没有审核或执行新模型训练、风险层、告警校准、评测解封与主比较。', '',
              '## 后续可直接使用的入口', '',
              '运行`.\\.venv\\Scripts\\python.exe bind_pems_bay.py`会核对已有绑定，不覆盖。读取审核用`audit_pems_bay_reader.py`；报告与实现哈希用`summarize_pems_bay_binding.py`。原10月10日方案及其15文件冻结仍通过，当前另保存`reader_implementation_freeze.json`，不替代未来模型/校准执行冻结。', '',
              '下一步通过已审核适配器训练42/43/44三种子缺测图TCN，只在02-12至02-18验证期选轮次，保存每轮日志、权重、访问记录和哈希。完成风险层及完整校准选择并冻结后，才实现/启用评测解封。未取得新实验结果前不填论文效果表。', '']
    report = OUTPUT/'report.zh-CN.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    sources = [ROOT/'pems_bay_reader.py', ROOT/'pems_bay_windows.py', ROOT/'bind_pems_bay.py',
               ROOT/'audit_pems_bay_reader.py', Path(__file__).resolve(), ROOT/'fetch_pems_bay.py',
               OUTPUT/'binding.json', OUTPUT/'reader_audit.json', report,
               ROOT/'configs/pems_bay_validation_v1.json', ROOT/'outputs/pems-bay-validation-plan/design_freeze.json']
    frozen = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'stage': 'reader_implementation_only', 'dataset_sha256': binding['dataset_sha256'],
              'evaluation_values_read': False, 'model_training_executed': False,
              'calibration_executed': False, 'validation_executed': False,
              'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}}
    manifest = OUTPUT/'reader_implementation_freeze.json'
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding='utf-8'))
        if prior['source_sha256'] != frozen['source_sha256']:
            raise ValueError('Frozen reader implementation changed; use a new explicit version.')
    else:
        manifest.write_text(json.dumps(frozen, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'reader_milestone': 'passed', 'training_station_availability': quality['minimum'],
                      'evaluation_values_read': False, 'model_training_executed': False,
                      'report': str(report)}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
