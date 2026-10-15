# Traffic Alert Lab

交通速度观测缺失下的PyTorch预测、风险校准与事件告警实验。

[English](README.md) · [在线演示](https://lanausee-debug.github.io/traffic-alert-lab/) · [离线查看](demo/index.html) · [从零运行](docs/pems_bay_pipeline_reproduction.zh-CN.md) · [PyTorch六步学习](docs/undergraduate_pytorch_learning.zh-CN.md)

**预定联合主检验未通过。** 起点MLP误报较少，但含迟报检测明显下降，当前没有建立原创架构或整体服务优势。

## 三个入口

1. 下载后双击 `demo/index.html`，不安装依赖，切换三种输入情景查看七种方法的汇总结果。页面离线运行，不含真实逐站时间序列。
2. Python3.12运行 `python verify_release.py` 和 `python build_thesis_experiment_draft.py --check`，核对包内来源、冻结数值与表格。第一条会按SHA与字节数还原三份压缩的大型参考JSON，拒绝覆盖已改动的记录。该核对不重建原始事件。
3. 安装研究依赖后运行 `python undergraduate_torch_demo.py --output runs/torch-lesson.json`，学习张量、mask与BCE梯度。练习随机初始化，只用合成输入，优化器更新为0，不把教学损失当研究成绩。

## 原实验结果

| 细网格方法 | ≥10分钟代理预警召回 | 含迟报检测 | 误报/有效站日 |
| --- | --- | --- | --- |
| 全局校准 | 8.92% | 80.39% | 0.187396 |
| 可用性分组校准 | 8.47% | 80.14% | 0.179023 |
| 起点逻辑回归 | 10.85% | 36.80% | 0.096126 |
| 起点MLP | 11.89% | 28.18% | 0.086581 |

24站子集、83日、673个完整低速代理事件，三情景等权平均。三个情景复用同一事件真值，不能当三倍独立样本。起点MLP相对全局的R10差值为+2.97个百分点，97.5%区间[−0.20,5.89]，包含零；相对分组为+3.42，[0.36,6.29]。预定两项下界都大于零的联合要求未通过，检测比全局下降52.20个百分点。

代理提前量使用 `L=5(s−a+1)`，尚未独立核验真实拥堵起点。完整七方法结果见[附表](docs/thesis_result_tables.zh-CN.md)。

## 从零复现

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-pilot.txt --extra-index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe restore_reference_records.py
& .\.venv\Scripts\python.exe run_pems_bay_pipeline.py --workspace runs/my-first-run --data C:/path/to/pems-bay.h5
```

原数据由使用者依照上游入口取得，不随仓库分发。统一入口在新目录执行18阶段，支持 `--resume` 和 `--verify-only`。已有一次本机完整流程与原结果数值一致，约22分钟是当时记录，不是运行时间保证。重跑已分析数据是工程复现，不能称新的独立确认。

项目代码和原创文档采用[MIT许可证](LICENSE)，数据与第三方材料保留原范围。详见 [CODE_LICENSE.md](CODE_LICENSE.md)、[数据范围](DATA_POLICY.md)和[来源说明](THIRD_PARTY_NOTICES.md)。学校论文、Word/PPT、真实回放窗口、数据、权重与缓存不在本公开包内。历史记录中的“尚未上传”等表述按其记录日期理解。

## 在线部署

[直接打开在线演示](https://lanausee-debug.github.io/traffic-alert-lab/)，无需安装依赖。GitHub Pages展示同一份已核对的汇总结果，不包含逐站速度窗口，也不在线运行模型。main分支后续更新会触发部署，实际状态以Actions为准。
