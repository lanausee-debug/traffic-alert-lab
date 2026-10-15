# 从张量演示到真正训练

这个例子用 **CPU、小型 MLP、合成交通速度** 跑通训练、独立校准和事件评测。不下载 PEMS-BAY，也不读取原论文权重。

在已安装研究依赖的环境：

```bash
python train_tiny_alert.py --output runs/tiny-first
```

仅想运行小教程，可以单独安装 CPU PyTorch（已验证版本为2.8.0）：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python train_tiny_alert.py --output runs/tiny-first
```

[Notebook](../notebooks/tiny_training.ipynb) 可在本地 Jupyter 打开，也提供 Colab 入口。Notebook 的各代码单元在本机 CPU 实际执行过；云端 Colab 环境尚未执行验证。首次远程运行会下载公开仓库；缺少 PyTorch 时从官方 CPU 软件源安装。

## 你学到哪些 PyTorch 内容

1. **`torch.tensor`**：每条输入为24个数，前12个是历史速度，后12个是观测标记。速度缺失时填0，并由标记解释这个0。输入只到 `t-1`。
2. **`nn.Sequential` / `Linear` / `ReLU`**：构建 `24→32→16→1` 的风险 MLP，共1345个参数。输出是 logit，经过 `sigmoid` 才是0到1的分数。
3. **`BCEWithLogitsLoss`**：训练目标是当前及接下来两步是否出现低速，未来真值只用于目标。`pos_weight` 从训练集计算，用来调整正样本损失的权重。
4. **`zero_grad → backward → step`**：清空旧梯度、反向传播、更新参数。默认120次更新；脚本核对参数确实改变，和原来的零次更新张量课有明确区别。
5. **`model.eval()` / `torch.no_grad()`**：训练结束后进行校准与推理。独立合成时间序列使用种子11、22、33，不跨集合复用窗口。
6. **校准和测试的区别**：只在校准集比较五个阈值的逐点F1，然后固定阈值，在测试序列输出事件报告；不根据测试结果再改阈值。

输出目录包含 `training.json`（损失曲线、实际更新次数和配置）、`predictions.csv`、`report.json` 和 `report.html`。不会保存模型权重。每次使用新目录。

## 如何理解结果

损失下降说明这个网络学会了一些模拟器模式；事件检出、提前召回与误报仍可能相互牵制。合成数据非常简化，不能证明真实交通效果，更不能替代原论文失败的联合主检验。原论文前端和风险头的训练流程保持原样。

英文快速入口：`python train_tiny_alert.py --output runs/tiny-first` trains a CPU MLP for 120 optimizer updates on independent synthetic train/calibration/evaluation timelines. Read `training.json` and open `report.html`. These are teaching results, not a traffic benchmark or the frozen thesis experiment.
