# LocalGP · 多智能体信息采集

在不知道用户位置的情况下，让多架空中基站协同感知、探索并提供通信服务。

仓库保留 2024 年毕业设计代码和实验存档。新版 `localgp/` 从信息边界、实际航程、通信目标和可复现性出发重建训练闭环，支持在 Apple Silicon Mac 上使用 **MPS** 训练。

> 新版是对原问题的可审计实验实现，不宣称精确复现论文，也不把新结果与旧版曲线直接拼接。建模假设、奖励的局限和验证标准见 [第一性原理说明](docs/first-principles.md)。

## 在 Mac 上训练

Python 3.10 或更新版本。以下命令均在仓库根目录执行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[training,test]'

# 先验证前向、反向、评估与检查点的完整流程
python -m localgp.train --config configs/smoke.json --output output/mps-smoke

# 默认 3 架基站、50 位用户、24×24 信念图、4096 个环境步
python -m localgp.train --config configs/mac.json --output output/mps-seed7
```

显式指定 `mps` 时，设备不可用会报错。可以用 `--device cpu` 运行 CPU 对照，或用 `--device auto` 自动选择 MPS、CUDA、CPU。GP 分解留在 CPU；神经网络前向、反向和优化器更新使用所选设备。随机采样噪声在 CPU 生成，随后传到所选设备。

配置中记录所有环境和训练参数，包括通信功率、噪声、门限、移动速度、航程和奖励权重。使用 `--seed` 更换训练种子；`--steps` 必须是 rollout 长度的整数倍。实验目录必须为空，避免覆盖已有证据。

## 恢复训练与独立评估

```sh
# 同一训练继续至 8192 个环境步，并写入新目录
python -m localgp.train --config configs/mac.json --steps 8192 \
  --resume output/mps-seed7/last.pt --output output/mps-seed7-resumed

# 独立测试种子，配对比较训练策略、随机、感知贪心、原地不动
python -m localgp.evaluate output/mps-seed7/best.pt \
  --device mps --episodes 8 --seed 200000 --output output/mps-evaluation

# 生成本次实验的离线逐帧回放与对照报告
python tools/build_training_report.py output/mps-evaluation
```

打开 `output/mps-evaluation/report.html`。它显示本次评估真实记录的 GP 地图、位置、测量、奖励、覆盖率与吞吐量。真实用户图层仅供评估查看，不会进入策略输入。

检查点保存网络、优化器、环境、随机数和未结束回合状态。恢复时只允许改变总训练步数和设备；跨平台或不同 PyTorch 版本不保证逐位一致。`last.pt` 用于继续训练，`best.pt` 按固定验证场景的平均吞吐量选择；独立测试使用另一组种子。

## 实验产物

每个训练目录包含：

- `manifest.json`：完整配置、设备、源提交、源文件哈希、依赖版本和恢复来源。
- `metrics.jsonl`：每次更新的损失、策略变化、梯度、耗时、回合结果和验证结果。
- `last.pt` / `best.pt`：可恢复的最新和最佳检查点。
- `evaluation-*.json`：固定验证场景的逐场景指标及汇总。

独立评估目录包含配对对照 `comparison.json`、训练与贪心策略的逐帧记录，以及可生成的 `report.html`。输出目录不进入 Git；代码、配置、测试和精简的实验结论进入 Git。一次训练的多个测试场景不等于多个独立训练种子，不能据此宣称统计显著性。

## 测试

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

测试覆盖 GP 数值正确性、真实状态不进入 Actor、随机数隔离、完整路径碰撞、实际预算、GAE 终止与截断、PPO 概率一致性、断点恢复和导出。MPS 测试仅在设备可用时执行；GitHub Actions 在 CPU 上验证 Python 3.10 和 3.12。历史验证记录见 [validation.md](docs/validation.md)。`requirements-verified.txt` 为前一轮环境记录，当前训练环境另行保存版本清单。

## 查看 2024 年存档

无需安装科学计算或训练依赖：

```sh
python3 tools/build_replay.py
```

打开 `output/replay.html`，可查看 45 组完整的三机历史轨迹，播放、拖动时间轴、查看可用的最终 GP 快照并导出带 SHA-256 来源的 JSON。导出器自动发现实验，不写死数量。

历史记录缺少逐帧用户位置、配置和完整 GP 地图。因此旧回放只显示存档中已有的信息：最终地图保持静态，不伪造中间预测或通信指标。导航底图取自当前仓库，不能确认每轮实验都使用了同一版本。

## 代码结构

- `localgp/config.py`：带校验的环境与训练配置。
- `localgp/environment.py`、`radio.py`、`gp.py`：连续移动、信道与服务指标、有界局部 GP。
- `localgp/model.py`、`rollout.py`、`trainer.py`：独立 Actor/Critic、逐智能体 GAE、PPO 与检查点。
- `localgp/evaluation.py`：固定种子配对评估及观测信息范围内的基线。
- `configs/`：Mac 训练与快速验证配置。
- `forth/`：保留的旧版实现；GP 修复仍有独立回归测试，旧训练入口仍有已记录的问题。
- `forth/path/`、`mu_sig_map/`、`log/`：保持不变的历史存档。
- `tools/`：两种离线回放导出器。

## 论文与项目演进

H. Liu et al., *Autonomous Deployment of Aerial Base Station Without Network-Side Assistance in Emergency Scenarios Based on Multi-Agent Deep Reinforcement Learning*, IEEE TNSM, vol. 23, 2026. DOI：[10.1109/TNSM.2025.3603875](https://doi.org/10.1109/TNSM.2025.3603875)。

[中文论文对照与路线图](docs/research-notes.md)记录了旧实现与论文的差异。提供的论文 PDF 仅用于对照，没有重新分发。
